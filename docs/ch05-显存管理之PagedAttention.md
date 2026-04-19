# 课时5 - vLLM核心组件-显存管理之PagedAttention

> **关键源码文件**：
> - `vllm/vllm/v1/core/kv_cache_manager.py` -- `KVCacheManager`，负责按请求分配/回收 KV Cache blocks
> - `vllm/vllm/v1/core/kv_cache_coordinator.py` -- `KVCacheCoordinator`，协调多个 KV cache group 的统一分配
> - `vllm/vllm/v1/core/single_type_kv_cache_manager.py` -- 单种 attention 类型的 block 管理逻辑
> - `vllm/vllm/v1/core/block_pool.py` -- `BlockPool`，管理所有物理 block、prefix cache 与空闲队列
> - `vllm/vllm/v1/core/kv_cache_utils.py` -- `KVCacheBlock` 和 `FreeKVCacheBlockQueue` 等底层数据结构

## 学习目标

1. 理解传统连续显存分配策略为什么不适合大模型在线推理
2. 掌握 vLLM 把 KV Cache 拆成固定大小 block 的核心思路
3. 理解 `KVCacheManager -> KVCacheCoordinator -> SingleTypeKVCacheManager -> BlockPool` 四层分工
4. 掌握 `allocate_slots()` 如何把“本轮要算多少 token”翻译成“要申请多少 block”
5. 理解 `BlockPool` 如何完成 block 分配、命中、驱逐、回收，以及 prefix cache 的维护

---

## 5.1 前言

### 原理讲解

PagedAttention 在工程上最核心的价值，不是某个 attention kernel 的细节，而是它改变了 **KV Cache 的组织方式**。

如果不用分页思路，在线推理会遇到三个老问题：

1. **请求长度不可预知**：你不知道一个请求最终会生成多少 token
2. **连续显存难扩容**：请求长了就要扩容，扩容就要搬迁
3. **并发时碎片严重**：大量长短不一的请求同时进出，会让显存越来越难管理

vLLM 的做法是把 KV Cache 视为很多个固定大小的 block，然后按请求按需拼接这些 block。这个思路和操作系统的分页内存非常像，所以可以把它理解为“KV Cache 的分页管理”。

### 源码解析

在当前 V1 实现里，这套管理不是单个类完成的，而是四层协作：

- `KVCacheManager`：站在调度器视角，负责“这个请求还能不能再拿 block”
- `KVCacheCoordinator`：站在多种 KV cache group 视角，负责“不同 attention group 怎么一起分”
- `SingleTypeKVCacheManager`：站在单类 attention 视角，负责“一个 request 需要几块”
- `BlockPool`：站在物理 block 视角，负责“全局 free/cached/evict 的底层管理”

### 代码示例

下面用一个最小例子说明“分页”和“连续大块”这两种思路的差别：

```python
class ContiguousKV:
    def __init__(self):
        self.buffers = {}

    def grow(self, req_id, total_tokens):
        self.buffers[req_id] = [0] * total_tokens


class PagedKV:
    def __init__(self, block_size=4):
        self.block_size = block_size
        self.req_to_blocks = {}

    def grow(self, req_id, total_tokens):
        num_blocks = (total_tokens + self.block_size - 1) // self.block_size
        self.req_to_blocks[req_id] = [f"blk{i}" for i in range(num_blocks)]


contiguous = ContiguousKV()
contiguous.grow("req1", 13)
print(len(contiguous.buffers["req1"]))  # 13

paged = PagedKV(block_size=4)
paged.grow("req1", 13)
print(paged.req_to_blocks["req1"])      # ['blk0', 'blk1', 'blk2', 'blk3']
```

### 架构总览

```text
                 vLLM 的 KV Cache 分页管理分层
  ====================================================================

  Scheduler
     |
     v
  KVCacheManager
     |
     v
  KVCacheCoordinator
     |
     v
  SingleTypeKVCacheManager
     |
     v
  BlockPool
     |
     v
  KVCacheBlock / FreeKVCacheBlockQueue
```

---

## 5.2 传统策略及其缺陷

### 原理讲解

如果不用分页，最自然的做法通常有两种。

**方案一：每个请求一整块连续 buffer**

- 优点：访问简单
- 缺点：长度一旦增长，就要重新申请更大连续空间

**方案二：按最大长度预留**

- 优点：中途不需要扩容
- 缺点：绝大部分请求都用不到这么多空间，浪费严重

这两种方法在离线批处理里还能勉强接受，但在在线服务里会很痛苦：

- 请求长度动态变化
- 并发数持续波动
- 长短请求交错结束
- Prefix cache 希望跨请求复用已有 KV

一旦这些条件叠在一起，连续分配就会同时遇到：

- 外部碎片
- 内部浪费
- 扩容搬迁成本
- 难以做 prefix cache 复用

### 源码解析

虽然 vLLM 源码里不会专门写一个“传统策略实现”，但可以从它当前数据结构的选择反推出它在规避什么问题。

**1. block 是固定粒度，不是按请求直接申请一大片**

```python
# vllm/vllm/v1/core/kv_cache_utils.py :: KVCacheBlock
@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None
```

这说明最底层的显存单位是 block，而不是 request。

**2. block 的 free queue 明确支持中间删除**

```python
# vllm/vllm/v1/core/kv_cache_utils.py :: FreeKVCacheBlockQueue
"""
We implement this class instead of using Python builtin deque
to support removing a block in the middle of the queue in O(1) time.
"""
```

如果仍然是连续大块分配，就不会有这么强的“单块级别的移动与回收”需求。

**3. `BlockPool` 还维护 block hash -> block 的缓存映射**

```python
# vllm/vllm/v1/core/block_pool.py :: BlockPool.__init__
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

这正是 prefix cache 的基础。连续请求级分配很难在这么细的粒度上复用。

### 代码示例

```python
def contiguous_waste(max_len, real_len):
    return max_len - real_len


def paged_waste(block_size, real_len):
    used = ((real_len + block_size - 1) // block_size) * block_size
    return used - real_len


print(contiguous_waste(max_len=128, real_len=37))  # 91
print(paged_waste(block_size=16, real_len=37))     # 11
```

这个例子不是要说明分页完全没有浪费，而是说明浪费被限制在“最后一个 block 的尾部”，不会膨胀成整段大 buffer 的预留浪费。

---

## 5.3 分页显存的管理办法

### 原理讲解

vLLM 的分页思路可以用一句话概括：

**把 KV Cache 看成大量固定大小 block，request 只保存自己引用了哪些 block。**

于是一个请求的 KV Cache 不再是：

- “一块连续显存”

而是：

- “一个 block 列表”

这样做立刻带来三个好处：

1. 请求增长时，只要追加 block，不需要整体搬迁
2. 请求结束时，只回收自己的 block，不影响别人
3. 前缀命中时，可以直接复用已有 block

### 源码解析

**1. 一个请求拿到的 blocks 被包装成 `KVCacheBlocks`**

```python
# vllm/vllm/v1/core/kv_cache_manager.py :: KVCacheBlocks
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]

    def get_block_ids(self, allow_none: bool = False):
        return tuple([blk.block_id for blk in group] for group in self.blocks)
```

这里为什么是 `tuple[Sequence[KVCacheBlock], ...]`？

- 因为 vLLM 不同 KV cache group 可能对应不同 attention 结构
- 所以不是简单的一维 block 列表，而是“每个 group 一组 block”

**2. `SingleTypeKVCacheManager` 维护 `req_id -> blocks`**

```python
# vllm/vllm/v1/core/single_type_kv_cache_manager.py :: __init__
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
self.num_cached_block: dict[str, int] = {}
```

这说明 request 和 block 之间是典型的“逻辑映射”关系，而不是“显存地址范围”关系。

**3. `KVCacheCoordinator` 负责把多个 single-type manager 协调起来**

```python
# vllm/vllm/v1/core/kv_cache_coordinator.py :: KVCacheCoordinator.__init__
self.single_type_managers = tuple(
    get_manager_for_kv_cache_spec(...)
    for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
)
```

所以分页不是只有一个 block list，而是：

- 每种 KV cache 规格都有自己的一套 block 视图
- 但底层物理 block 仍然统一来自 `BlockPool`

### 代码示例

```python
class RequestKV:
    def __init__(self):
        self.req_to_blocks = {}

    def bind(self, req_id, block_ids):
        self.req_to_blocks[req_id] = block_ids


kv = RequestKV()
kv.bind("req1", [3, 7, 8])
kv.bind("req2", [4, 5])
print(kv.req_to_blocks)
```

### 数据流图

```text
               请求视角下的分页 KV Cache
  ====================================================================

  req1 -> [block 3, block 7, block 8]
  req2 -> [block 4, block 5]
  req3 -> [block 9]

  这些 block 不要求连续
  只要求:
  - request 能记住自己的 block 列表
  - worker 能根据 block_ids 正确访问 KV 数据
```

---

## 5.4 如何分配显存

### 原理讲解

PagedAttention 的核心不是“有 block”这么简单，而是调度器如何知道：

- 本轮一个请求到底要新增多少 block
- 这些 block 该从哪里拿
- prefix cache 命中的 block 需不需要再申请

在 vLLM 里，这个逻辑主要落在 `KVCacheManager.allocate_slots()`。

它的思路可以拆成五步：

1. 先算本轮一共需要覆盖多少 token slot
2. 再算这些 token 对应多少 block
3. 对 prefix cache 命中的 block 执行 `touch`
4. 申请新的 block
5. 如果 caching 开着，再把新凑满的 block 写回 prefix cache

### 源码解析

**1. `allocate_slots()` 先把 token 需求转成 slot 需求**

```python
# vllm/vllm/v1/core/kv_cache_manager.py :: KVCacheManager.allocate_slots
num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
num_tokens_need_slot = min(
    num_computed_tokens + num_new_tokens + num_lookahead_tokens,
    self.max_model_len,
)
```

这里的 `num_tokens_need_slot` 不是“本轮新生成多少 token”，而是“这个请求当前总共需要多少可用 slot”。

**2. 让 coordinator 计算一共要新申请多少 block**

```python
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=request.request_id,
    num_tokens=num_tokens_need_slot,
    new_computed_blocks=new_computed_block_list,
    num_encoder_tokens=num_encoder_tokens,
)
```

这一步把“token 数”翻译成了“block 数”。

**3. free block 不够就直接返回 `None`**

```python
if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
    return None
```

这就是调度器里 `allocate_slots()` 失败后必须做抢占的根本原因。

**4. 对命中的 prefix blocks 执行 `touch`**

```python
if self.enable_caching:
    self.block_pool.touch(new_computed_block_list)
```

`touch` 的含义是：

- 这些 block 虽然可能原本在 free queue 里、可被驱逐
- 但本轮被新请求命中后，必须提升引用计数，避免被错误回收

**5. 申请新 block，再决定要不要缓存**

```python
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id, num_tokens_need_slot, num_encoder_tokens
)

num_tokens_to_cache = min(
    num_computed_tokens + num_new_tokens, request.num_tokens
)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

注意这里的缓存不是“本轮所有 token 都缓存”，而是只缓存已经稳定、可提交的 token。

### 代码示例

```python
import math


def allocate_slots(num_existing_blocks, num_tokens_need_slot, block_size, free_blocks):
    num_required_blocks = math.ceil(num_tokens_need_slot / block_size)
    num_new_blocks = num_required_blocks - num_existing_blocks
    if num_new_blocks > free_blocks:
        return None
    return [f"blk{i}" for i in range(num_new_blocks)]


print(allocate_slots(2, num_tokens_need_slot=13, block_size=4, free_blocks=3))
print(allocate_slots(2, num_tokens_need_slot=25, block_size=4, free_blocks=3))
```

### 架构总览

```text
              allocate_slots() 的核心计算过程
  ====================================================================

  request + num_new_tokens
       |
       v
  num_tokens_need_slot
       |
       v
  coordinator.get_num_blocks_to_allocate()
       |
       +--> free blocks 不够 -> return None
       |
       +--> 够用:
             1. touch prefix-hit blocks
             2. allocate_new_blocks()
             3. cache_blocks()
```

---

## 5.5 BlockPool释放、分配显存块

### 原理讲解

真正的“物理 block 池”由 `BlockPool` 管理。

它维护三类核心状态：

1. **所有 block 元数据**：`self.blocks`
2. **空闲/可驱逐队列**：`FreeKVCacheBlockQueue`
3. **prefix cache 哈希表**：`cached_block_hash_to_block`

所以 `BlockPool` 既是 allocator，又是 prefix cache 的元数据中心。

### 源码解析

**1. `BlockPool` 初始化时创建所有 block**

```python
# vllm/vllm/v1/core/block_pool.py :: BlockPool.__init__
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)
]
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

这说明 block 本身是预先建好的，后续分配只是“从池里拿/还”。

**2. `get_new_blocks()` 从 free queue 里弹出 block**

```python
# vllm/vllm/v1/core/block_pool.py :: BlockPool.get_new_blocks
ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
for block in ret:
    self._maybe_evict_cached_block(block)
    block.ref_cnt += 1
```

为什么分配时还要 `_maybe_evict_cached_block`？

- 因为 free queue 里的 block 不一定是“纯空白”
- 它可能只是引用计数已经降到 0、目前可被驱逐的 prefix cache block

**3. `touch()` 用于命中已有缓存 block**

```python
def touch(self, blocks):
    for block in blocks_per_group:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

这一步是 prefix cache 命中的关键：命中的 block 必须重新变成“被引用中”。

**4. `free_blocks()` 回收 block**

```python
def free_blocks(self, ordered_blocks):
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    self.free_block_queue.append_n(
        [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
    )
```

这里最关键的是：

- 不是所有 free 都立即删除元数据
- 只有 `ref_cnt == 0` 的 block 才重新回到 free queue

**5. `FreeKVCacheBlockQueue` 用双向链表支持 O(1) 删除中间节点**

```python
# vllm/vllm/v1/core/kv_cache_utils.py :: FreeKVCacheBlockQueue
"""
support removing a block in the middle of the queue in O(1) time
"""
```

这正是为什么它不用 Python 自带 `deque`，而要手写一个双向链表队列。

### 代码示例

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0


class SimpleBlockPool:
    def __init__(self, num_blocks):
        self.free_blocks = [Block(i) for i in range(num_blocks)]

    def get_new_blocks(self, n):
        blocks = self.free_blocks[:n]
        self.free_blocks = self.free_blocks[n:]
        for blk in blocks:
            blk.ref_cnt += 1
        return blocks

    def free(self, blocks):
        for blk in blocks:
            blk.ref_cnt -= 1
            if blk.ref_cnt == 0:
                self.free_blocks.append(blk)


pool = SimpleBlockPool(4)
b = pool.get_new_blocks(2)
print([x.block_id for x in b])
pool.free(b)
print([x.block_id for x in pool.free_blocks])
```

### 数据流图

```text
                 BlockPool 的 block 生命周期
  ====================================================================

  初始状态: 全部 block 在 free_block_queue
       |
       v
  get_new_blocks()
       |
       +--> 如果 block 曾经被缓存: _maybe_evict_cached_block()
       +--> ref_cnt += 1
       v
  请求持有 block
       |
       +--> prefix hit: touch()
       |
       +--> 请求结束: free_blocks()
             - ref_cnt -= 1
             - ref_cnt == 0 -> 回 free queue
       v
  可再次分配 / 可作为 prefix cache eviction candidate
```

---

## 总结

本章重点解释了 vLLM 如何把 PagedAttention 落到显存管理上：

1. **传统连续分配不适合在线推理**：它会同时遇到扩容搬迁、显存浪费和碎片化问题
2. **vLLM 采用固定大小 block 管理 KV Cache**：请求只保存 block 列表，不再依赖连续显存
3. **分页管理是分层实现的**：`KVCacheManager` 管请求、`Coordinator` 管 group、`SingleTypeManager` 管单类 attention、`BlockPool` 管物理 block
4. **显存分配入口是 `allocate_slots()`**：它把 token 需求翻译成 block 需求，并结合 prefix cache / free block 数决定能否继续调度
5. **`BlockPool` 是底层 allocator**：负责 block 分配、命中、驱逐、回收，以及 prefix cache 的哈希映射维护
6. **性能关键在 block 粒度复用**：请求增长时追加 block，请求结束时只回收自己的 block，前缀命中时直接复用已有 block

---

## 思考题

1. **为什么说分页显存管理把浪费限制在“最后一个 block 的尾部”？** 它相比整段连续预留最大的收益是什么？

2. **`KVCacheManager.allocate_slots()` 为什么先算 `num_tokens_need_slot`，再算 `num_blocks_to_allocate`？** 这两个量分别代表什么？

3. **prefix cache 命中后，为什么还需要对 block 执行 `touch()`？** 如果省略这一步，会在哪种情况下出错？

4. **`FreeKVCacheBlockQueue` 为什么要支持 O(1) 删除中间 block？** 这个需求是由哪条上层调用链逼出来的？

5. **设计题**：如果你要把 block 大小从固定值改成“按模型层动态变化”，你认为 `KVCacheManager`、`Coordinator`、`BlockPool` 三层里谁需要改动最大？为什么？
