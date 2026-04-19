# 课时15 - vLLM 分布式推理-EP 负载均衡

> **关键源码文件**：
> - `vllm/vllm/config/parallel.py` -- `EPLBConfig`、`enable_eplb`、`num_redundant_experts`
> - `vllm/vllm/distributed/eplb/eplb_state.py` -- EPLB 状态、负载窗口、重平衡触发与执行主流程
> - `vllm/vllm/distributed/eplb/policy/default.py` -- 默认 EPLB 策略，负责复制专家与重排映射
> - `vllm/vllm/distributed/eplb/rebalance_execute.py` -- 专家权重的本地搬运与跨 rank P2P 传输
> - `vllm/vllm/distributed/eplb/async_worker.py` -- 异步 EPLB 的后台传输线程
> - `vllm/vllm/model_executor/layers/fused_moe/layer.py` -- `set_eplb_state()`、`select_experts()` 中的物理专家映射与负载记录
> - `vllm/examples/online_serving/elastic_ep/serve_deepseek_v2.sh` -- EP + EPLB 在线服务示例
> - `vllm/examples/online_serving/elastic_ep/scale.py` -- Elastic EP / EPLB 缩扩容示例

## 学习目标

1. 理解 EPLB 解决的是什么问题，以及它和普通 EP 的边界在哪里
2. 掌握 logical expert、physical expert、redundant expert 这几个关键概念
3. 理解 `EplbState` 如何记录负载、维护专家映射，并在合适时机触发重排
4. 掌握 `DefaultEplbPolicy` 和 `rebalance_execute.py` 如何把“负载统计”变成“权重迁移”
5. 能够从源码角度解释同步 EPLB、异步 EPLB 和 Elastic EP 的主流程

---

## 15.1 EPLB 模块概述

### 原理讲解

专家并行解决的是：

**专家权重如何分布在多张卡上。**

但这还不够。

因为在真实请求中，Router 往往不会把 token 平均地发给每个专家。于是会出现：

- 某些专家特别热门
- 某些专家长期空闲
- 某些 GPU 上的专家总是更忙

这就是 EPLB（Expert Parallel Load Balancing）要解决的问题。

它的目标不是改 Router 算法，而是：

**根据最近一段时间的专家负载，重新安排物理专家的布局，并给热点逻辑专家增加冗余副本。**

### 源码解析

`eplb_state.py` 一开头就给出了这套概念的官方定义：

- **Logical Expert**：模型逻辑结构里的专家
- **Redundant Expert**：为负载均衡额外复制出来的专家副本
- **Physical Expert**：实际部署在某张设备上的专家实例
- **Local Physical Expert**：当前 rank 本地持有的物理专家

这几个概念是理解 EPLB 的前提。

例如一个模型有：

- 256 个逻辑专家
- 32 个冗余专家

那么某个 MoE 层里真正部署的就不是 256 套专家权重，而是：

```text
256 + 32 = 288 个 physical experts
```

**EPLB 改的是 physical expert 的布局，不是 logical expert 的语义。**

### 代码示例

下面用一个最小例子说明 logical / physical 的区别：

```python
logical_experts = [0, 1, 2, 3]
physical_to_logical = [0, 1, 2, 3, 0, 1]

print("logical experts:", logical_experts)
print("physical -> logical:", physical_to_logical)
```

这里：

- 逻辑专家还是 4 个
- 但物理专家已经是 6 个
- 其中逻辑专家 `0` 和 `1` 各多了一个副本

### 架构总览

```text
                   EPLB 要解决的问题
========================================================================

逻辑专家负载不均
   |
   v
某些物理 rank 更忙
   |
   v
记录一段时间的专家负载
   |
   v
重新计算:
- 哪些逻辑专家需要复制
- 这些物理专家应该放在哪些 rank
   |
   v
搬迁专家权重并更新映射表
```

---

## 15.2 专家并行 + EPLB 参数和服务实例

### 原理讲解

EPLB 不是单独存在的功能，它依赖于 EP。

换句话说：

- 没有 `enable_expert_parallel`
- 就不存在 `enable_eplb`

所以在使用层面，EPLB 要看两类参数：

1. **是否启用 EP / EPLB**
2. **负载均衡策略如何工作**

### 源码解析

**1. `ParallelConfig` 和 `EPLBConfig` 定义了核心参数**

```python
class ParallelConfig:
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    all2all_backend: ...
```

而 `EPLBConfig` 里最关键的字段是：

```python
class EPLBConfig:
    window_size: int = 1000
    step_interval: int = 3000
    num_redundant_experts: int = 0
    log_balancedness: bool = False
    use_async: bool = False
    policy: "default"
```

可以把它们理解成：

- `window_size`：用多长的历史窗口统计负载
- `step_interval`：多少 step 触发一次重平衡
- `num_redundant_experts`：允许增加多少冗余专家
- `use_async`：是否后台异步迁移专家

**2. 在线服务示例里，EP 和 EPLB 是一起打开的**

`examples/online_serving/elastic_ep/serve_deepseek_v2.sh` 里直接用了：

```bash
vllm serve $MODEL_NAME \
    --data-parallel-size $DATA_PARALLEL_SIZE \
    --enable-expert-parallel \
    --enable-eplb \
    --num-redundant-experts $REDUNDANT_EXPERTS
```

这段脚本的价值很高，因为它把实际部署场景说明白了：

- EP 负责把专家切到多张卡
- EPLB 负责在这个基础上做副本与重排

**3. `scale.py` 展示了 Elastic EP 的服务侧入口**

```python
url = f"http://{host}:{port}/scale_elastic_ep"
payload = {"new_data_parallel_size": new_dp_size}
response = requests.post(url, json=payload, ...)
```

这说明 vLLM 不只是支持“静态的 EPLB”，还支持随着服务规模变化，重新计算 EP/EPLB 布局。

### 代码示例

下面给出一个典型的启动命令示意：

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --data-parallel-size 4 \
  --data-parallel-backend ray \
  --enable-expert-parallel \
  --enable-eplb \
  --num-redundant-experts 8
```

如果需要触发缩扩容，可以调用示例脚本：

```bash
python vllm/examples/online_serving/elastic_ep/scale.py \
  --host localhost \
  --port 8006 \
  --new-dp-size 2
```

### 参数关系图

```text
                EP + EPLB 配置关系
========================================================================

enable_expert_parallel = True
   |
   +--> all2all_backend
   +--> expert_placement_strategy
   |
   +--> enable_eplb = True
           |
           +--> window_size
           +--> step_interval
           +--> num_redundant_experts
           +--> use_async
           +--> policy
```

---

## 15.3 vLLM EPLB 设计方案

### 原理讲解

EPLB 的设计可以分成两部分：

1. **状态设计**：怎么记录谁是谁、谁有几个副本、最近谁最忙
2. **策略设计**：如何根据这些统计结果计算新布局

vLLM 这两部分分别落在：

- `EplbState`
- `DefaultEplbPolicy`

### 源码解析

**1. `EplbState.add_model()` 初始化三张核心映射表**

初始化阶段最重要的三个张量是：

```python
physical_to_logical_map
logical_to_physical_map
logical_replica_count
```

它们分别表示：

- 物理专家 -> 逻辑专家
- 逻辑专家 -> 所有物理副本
- 每个逻辑专家当前有多少副本

这是 EPLB 的核心状态。

**2. 同时还会初始化负载统计窗口**

```python
expert_load_pass = torch.zeros(...)
expert_load_window = torch.zeros(
    (window_size, num_moe_layers, num_physical_experts), ...
)
```

这里分成两层统计：

- `expert_load_pass`：当前 forward pass 的负载
- `expert_load_window`：滑动窗口历史

也就是说，EPLB 不是看单步尖峰，而是看一段时间的统计趋势。

**3. `FusedMoE.set_eplb_state()` 把状态注册到具体 MoE 层**

```python
self.expert_load_view = expert_load_view[moe_layer_idx]
self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
self.logical_replica_count = logical_replica_count[moe_layer_idx]
```

一旦注册进去，MoE 层在 forward 时就知道：

- 当前层有哪些逻辑专家
- 每个逻辑专家对应哪些物理副本
- 当前 step 的负载应该记到哪里

**4. `select_experts()` 里会做“逻辑专家 -> 物理专家”改写**

```python
topk_ids = eplb_map_to_physical_and_record(
    topk_ids=topk_ids,
    expert_load_view=self.expert_load_view,
    logical_to_physical_map=self.logical_to_physical_map,
    logical_replica_count=self.logical_replica_count,
)
```

这一步一箭双雕：

1. 把路由结果从逻辑专家改成具体物理专家
2. 顺手把当前 token 负载记到 `expert_load_view`

这意味着 EPLB 不是在 forward 之后额外扫一遍日志，而是在路由阶段就自然地把统计拿到了。

**5. `DefaultEplbPolicy` 根据负载重新计算映射**

`default.py` 里最关键的几个步骤是：

- `balanced_packing()`：尽量把不同负载的对象均匀装箱
- `replicate_experts()`：给热点逻辑专家增加物理副本
- `rebalance_experts_hierarchical()`：优先考虑 node / GPU 分层放置
- `rebalance_experts()`：作为总入口产出新的 `phy2log / log2phy / logcnt`

可以把它理解成：

**先决定哪些逻辑专家值得复制，再决定这些副本应该落在哪些节点和 GPU。**

### 代码示例

下面用简化版映射说明 EPLB 的状态含义：

```python
physical_to_logical = [0, 1, 2, 3, 0, 1]
logical_to_physical = {
    0: [0, 4],
    1: [1, 5],
    2: [2],
    3: [3],
}

logical_replica_count = {k: len(v) for k, v in logical_to_physical.items()}

print(physical_to_logical)
print(logical_to_physical)
print(logical_replica_count)
```

### 状态图

```text
                 EPLB 的核心状态设计
========================================================================

router top-k ids
   |
   v
logical_to_physical_map + logical_replica_count
   |
   v
physical expert ids
   |
   +--> 执行 forward
   +--> expert_load_pass 记本轮负载
   |
   v
expert_load_window
   |
   v
DefaultEplbPolicy.rebalance_experts()
   |
   v
new physical_to_logical_map / logical_to_physical_map
```

---

## 15.4 vLLM EPLB 流程解析

### 原理讲解

EPLB 的完整流程可以概括成：

1. 每轮 forward 记录专家负载
2. 把负载写入滑动窗口
3. 到达 `step_interval` 后触发重平衡
4. 计算新的专家映射
5. 搬运专家权重
6. 更新各层映射状态

这里最复杂的地方不是“算一个新映射”，而是：

**如何把新映射真正落到权重和运行态上。**

### 源码解析

**1. `step()` 负责驱动整个状态机**

`EplbState.step()` 主要做四件事：

1. 可选地同步 / 打印 balancedness 指标
2. 把 `expert_load_pass` 写入 `expert_load_window`
3. 递增 `expert_rearrangement_step`
4. 到阈值时调用 `rearrange()`

其中负载日志的统计方式很清楚：

```python
num_tokens_per_rank = expert_load_pass.reshape(...).sum(dim=-1).float()
avg_tokens_tensor = num_tokens_per_rank.mean(dim=0).sum(dim=0)
max_tokens_tensor = num_tokens_per_rank.max(dim=0).values.sum(dim=0)
balancedness = avg_tokens / max_tokens
```

也就是说，EPLB 的目标就是尽量提升这个 `balancedness`。

**2. `rearrange()` 先把 physical load 聚合回 logical load**

```python
logical_expert_load_window.scatter_add_(
    dim=-1,
    index=physical_to_logical_map.unsqueeze(0).expand_as(expert_load_window).long(),
    src=expert_load_window,
)
global_expert_load_window = logical_expert_load_window.sum(dim=0)
global_expert_load_windows = self._allreduce_list(global_expert_load_windows)
```

这是一个非常关键的步骤：

- forward 期间记录的是 physical expert 负载
- 但决策时需要知道 logical expert 热不热门

所以要先把所有物理副本的负载聚回逻辑专家视角，再做跨 rank all-reduce。

**3. 然后调用策略层生成新映射**

```python
new_physical_to_logical_map, new_logical_to_physical_map, new_logical_replica_count = (
    self.policy.rebalance_experts(...)
)
```

到这一步为止，还只是“算出来”新布局，并没有真正动权重。

**4. 同步路径：直接 `rearrange_expert_weights_inplace()`**

如果当前不是 async 模式：

```python
rearrange_expert_weights_inplace(
    eplb_model_state.physical_to_logical_map,
    new_physical_to_logical_map,
    eplb_model_state.model.expert_weights,
    ep_group,
    is_profile,
    rank_mapping,
)
```

`rebalance_execute.py` 里又把这件事拆成两段：

- `move_to_buffer()`：先本地复制，再通过 `batch_isend_irecv` 发 / 收权重
- `move_from_buffer()`：把 buffer 中的新权重放回工作区

这说明 EPLB 的本质不是“改几个索引”，而是真实地迁移专家参数。

**5. 异步路径：后台线程先搬到 buffer，主线程再切换**

如果开启 `use_async=True`：

- `rearrange()` 只会把 `new_*_map` 存下来，并设置 `rebalanced=True`
- `start_async_loop()` 会启动 `async_worker.py` 里的后台线程
- 后台线程调用 `transfer_layer()`，逐层把专家权重搬到 buffer
- 主线程在 `move_to_workspace()` 中等待 `buffer_ready_event`，再把对应层切换过去

这条链路的意义是：

**把最重的权重迁移过程尽量放到后台，降低对推理主路径的阻塞。**

**6. Elastic EP 场景下还要额外处理 `rank_mapping`**

在缩容或扩容时，老 rank 和新 rank 不一定一一对应。

因此 `rebalance_execute.py` 里提供了：

- `_map_old_expert_indices_with_rank_mapping()`
- `_map_new_expert_indices_with_rank_mapping()`

用于把旧布局和新布局对齐到统一索引空间。

### 代码示例

下面用一个极简版本表示 EPLB 的主循环：

```python
load_window.append(current_step_load)

if len(load_window) >= window_size and step % step_interval == 0:
    logical_load = aggregate_physical_to_logical(load_window)
    new_mapping = rebalance_policy(logical_load)
    move_weights(old_mapping, new_mapping)
    old_mapping = new_mapping
```

真实 vLLM 实现比这个复杂很多，但骨架基本就是这几步。

### 完整流程图

```text
                    EPLB 完整主流程
============================================================================

FusedMoE.select_experts()
   |
   +--> eplb_map_to_physical_and_record()
   |      - topk logical ids -> physical ids
   |      - 记录 expert_load_pass
   |
   v
EplbState.step()
   |
   +--> 写入 expert_load_window
   +--> 累加 rearrangement_step
   +--> 到达阈值后调用 rearrange()
   |
   v
rearrange()
   |
   +--> physical load -> logical load
   +--> all-reduce global load
   +--> DefaultEplbPolicy.rebalance_experts()
   |
   +--> 同步模式:
   |      rearrange_expert_weights_inplace()
   |
   +--> 异步模式:
          async_worker.transfer_layer()
          -> move_to_workspace()
          -> post_eplb()
```

---

## 总结

1. EPLB 解决的是 EP 下专家负载长期不均衡的问题，它通过增加冗余专家和重排物理布局来改善 balancedness。
2. `EplbState` 是整个模块的中枢，维护了映射表、滑动窗口、重排步数和异步迁移状态。
3. `FusedMoE.select_experts()` 在路由阶段就把逻辑专家映射到物理专家，并同步记录每步负载。
4. `DefaultEplbPolicy` 负责根据逻辑专家负载计算新布局，`rebalance_execute.py` 负责真正搬迁专家权重。
5. 同步 EPLB 简单直接，但会阻塞主路径；异步 EPLB 更复杂，但能把权重迁移分摊到后台线程。
6. Elastic EP 进一步把 EPLB 从“静态重平衡”扩展到了“伴随服务规模变化的动态重平衡”。

## 思考题

1. 为什么 EPLB 在决策时要先把 physical expert 的负载聚合回 logical expert 视角？
2. `num_redundant_experts` 提高之后，为什么不一定线性提升负载均衡效果？
3. 异步 EPLB 为什么需要 `buffer_ready_event`、`buffer_lock` 和 `layer_to_transfer` 这些额外状态？
4. 在什么情况下，同步 EPLB 可能比异步 EPLB 更合适？
5. 结合第14章和本章，说明 EP 解决的是“专家如何分布”，而 EPLB 解决的是“分布之后如何长期保持均衡”。
