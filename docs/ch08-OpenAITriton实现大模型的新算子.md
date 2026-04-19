# 课时8 - OpenAI Triton实现大模型的新算子

> **关键源码文件**：
> - `vllm/vllm/v1/attention/backends/triton_attn.py` -- Triton attention 后端入口，负责 metadata 构建和 `TritonAttentionImpl.forward`
> - `vllm/vllm/attention/ops/triton_unified_attention.py` -- Triton unified attention kernel 主实现
> - `vllm/vllm/attention/ops/triton_reshape_and_cache_flash.py` -- 把 K/V 写入分页 KV Cache 的 Triton kernel
> - `vllm/vllm/triton_utils.py` -- Triton 与 `tl` 的统一导出入口
> - `vllm/vllm/v1/worker/gpu/attn_utils.py` -- attention backend 初始化与 metadata 构建入口

## 学习目标

1. 理解 attention 为什么会成为大模型推理中的核心性能瓶颈
2. 掌握 Triton 和 CUDA 的关系，以及 Triton kernel 的基本写法和编译触发方式
3. 能够通过向量加法、Softmax、Matmul 三个例子建立 Triton kernel 的基本直觉
4. 理解 attention kernel 设计里最关键的 tile、mask、softmax 累积与 KV cache 访问模式
5. 掌握 vLLM 中 `TritonAttentionBackend -> TritonAttentionImpl -> Triton ops` 的真实调用链

---

## 8.1 Attention 算子成为性能瓶颈的原因

### 原理讲解

在大模型推理里，attention 容易成为瓶颈，通常不是因为“只有它算得慢”，而是因为它同时踩中了三类高成本操作：

1. **算子复杂度高**：序列长度增长后，注意力分数计算和归一化成本快速上升
2. **访存压力大**：Q、K、V、KV Cache、mask、输出张量都要频繁读写
3. **动态形状复杂**：在线推理里 batch size、query len、seq len 都在变化

对 vLLM 来说，这个问题更复杂一些，因为它还要处理：

- 分页 KV Cache
- 不同请求长度混合 batch
- prefill 和 decode 共存
- FP8 KV cache、sliding window、MM prefix 等特性

所以 attention 后端不是“写一个矩阵乘法”那么简单，而是要把：

**计算、访存、分页寻址和动态调度一起考虑。**

### 源码解析

在 vLLM 的 Triton attention 后端里，这种复杂性直接体现在 `TritonAttentionImpl.forward()` 的输入上：

```python
# vllm/vllm/v1/attention/backends/triton_attn.py :: TritonAttentionImpl.forward
def forward(
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
    ...
)
```

这几个参数对应的不是一个“普通 attention 函数”，而是一个带完整运行时上下文的后端接口：

- `query / key / value`：本轮新 token 的 QKV
- `kv_cache`：历史上下文已经写入的分页缓存
- `attn_metadata`：query 起止位置、seq len、block table、slot mapping 等执行元数据
- `output`：预先分配好的输出 buffer

也就是说，vLLM 里的 attention 已经不是单纯的张量公式，而是一个 **带运行时状态的系统算子**。

### 代码示例

下面用一个最小估算例子说明 attention 为什么会随序列长度迅速变重：

```python
def approx_attention_work(num_tokens, head_size, num_heads):
    score_ops = num_heads * num_tokens * num_tokens * head_size
    value_ops = num_heads * num_tokens * num_tokens * head_size
    return score_ops + value_ops


for n in [128, 512, 1024]:
    print(n, approx_attention_work(n, head_size=128, num_heads=32))
```

这个估算很粗糙，但能帮助我们建立第一层直觉：

**序列一长，attention 不只是“慢一点”，而是会同时放大计算量和访存压力。**

---

## 8.2 OpenAI Triton 定义与核心特性

### 原理讲解

Triton 可以把它理解成一种“用 Python 写 GPU kernel”的方式。

它的目标不是替代 PyTorch，而是解决 PyTorch 很难细调、CUDA 又太重的中间地带：

- 你想控制 tile、warp、访存布局
- 你不想从头写一整套 CUDA C++ 扩展

Triton 的几个核心特性是：

1. **Python 语法写 kernel**
2. **显式控制 block / tile / program id**
3. **JIT 编译并缓存**
4. **对矩阵、向量、block 级操作表达力强**

因此它非常适合 attention、softmax、matmul、cache reshape 这类“结构规则但性能敏感”的内核。

### 源码解析

在 vLLM 里，Triton kernel 的入口形式非常直接：

```python
# vllm/vllm/attention/ops/triton_unified_attention.py
from vllm.triton_utils import tl, triton

@triton.jit
def kernel_unified_attention_2d(...):
    ...
```

这说明在 vLLM 里：

- `triton` 用来声明 kernel
- `tl` 提供 Triton language 原语，比如 `tl.load`、`tl.store`、`tl.dot`

再比如 `triton_reshape_and_cache_flash.py` 里的 cache 写回 kernel：

```python
@triton.jit
def reshape_and_cache_kernel_flash(...):
    ...
```

这类 kernel 本质上就是：

**把 PyTorch 张量和底层 GPU 访存布局之间的那层逻辑，直接写成可控的 GPU 程序。**

### 代码示例

一个最简单的 Triton kernel 通常长这样：

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

这里最值得注意的就三个点：

- `@triton.jit`：告诉 Triton 这是一个要编译的 kernel
- `tl.program_id(0)`：获取当前 program 的 block 索引
- `tl.arange(...)`：生成 block 内的局部索引

---

## 8.3 Triton 与 CUDA 的差异

### 原理讲解

Triton 和 CUDA 解决的是同一类问题，但抽象层次不一样。

如果用一句话概括：

- **CUDA 更底层**：线程块、warp、shared memory、同步都需要自己细管
- **Triton 更聚焦张量 tile**：你主要描述“这个 block 处理哪一片数据”

所以 Triton 的优势是开发效率高、表达更贴近张量计算；代价是某些极端细节控制不如手写 CUDA 自由。

### 源码解析

这种差异在 vLLM 的 Triton 代码里非常明显。

比如 `triton_reshape_and_cache_flash()` 的 launch 方式：

```python
# vllm/vllm/attention/ops/triton_reshape_and_cache_flash.py
grid = lambda meta: (
    slot_mapping.shape[0],
    triton.cdiv(n, meta["TILE_SIZE"]),
)

reshape_and_cache_kernel_flash[grid](...)
```

在 CUDA 里，你通常会显式写：

- gridDim.x / blockDim.x
- 线程索引计算
- 线程同步

而 Triton 更强调：

- 当前 program 负责哪块 tile
- tile 内的元素如何被 `tl.load / tl.store` 处理

再看 `kernel_unified_attention_2d()`：

```python
q_block_global_idx = tl.program_id(0)
kv_head_idx = tl.program_id(1)
```

这相当于把二维 launch grid 映射到：

- 第 0 维：query block
- 第 1 维：KV head

所以 Triton 更像在写“tile 程序”，而不是传统意义上的“线程程序”。

### 代码示例

下面用伪代码对比一下思路差异：

```python
# CUDA 思维：先想线程
thread_idx = blockIdx.x * blockDim.x + threadIdx.x
out[thread_idx] = x[thread_idx] + y[thread_idx]

# Triton 思维：先想 tile
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x_tile = tl.load(x_ptr + offsets)
y_tile = tl.load(y_ptr + offsets)
tl.store(out_ptr + offsets, x_tile + y_tile)
```

前者更接近 GPU 硬件线程，后者更接近张量块操作。

---

## 8.4 Triton 编译过程

### 原理讲解

Triton kernel 虽然是 Python 写的，但执行时不会逐行解释运行。

它的大致流程可以概括成：

1. Python 中定义 `@triton.jit` kernel
2. 第一次按某组参数 launch 时触发 JIT
3. Triton 根据 kernel 代码、dtype、shape、meta 参数生成底层代码
4. 编译结果缓存下来
5. 后续相同签名的调用直接复用已编译结果

这也是为什么 Triton kernel 常常表现出：

- 第一次调用慢一些
- 后续调用明显更快

### 源码解析

在 vLLM 里，编译触发点通常就藏在 kernel launch 语法里：

```python
reshape_and_cache_kernel_flash[grid](...)
```

或者更高一层的：

```python
unified_attention(
    q=query[:num_actual_tokens],
    k=key_cache,
    v=value_cache,
    ...
)
```

只要第一次遇到新的 kernel 签名，Triton 就会在底层做编译和缓存。

因此 vLLM 会在 warmup、dummy run、CUDA graph capture 这些环节尽量把常见路径先走一遍，本质上就是为了减少正式请求到来时的首次编译抖动。

### 代码示例

下面这个小例子能帮助理解“首次编译”和“后续复用”的区别：

```python
import time


def first_run():
    t0 = time.perf_counter()
    # 第一次 launch Triton kernel，可能触发 JIT
    t1 = time.perf_counter()
    return t1 - t0


def second_run():
    t0 = time.perf_counter()
    # 同一签名再次 launch，通常直接复用缓存
    t1 = time.perf_counter()
    return t1 - t0
```

真实环境里第一次 launch 和第二次 launch 的差异会比这个伪例子更明显。

---

## 8.5 第一个 Triton 算子：一维向量相加

### 原理讲解

学 Triton 最好的起点就是向量加法，因为它只保留了三件最核心的事：

1. 如何切 tile
2. 如何 load/store
3. 如何处理越界 mask

只要这三个概念建立起来，后面看 softmax、matmul、attention 都会顺很多。

### 源码解析

虽然 vLLM 没专门放一个“向量加法 demo”，但 `triton_reshape_and_cache_flash.py` 已经体现了同样的结构：

- `token_idx = tl.program_id(axis=0)`：确定当前 tile 处理哪个 token
- `tile_pos = tile_i * TILE_SIZE + tile_offs`：确定 tile 内局部位置
- `tl.load(..., mask=...)` / `tl.store(..., mask=...)`：处理越界和写回

所以向量加法不是和 vLLM 无关，它其实就是理解这些真实 kernel 的前置练习。

### 代码示例

```python
import torch
import triton
import triton.language as tl


@triton.jit
def vec_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vec_add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    vec_add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

这个例子里最重要的不是“加法”本身，而是：

- 一个 program 负责一段连续数据
- 越界由 mask 处理
- host 侧通过 `kernel[grid](...)` 触发执行

---

## 8.6 更复杂的 Softmax 算子（PyTorch vs Triton 实现）

### 原理讲解

Softmax 比向量加法复杂得多，因为它至少包含三步：

1. 求最大值，防止数值溢出
2. 指数化
3. 求和再归一化

attention 的核心之一就是对分数矩阵做 softmax，所以如果 Softmax 写不好，attention 性能和数值稳定性都很难好。

### 源码解析

在 `kernel_unified_attention_2d()` 里，Triton attention 并没有调用一个单独的 softmax API，而是把 softmax 逻辑嵌进 tile 循环里：

```python
m_j = tl.maximum(M, tl.max(S, axis=1))
P = tl.exp(S - m_j[:, None])
l_j = tl.sum(P, axis=1)
alpha = tl.exp(M - m_j)
acc = acc * alpha[:, None]
L = L * alpha + l_j
M = m_j
```

这段代码本质上是在做 **在线 softmax 累积**：

- `M`：当前累积最大值
- `L`：当前累积分母
- `acc`：当前累积输出

为什么要这么写？

因为 attention 往往不能一次把整条序列都装进片上资源里，需要分 tile 处理；分 tile 时又不能丢掉全局 softmax 的数值正确性，所以要用这种递推写法。

### 代码示例

先看 PyTorch 参考版：

```python
import torch


def softmax_torch(x):
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

再看一个教学版 Triton 思路：

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptrs = x_ptr + row_id * stride + cols
    mask = cols < n_cols
    x = tl.load(ptrs, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row_id * stride + cols, num / den, mask=mask)
```

这个版本还不够真实 attention 那么复杂，但已经能体现 Triton 版 Softmax 的基本骨架。

---

## 8.7 实现 Matmul 算子

### 原理讲解

Matmul 是 attention 的底座之一：

- `Q @ K^T` 得到 attention score
- `P @ V` 得到最终输出

所以理解 Triton matmul，本质上是在为 attention 做准备。

Matmul kernel 最核心的思想是：

1. 把输出矩阵切成多个 tile
2. 每个 program 负责一个输出 tile
3. 沿着 K 维分块累积

### 源码解析

`kernel_unified_attention_2d()` 中的两次 `tl.dot` 就体现了这种思路：

```python
S += scale * tl.dot(Q, K)
acc += tl.dot(P.to(V.dtype), V)
```

这里的含义分别是：

- `Q x K`：算当前 tile 的注意力分数
- `P x V`：算当前 tile 对输出的贡献

所以 attention kernel 可以看成：

**在更复杂 mask 和 softmax 约束下的两次分块 matmul。**

### 代码示例

```python
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
```

这个示例省略了不少边界处理，但核心思路已经够用了：**输出按 tile 切，K 维按块累积。**

---

## 8.8 实现 Attention 算子

### 原理讲解

从公式上看，attention 就三步：

1. `S = QK^T`
2. `P = softmax(S)`
3. `O = PV`

但真实实现比公式复杂得多，因为还要同时处理：

- causal mask
- sliding window
- prefix / multimodal 范围放宽
- 分页 KV Cache
- 数值稳定的在线 softmax
- 不同请求长度的混合 batch

所以一个能进生产的 attention kernel，核心难点不是公式本身，而是：

**如何在 tile 粒度下把这些约束一起融进去。**

### 源码解析

`kernel_unified_attention_2d()` 已经把这件事写得很完整了。

它的大致流程是：

1. 根据 `program_id` 找到当前 query block 和 KV head
2. 通过 `block_table` 找到逻辑序列对应的物理 KV blocks
3. 对每个 tile：
   - load `K` / `V`
   - 计算 `S = QK^T`
   - 应用 causal / sliding window / MM prefix mask
   - 更新在线 softmax 状态
   - 累积 `P @ V`
4. 写回输出

这也是为什么 vLLM 的 attention kernel 参数非常多：它不是一个“纯数学 kernel”，而是一个带运行时元数据和分页寻址的系统 kernel。

### 代码示例

下面给一个教学版 attention 骨架，省略了分页 KV Cache，但保留最核心的结构：

```python
@triton.jit
def toy_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                         stride_qm, stride_qk,
                         stride_kn, stride_kk,
                         stride_vn, stride_vk,
                         stride_om, stride_ok,
                         seq_len, head_size,
                         BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.ones((BLOCK_M,), dtype=tl.float32)

    for n in range(0, seq_len, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk)
        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        s = tl.dot(q, k)
        s = tl.where(offs_n[None, :] <= offs_m[:, None], s, float("-inf"))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    acc = acc / l[:, None]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, acc)
```

这个教学版和 vLLM 真正的 `unified_attention` 还差很远，但它已经把：

- 分块 matmul
- causal mask
- 在线 softmax
- 输出累积

这四个核心思想串起来了。

---

## 8.9 vLLM 中的 Triton Attention 后端

### 原理讲解

前面的例子解决的是“怎么写 Triton kernel”，但 vLLM 真正关心的是：

**怎么把 Triton kernel 放进完整推理系统里。**

在 vLLM 里，Triton attention 后端可以概括成三层：

1. `TritonAttentionMetadataBuilder`：构建执行元数据
2. `TritonAttentionImpl.forward()`：组织输入并调用 Triton ops
3. `triton_reshape_and_cache_flash()` + `unified_attention()`：真正的 Triton kernel / wrapper

### 源码解析

**1. metadata builder 先把后端需要的执行信息准备好**

```python
# vllm/vllm/v1/attention/backends/triton_attn.py
self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv
...
attn_metadata = TritonAttentionMetadata(
    query_start_loc=query_start_loc,
    seq_lens=seq_lens,
    block_table=block_table_tensor,
    slot_mapping=slot_mapping,
    ...
)
```

这里做了两件非常重要的事：

- 决定 2D / 3D kernel 切换阈值
- 把 query 起点、序列长度、block table、slot mapping 等全部打包进 metadata

**2. `TritonAttentionImpl.forward()` 先写 KV cache，再跑 unified attention**

```python
triton_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    attn_metadata.slot_mapping,
    ...
)

unified_attention(
    q=query[:num_actual_tokens],
    k=key_cache,
    v=value_cache,
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    seqused_k=seqused_k,
    block_table=block_table,
    ...
)
```

这两步非常符合 vLLM 的分页执行模型：

- 第一步：把新产生的 K/V 写入分页 KV Cache
- 第二步：结合 block table，从分页 KV Cache 里读取历史上下文做 attention

**3. `triton_reshape_and_cache_flash()` 负责“写 cache”**

它会根据 `slot_mapping` 决定：

- 当前 token 写到哪个 block
- 写到 block 内哪个 offset

所以这个 kernel 解决的是“逻辑 token 到物理分页地址”的写入问题。

**4. `unified_attention()` 负责“读 cache + 算 attention”**

`kernel_unified_attention_2d/3d` 会通过 `block_table` 完成逻辑序列到物理 block 的寻址，并在 tile 循环里完成 attention 计算。

### 代码示例

下面用伪代码概括 vLLM Triton attention 的主路径：

```python
def triton_attn_forward(query, key, value, kv_cache, metadata):
    # 1. 把新 token 的 K/V 写进分页 KV Cache
    reshape_and_cache(
        key=value,
        value=value,
        slot_mapping=metadata.slot_mapping,
        kv_cache=kv_cache,
    )

    # 2. 从 block_table 指向的 KV blocks 中读取历史上下文
    # 3. 结合 seq_lens/query_start_loc 计算 attention
    return unified_attention(
        q=query,
        kv_cache=kv_cache,
        block_table=metadata.block_table,
        seq_lens=metadata.seq_lens,
    )
```

### 架构总览

```text
        vLLM 中 Triton Attention 后端的主调用链
========================================================================

InputBatch / AttentionMetadata
   |
   v
TritonAttentionMetadataBuilder
   |
   v
TritonAttentionImpl.forward
   |
   +--> triton_reshape_and_cache_flash
   |        |
   |        v
   |     写入分页 KV Cache
   |
   +--> unified_attention
            |
            +--> kernel_unified_attention_2d
            +--> kernel_unified_attention_3d
            |
            v
         输出 attention 结果
```

---

## 总结

1. Attention 成为瓶颈，不只是因为公式复杂，而是因为它同时叠加了高复杂度计算、重访存和动态形状处理
2. Triton 适合这类算子，因为它让我们能用 Python 写出 tile 级 GPU kernel，同时保留较强的性能控制能力
3. Triton kernel 的基本思路可以从三个教学例子建立起来：向量加法、Softmax、Matmul
4. 真实 attention kernel 的难点不在公式，而在 tile 化、在线 softmax、mask 逻辑和分页 KV Cache 寻址
5. vLLM 里的 Triton attention 后端不是单个 kernel，而是 `metadata builder + cache write kernel + unified attention kernel` 的组合
6. `triton_reshape_and_cache_flash()` 负责写分页 KV Cache，`unified_attention()` 负责从分页 KV Cache 读数据并完成注意力计算

## 思考题

1. 为什么说 attention kernel 的难点更多在“访存与布局”而不是“公式本身”？
2. Triton 和 CUDA 都能写高性能 kernel，vLLM 在什么场景下更适合用 Triton？什么时候你可能还是会考虑手写 CUDA？
3. `TritonAttentionMetadata` 为什么要携带 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping` 这么多字段？如果只保留 `Q/K/V` 会缺什么？
4. `triton_reshape_and_cache_flash()` 和 `unified_attention()` 为什么要拆成两个 kernel，而不是直接在一个大 kernel 里全做完？
5. 如果你要继续优化 vLLM 的 Triton attention 后端，你会优先看 `2D/3D kernel 切换策略`、`KV cache 写入` 还是 `online softmax`？为什么？
