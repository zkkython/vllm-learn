# 课时9 - vLLM 高级特性-cudagraph详解

> **关键源码文件**：
> - `vllm/vllm/compilation/cuda_graph.py` -- `CUDAGraphWrapper`、`CUDAGraphEntry`，定义 capture / replay 的通用包装逻辑
> - `vllm/vllm/v1/cudagraph_dispatcher.py` -- `CudagraphDispatcher`，负责把运行时 batch 映射到 FULL / PIECEWISE / NONE
> - `vllm/vllm/forward_context.py` -- `BatchDescriptor` 与 `set_forward_context()`，负责把 cudagraph 运行模式传给 forward 过程
> - `vllm/vllm/v1/worker/gpu/cudagraph_utils.py` -- `CudaGraphManager`，负责 worker 侧 full CUDA graph 的 capture 和 replay
> - `vllm/vllm/v1/worker/gpu/model_runner.py` -- `capture_model()`、`get_cudagraph_and_dp_padding()`、`execute_model()`，负责 V1 worker 侧 CUDA graph 接入
> - `vllm/vllm/compilation/monitor.py` -- capture 合法性检查
> - `vllm/vllm/compilation/counter.py` -- 编译与 CUDAGraph 捕获统计计数

## 学习目标

1. 理解 CUDA Graph 为什么能降低推理时的 CPU launch 开销，以及它对静态形状的依赖
2. 掌握使用 CUDA Graph 时最关键的几个约束：固定地址、固定 batch 描述、capture 时机合法
3. 能够用一个最小 PyTorch 例子把 `warmup -> capture -> replay` 跑通
4. 理解 vLLM 中 FULL cudagraph、PIECEWISE cudagraph 和 eager fallback 的区别
5. 掌握 `Dispatcher -> ForwardContext -> Wrapper` 和 `CudaGraphManager -> GPUModelRunner` 这两条实际接入路径

---

## 9.1 CUDA Graph 原理

### 原理讲解

CUDA Graph 的目标很直接：

**把一串固定结构的 GPU 操作录下来，后面重复执行时直接 replay，减少 CPU 一次次发射 kernel 的开销。**

它最适合的场景通常有两个特点：

1. 执行图结构稳定
2. 输入 shape 和内存布局稳定

在大模型推理里，decode 阶段恰好经常满足这些条件：

- 每轮 query 很短
- 常见 batch size 可枚举
- 执行路径比较固定

所以 vLLM 会尝试对一批常见 size 提前 capture，然后在运行时复用。

### 源码解析

在 vLLM 的通用包装层里，CUDA Graph 的核心对象是 `CUDAGraphEntry`：

```python
# vllm/vllm/compilation/cuda_graph.py
@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None
```

它记录了一个 graph 至少需要保存的三样东西：

- 这个 graph 对应什么 batch 描述
- 已经 capture 好的 `torch.cuda.CUDAGraph`
- replay 时要返回的输出引用

`CUDAGraphWrapper.__call__()` 则把 runtime 调用分成了三种情况：

1. 当前模式不是图模式，直接 eager 执行
2. 当前 batch 还没 capture，先 capture
3. 当前 batch 已经 capture，直接 replay

这正是 CUDA Graph 在运行时最核心的三态。

### 代码示例

下面先用一句话版伪代码建立直觉：

```python
if batch_key not in graph_cache:
    graph_cache[batch_key] = capture(batch_key)
return replay(graph_cache[batch_key])
```

这就是 CUDA Graph 的核心思想：**同一个“稳定 batch 模板”只 capture 一次，后面反复 replay。**

---

## 9.2 CUDA Graph 使用的核心注意事项

### 原理讲解

CUDA Graph 好用，但限制也很强。最重要的约束通常有四个：

1. **输入张量地址要稳定**
2. **shape / 执行路径要稳定**
3. **capture 前要做 warmup**
4. **capture 时机要合法**

如果把这些约束忘了，最常见的问题就是：

- replay 出错
- 图命不中，频繁回退 eager
- capture 期间发生不允许的操作

### 源码解析

**1. vLLM 会在 debug 模式下检查输入地址是否稳定**

```python
# vllm/vllm/compilation/cuda_graph.py :: CUDAGraphWrapper.__call__
input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
entry.input_addresses = input_addresses
...
assert new_input_addresses == entry.input_addresses
```

这就把“地址稳定”这个要求写死在代码里了。

**2. vLLM 会在 capture 前检查当前时机是否合法**

```python
# vllm/vllm/compilation/cuda_graph.py
validate_cudagraph_capturing_enabled()
```

对应的检查函数在 `compilation/monitor.py` 中，如果当前不允许 capture，会直接抛错。

**3. worker 侧会重用固定输入 buffer**

```python
# vllm/vllm/v1/worker/gpu/cudagraph_utils.py :: CudaGraphManager.capture_graph
input_ids = input_buffers.input_ids[:num_tokens]
positions = input_buffers.positions[:num_tokens]
```

这里不是每次 new 一批新张量，而是从固定 `InputBuffers` 里切片。这样做的目的就是为了让 replay 时的输入地址可复用。

**4. vLLM 还会保存 batch 描述而不是只看 token 数**

在 `forward_context.py` 里，`BatchDescriptor` 不只记录 `num_tokens`，还可以记录：

- `num_reqs`
- `uniform`
- `has_lora`

这是因为“token 数一样”并不一定意味着“执行路径完全一样”。

### 代码示例

下面给一个“错误姿势”和“正确姿势”的对比：

```python
# 错误姿势：每次都新建输入，地址不稳定
for _ in range(10):
    x = torch.randn(1024, device="cuda")
    y = model(x)

# 正确姿势：固定输入 buffer，只 copy_ 新数据
static_x = torch.empty(1024, device="cuda")
for _ in range(10):
    static_x.copy_(torch.randn(1024, device="cuda"))
    y = model(static_x)
```

对于 CUDA Graph，后者才是可 replay 的典型写法。

---

## 9.3 CUDA Graph + Model Forward 实战

### 原理讲解

如果只用一句话描述 CUDA Graph 的实战流程，可以写成：

1. 先准备固定输入 buffer
2. 做一次 warmup
3. 在 graph context 里 capture
4. 后续把新数据 copy 到固定 buffer，再 replay

注意这里真正“固定”的不是输入值，而是：

- 张量地址
- shape
- 执行路径

### 源码解析

vLLM worker 侧的 `CudaGraphManager.capture_graph()` 就是这套流程的直接实现：

```python
# vllm/vllm/v1/worker/gpu/cudagraph_utils.py
with set_forward_context(...):
    hidden_states = model(input_ids=input_ids, positions=positions)

graph = torch.cuda.CUDAGraph()
with (
    set_forward_context(...),
    torch.cuda.graph(graph, self.pool),
):
    hidden_states = model(input_ids=input_ids, positions=positions)
    self.hidden_states[:num_tokens] = hidden_states
```

这里能看到一个很标准的流程：

1. 先做一次 warmup forward
2. 再创建 `torch.cuda.CUDAGraph()`
3. 在 `torch.cuda.graph(...)` 作用域内做 capture
4. 把输出保存到稳定的 `self.hidden_states`

后面 replay 时就只需要：

```python
self.graphs[num_tokens].replay()
return self.hidden_states[:num_tokens]
```

### 代码示例

下面给一个最小可运行的 PyTorch 版 CUDA Graph 示例：

```python
import torch
import torch.nn as nn


device = "cuda"
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.GELU(),
    nn.Linear(2048, 1024),
).to(device).eval()

static_x = torch.empty(8, 1024, device=device)
graph = torch.cuda.CUDAGraph()

# warmup
warmup_x = torch.randn(8, 1024, device=device)
_ = model(warmup_x)
torch.cuda.synchronize()

with torch.cuda.graph(graph):
    static_y = model(static_x)

# replay
for _ in range(3):
    static_x.copy_(torch.randn(8, 1024, device=device))
    graph.replay()
    print(static_y.shape)
```

这个例子里：

- `static_x` 是固定输入 buffer
- `static_y` 是固定输出位置
- `graph.replay()` 只重放之前录下来的那条计算图

---

## 9.4 CUDA Graph 小结

### 原理讲解

到这里可以把 CUDA Graph 的优缺点一起收一下。

它的优点很明显：

- 减少 CPU launch 开销
- 对高频重复 batch 很有效
- 对 decode 这类稳定路径尤其友好

它的限制也很明确：

- 需要枚举常见 batch 模板
- 对动态 shape 不友好
- capture 和 replay 需要稳定地址和执行路径

所以 CUDA Graph 本质上不是“让所有 batch 都更快”，而是：

**让一批足够稳定的热点 batch 更快。**

### 源码解析

这一点在 vLLM 的统计信息里也有体现：

```python
# vllm/vllm/compilation/counter.py
num_gpu_runner_capture_triggers: int = 0
num_cudagraph_captured: int = 0
```

vLLM 会区分：

- 触发了多少次 capture 尝试
- 真正 capture 成功了多少个图

这说明工程上很清楚一个事实：**不是每次都能命中，也不是每个 batch 都值得 capture。**

### 代码示例

下面用一个小函数模拟“batch size 到 graph size”的命中逻辑：

```python
capture_sizes = [8, 16, 32]


def choose_graph_size(num_tokens):
    for size in capture_sizes:
        if num_tokens <= size:
            return size
    return None


for n in [5, 8, 13, 40]:
    print(n, "->", choose_graph_size(n))
```

这和 vLLM worker 侧的 `get_cudagraph_size()` 思路是接近的：如果能映射到一个已捕获模板，就走 graph；否则回退 eager。

---

## 9.5 vLLM 中的 CUDA Graph

### 原理讲解

vLLM 里的 CUDA Graph 不是单一路径，而是两层机制同时存在：

1. **通用编译层的 dispatch + wrapper 机制**
2. **V1 worker 侧的 full CUDA graph 机制**

前者更通用，支持 FULL / PIECEWISE 这类运行时分发；
后者更直接，主要服务当前 `GPUModelRunner` 的 full graph replay。

因此读源码时一定要把这两层分开看。

### 源码解析

**1. 运行时先由 `CudagraphDispatcher` 决定当前 batch 该走哪种模式**

```python
# vllm/vllm/v1/cudagraph_dispatcher.py :: dispatch
if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
    return CUDAGraphMode.FULL, batch_desc
...
if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
    return CUDAGraphMode.PIECEWISE, relaxed_batch_desc
return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

这层做的是：

- FULL 能不能命中
- PIECEWISE 能不能命中
- 都不行就回退 eager

**2. `set_forward_context()` 会把运行时模式和 batch 描述塞进当前 forward 上下文**

```python
# vllm/vllm/forward_context.py
forward_context = create_forward_context(
    attn_metadata,
    vllm_config,
    ...
    cudagraph_runtime_mode,
    batch_descriptor,
)
```

这一步很关键，因为后面的 `CUDAGraphWrapper` 就是从 forward context 里取：

- `cudagraph_runtime_mode`
- `batch_descriptor`

**3. `CUDAGraphWrapper` 根据 runtime mode 决定 capture / replay / eager**

```python
# vllm/vllm/compilation/cuda_graph.py :: CUDAGraphWrapper.__call__
if cudagraph_runtime_mode == CUDAGraphMode.NONE
   or cudagraph_runtime_mode != self.runtime_mode:
    return self.runnable(*args, **kwargs)

if entry.cudagraph is None:
    # capture
else:
    # replay
```

也就是说，wrapper 本身不做复杂决策，它只“盲信” dispatcher 给出的 runtime mode 和 batch key。

**4. 但当前 V1 `GPUModelRunner` 直连路径主要使用 FULL cudagraph**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py :: get_cudagraph_and_dp_padding
if cudagraph_size is not None:
    return CUDAGraphMode.FULL, cudagraph_size, None
# Fall back to eager mode.
# TODO(woosuk): Support piecewise CUDA graphs.
return CUDAGraphMode.NONE, total_num_scheduled_tokens, None
```

这个细节非常重要：

- 通用编译层里，FULL / PIECEWISE 机制都在
- 但当前 `GPUModelRunner.execute_model()` 这条直连路径里，主要命中的是 FULL graph
- 命不中时直接回退 eager，而不是在这里切 PIECEWISE

**5. worker 侧 full graph 的 capture 和 replay 由 `CudaGraphManager` 管**

```python
# capture
self.cudagraph_manager.capture(...)

# replay
hidden_states = self.cudagraph_manager.run(
    input_batch.num_tokens_after_padding
)
```

这层更像一个“按 num_tokens 管理 graph 模板”的本地缓存器。

### 代码示例

下面用伪代码概括 vLLM 中 CUDA Graph 的两条主线：

```python
def runtime_dispatch(num_tokens, has_lora):
    mode, batch_desc = dispatcher.dispatch(num_tokens, uniform_decode=True, has_lora=has_lora)
    return mode, batch_desc


def worker_execute(scheduler_output):
    mode, padded_tokens = model_runner.get_cudagraph_and_dp_padding(scheduler_output)
    if mode == "FULL":
        return cudagraph_manager.run(padded_tokens)
    return eager_forward(scheduler_output)
```

这个伪代码刻意把两层拆开：

- 上面一层是“运行时模式选择”
- 下面一层是“worker 侧实际怎么跑”

### 架构总览

```text
            vLLM 中 CUDA Graph 的两层接入路径
========================================================================

                   通用编译层
-----------------------------------------------------------------------
 CudagraphDispatcher
        |
        v
 set_forward_context(batch_descriptor, runtime_mode)
        |
        v
 CUDAGraphWrapper
        |
        +--> 首次命中: capture
        +--> 再次命中: replay
        +--> 不匹配: eager


                   V1 Worker 直连路径
-----------------------------------------------------------------------
 GPUModelRunner.capture_model()
        |
        v
 CudaGraphManager.capture()
        |
        v
 GPUModelRunner.execute_model()
        |
        +--> FULL 命中: cudagraph_manager.run(...)
        +--> 否则: eager forward
```

---

## 总结

1. CUDA Graph 的核心价值是把稳定执行路径录下来并重放，从而减少 CPU 侧 kernel launch 开销
2. 它最关键的使用条件是：地址稳定、shape 稳定、执行路径稳定、capture 时机合法
3. `CUDAGraphWrapper` 代表通用 capture/replay 包装逻辑，`BatchDescriptor` 代表运行时图模板的 key
4. `CudagraphDispatcher` 决定 FULL / PIECEWISE / NONE，`set_forward_context()` 把这个决策传到实际 forward 过程
5. 当前 V1 `GPUModelRunner` 这条执行主链主要直接使用 FULL CUDA graph，命不中时会回退 eager
6. `CudaGraphManager` 是 worker 侧 full graph 的核心管理器，负责固定输入 buffer、capture 和 replay

## 思考题

1. 为什么 CUDA Graph 要求“输入地址稳定”，而不只是“输入 shape 一样”？
2. `BatchDescriptor` 为什么除了 `num_tokens` 之外还要记录 `num_reqs`、`uniform`、`has_lora` 这类信息？
3. `CUDAGraphWrapper` 和 `CudaGraphManager` 都能做 capture/replay，它们分别更适合解决什么层面的问题？
4. 当前 `GPUModelRunner.execute_model()` 命不中 FULL graph 时会回退 eager。如果你要在这里继续接入 piecewise graph，最难处理的部分会是什么？
5. CUDA Graph 能显著提升热点 batch 的效率，但也会带来额外的显存和预热成本。你会如何为线上服务选择 capture sizes？
