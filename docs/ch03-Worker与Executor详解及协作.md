# 课时3 - Worker/Executor 详解及协作

> **关键源码文件**：
> - `vllm/vllm/v1/executor/abstract.py` -- `Executor` 抽象接口，负责后端选择、RPC 入口定义和执行控制
> - `vllm/vllm/v1/executor/multiproc_executor.py` -- 多进程执行器实现，包含 `MessageQueue` 广播、`WorkerProc` 生命周期和 RPC 收发
> - `vllm/vllm/v1/worker/worker_base.py` -- `WorkerBase` 与 `WorkerWrapperBase`，定义 Worker 抽象接口与延迟初始化流程
> - `vllm/vllm/v1/worker/gpu_worker.py` -- GPU Worker 设备初始化、模型加载、模型执行与流水线并行协作
> - `vllm/vllm/v1/worker/gpu/model_runner.py` -- `GPUModelRunner`，负责模型加载、KV Cache 初始化、输入构造、前向计算与采样

## 学习目标

1. 理解为什么 vLLM 要把执行层拆成 `Executor` 和 `Worker` 两个角色
2. 掌握 `WorkerBase`、`WorkerWrapperBase`、`MultiprocExecutor`、`WorkerProc` 之间的职责边界
3. 理解 `collective_rpc()` 如何把 `SchedulerOutput` 从 Executor 广播到各个 Worker
4. 掌握 `Worker.execute_model()` 和 `GPUModelRunner.execute_model()/sample_tokens()` 的协作关系
5. 能够从源码层面串起一次 `SchedulerOutput -> Executor -> Worker -> ModelRunnerOutput` 的完整路径

---

## 3.1 前言

### 原理讲解

在 vLLM 的执行层里，`Executor` 和 `Worker` 并不是同义词。

可以先用一句话区分它们：

- **Executor**：站在“调度器/引擎”视角，负责把命令发给执行侧
- **Worker**：站在“设备/GPU”视角，负责真正执行模型

为什么要这样拆分？因为在真实部署里，模型执行往往需要满足三个条件：

1. **控制面和执行面分离**：调度器只关心“做什么”，不关心“在哪块 GPU 上怎么做”
2. **支持多种后端**：单进程、multiprocessing、Ray、外部 launcher 都要复用同一套上层逻辑
3. **支持多设备/多并行策略**：TP、PP、DP、EP 都要求“控制命令统一，执行细节下沉”

因此，vLLM 采用了一种非常经典的分层：

- 上层通过 `Executor` 发控制命令
- 下层通过 `Worker` 绑定具体设备和模型运行器
- 更底层由 `ModelRunner` 负责真正的 forward / sample

从源码阅读角度看，**理解 `Executor` 和 `Worker` 的边界，就是理解 vLLM 执行层的骨架。**

### 源码解析

**1. `Executor.get_class()` 决定用哪种执行后端**

```python
# vllm/vllm/v1/executor/abstract.py 第45-85行
@staticmethod
def get_class(vllm_config: VllmConfig) -> type["Executor"]:
    distributed_executor_backend = parallel_config.distributed_executor_backend
    ...
    elif distributed_executor_backend == "mp":
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        executor_class = MultiprocExecutor
    elif distributed_executor_backend == "uni":
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        executor_class = UniProcExecutor
```

这意味着上层引擎并不直接写死“只能用多进程”。它只依赖 `Executor` 抽象，再根据配置挑选具体实现。

**2. `Executor` 定义的是“控制接口”**

```python
# vllm/vllm/v1/executor/abstract.py 第110-116行
def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
    self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
    self.collective_rpc("compile_or_warm_up_model")
```

```python
# vllm/vllm/v1/executor/abstract.py 第200-226行
def execute_model(
    self, scheduler_output: SchedulerOutput, non_block: bool = False
) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
    output = self.collective_rpc(
        "execute_model", args=(scheduler_output,), non_block=non_block
    )
    return output[0]
```

也就是说，`Executor` 真正负责的是：

- 广播控制命令
- 收集返回值
- 管理失败回调和健康状态

它不直接做 forward，也不直接持有模型参数。

**3. `WorkerBase` 定义的是“设备执行接口”**

```python
# vllm/vllm/v1/worker/worker_base.py 第35-167行
class WorkerBase:
    def init_device(self) -> None:
        raise NotImplementedError

    def load_model(self) -> None:
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        raise NotImplementedError
```

这一层已经明显下沉到设备执行语义：

- 初始化设备
- 加载模型
- 执行模型
- 采样 token

**4. `WorkerWrapperBase` 负责“延迟初始化”**

```python
# vllm/vllm/v1/worker/worker_base.py 第170-248行
class WorkerWrapperBase:
    def __init__(...):
        self.worker: WorkerBase | None = None

    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        kwargs = all_kwargs[self.rpc_rank]
        self.vllm_config = kwargs.get("vllm_config")
        self.vllm_config.enable_trace_function_call_for_thread()
        ...
```

为什么还要再包一层 `Wrapper`？

- 因为多进程场景下，环境变量、rank、plugin、worker class 解析等准备工作，需要在真正实例化 `Worker` 前先做掉
- `Wrapper` 把“进程侧生命周期管理”和“真正的设备执行对象”分开了

### 代码示例

下面用一个极简示例模仿这种分层关系：

```python
class SimpleWorker:
    def init_device(self):
        print("[Worker] init device")

    def load_model(self):
        print("[Worker] load model")

    def execute_model(self, payload):
        print(f"[Worker] execute: {payload}")
        return {"result": payload.upper()}


class SimpleExecutor:
    def __init__(self, worker):
        self.worker = worker

    def initialize(self):
        self.worker.init_device()
        self.worker.load_model()

    def collective_rpc(self, method, *args):
        fn = getattr(self.worker, method)
        return fn(*args)


worker = SimpleWorker()
executor = SimpleExecutor(worker)
executor.initialize()
print(executor.collective_rpc("execute_model", "hello"))
```

这个 demo 没有多进程，但它已经体现出最核心的结构：**Executor 负责下命令，Worker 负责真执行。**

### 架构总览

```text
                 vLLM 执行层的角色分工
  ====================================================================

  Scheduler / Engine
        |
        v
  Executor
    - 选择执行后端
    - 广播 RPC
    - 收集结果
        |
        v
  Worker
    - 绑定 device / rank
    - 加载模型
    - 执行 execute_model / sample_tokens
        |
        v
  ModelRunner
    - 构造输入
    - forward
    - sample
```

---

## 3.2 Worker 组件介绍及初始化、执行

### 原理讲解

当 `Executor` 选择好多进程或单进程后，真正落到设备上的执行者就是 `Worker`。

以最常见的 GPU 场景为例，一个 Worker 的生命周期大致如下：

1. 构造 Worker 对象，接收 rank / local_rank / distributed_init_method
2. `init_device()`：绑定 CUDA 设备，初始化分布式环境，创建 `GPUModelRunner`
3. `load_model()`：把模型权重加载到设备上
4. `determine_available_memory()`：做一次 profile，推算 KV Cache 可用显存
5. `initialize_from_config()`：初始化 KV Cache
6. `compile_or_warm_up_model()`：预热 kernel，必要时 capture CUDA Graph
7. `execute_model()`：接收本轮调度输出并执行 forward
8. `sample_tokens()`：对 forward 结果做采样，整理为 `ModelRunnerOutput`

注意这里最重要的一点：**Worker 并不直接把“模型执行细节”写在自己体内，而是继续下沉到 `GPUModelRunner`。**

### 源码解析

**1. `Worker.__init__()` 和 `init_device()` 完成设备初始化**

```python
# vllm/vllm/v1/worker/gpu_worker.py 第66-112行
class Worker(WorkerBase):
    def __init__(...):
        super().__init__(...)
        ...
        profiler_config = vllm_config.profiler_config
        if profiler_config.profiler == "torch":
            self.profiler = TorchProfilerWrapper(...)
        elif profiler_config.profiler == "cuda":
            self.profiler = CudaProfilerWrapper(profiler_config)
```

```python
# vllm/vllm/v1/worker/gpu_worker.py 第179-282行
def init_device(self):
    self.device = torch.device(f"cuda:{self.local_rank}")
    current_platform.set_device(self.device)

    init_worker_distributed_environment(
        self.vllm_config,
        self.rank,
        self.distributed_init_method,
        self.local_rank,
        current_platform.dist_backend,
    )

    self.init_snapshot = MemorySnapshot()
    ...
    if self.use_v2_model_runner:
        from vllm.v1.worker.gpu.model_runner import (
            GPUModelRunner as GPUModelRunnerV2,
        )
        self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
    else:
        from vllm.v1.worker.gpu_model_runner import (
            GPUModelRunner as GPUModelRunnerV1,
        )
        self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

`init_device()` 的职责非常重，它至少完成了四件事：

- 选择并绑定本进程使用的 CUDA 设备
- 初始化分布式通信环境
- 记录显存快照
- 按运行配置创建真正的模型执行器 `GPUModelRunner`

**2. `load_model()` 并不自己加载，而是委托给 `model_runner`**

```python
# vllm/vllm/v1/worker/gpu_worker.py 第286-289行
def load_model(self) -> None:
    eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
    with self._maybe_get_memory_pool_context(tag="weights"):
        self.model_runner.load_model(eep_scale_up=eep_scale_up)
```

这说明 Worker 更像一个 orchestration 层：

- Worker 负责上下文
- ModelRunner 负责模型内部细节

**3. Worker 会先 profile 显存，再确定 KV Cache 空间**

```python
# vllm/vllm/v1/worker/gpu_worker.py 第297-382行
def determine_available_memory(self) -> int:
    with memory_profiling(
        self.init_snapshot,
        weights_memory=int(self.model_runner.model_memory_usage),
    ) as profile_result:
        self.model_runner.profile_run()

    self.available_kv_cache_memory_bytes = (
        self.requested_memory - profile_result.non_kv_cache_memory
    )
    return int(self.available_kv_cache_memory_bytes)
```

这一步非常关键，因为 vLLM 的高吞吐依赖 KV Cache，而 KV Cache 大小又直接取决于权重、激活和 CUDA Graph 占用之后还剩多少显存。

**4. `compile_or_warm_up_model()` 完成预热与 CUDA Graph capture**

```python
# vllm/vllm/v1/worker/gpu_worker.py 第421-542行
def compile_or_warm_up_model(self) -> None:
    ...
    for size in sorted(warmup_sizes, reverse=True):
        self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)

    kernel_warmup(self)

    if not self.model_config.enforce_eager:
        cuda_graph_memory_bytes = self.model_runner.capture_model()
```

这一段把“初始化完成”和“可高性能执行”区分开了：

- `load_model()` 只是把模型放到设备上
- `compile_or_warm_up_model()` 才是把运行时状态预热到最佳

**5. `GPUModelRunner` 才是具体的模型执行核心**

构造函数里初始化请求状态、输入缓冲区、采样器和 CUDA Graph 管理器：

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第67-145行
class GPUModelRunner(...):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.req_states = RequestState(...)
        self.input_buffers = InputBuffers(...)
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)
        self.cudagraph_manager = CudaGraphManager(self.vllm_config, self.device)
```

加载模型时再去找真正的 model loader：

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第149-174行
def load_model(self, *args, **kwargs) -> None:
    with DeviceMemoryProfiler() as m:
        model_loader = get_model_loader(self.vllm_config.load_config)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.vllm_config.model_config,
        )
```

初始化 KV Cache 时再构造 block table 和 attention backend：

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第182-225行
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    self.block_tables = BlockTables(...)
    self.attn_backends, self.attn_metadata_builders = init_attn_backend(...)
    self.kv_caches = []
    init_kv_cache(...)
```

**6. 真正的执行分成两步：`execute_model()` 和 `sample_tokens()`**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第857-947行
def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
    cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp = (
        self.get_cudagraph_and_dp_padding(scheduler_output)
    )
    ...
    input_batch = self.prepare_inputs(
        scheduler_output,
        num_tokens_after_padding,
    )
    ...
    hidden_states = self.model(
        input_ids=input_batch.input_ids,
        positions=input_batch.positions,
    )

    self.execute_model_state = hidden_states, input_batch, sampling_metadata
    return None
```

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第949-1006行
def sample_tokens(self, grammar_output):
    hidden_states, input_batch, sampling_metadata = self.execute_model_state
    sampler_output, num_sampled, num_rejected = self.sample(
        hidden_states, input_batch, sampling_metadata, grammar_output
    )
    ...
    model_runner_output = ModelRunnerOutput(...)
    async_output = AsyncOutput(...)
    self.postprocess(...)
    return async_output.get_output()
```

为什么要拆成两步？

- `execute_model()` 专注 forward
- `sample_tokens()` 专注采样与后处理

这样做能更好适配结构化输出、异步调度以及某些并行策略。

### 代码示例

下面的简化代码模仿 Worker + ModelRunner 的生命周期：

```python
class SimpleModelRunner:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = lambda x: [token.upper() for token in x]
        print("[ModelRunner] model loaded")

    def execute_model(self, scheduler_output):
        tokens = scheduler_output["tokens"]
        hidden_states = self.model(tokens)
        return {"hidden_states": hidden_states}


class SimpleGPUWorker:
    def __init__(self):
        self.device = None
        self.model_runner = None

    def init_device(self):
        self.device = "cuda:0"
        self.model_runner = SimpleModelRunner()
        print(f"[Worker] bind device {self.device}")

    def load_model(self):
        self.model_runner.load_model()

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)


worker = SimpleGPUWorker()
worker.init_device()
worker.load_model()
print(worker.execute_model({"tokens": ["hello", "vllm"]}))
```

### 数据流图

```text
                  Worker 的初始化与执行生命周期
  ====================================================================

  Worker.__init__()
        |
        v
  init_device()
    - 绑定 CUDA device
    - 初始化 distributed
    - 创建 GPUModelRunner
        |
        v
  load_model()
        |
        v
  determine_available_memory()
        |
        v
  initialize_from_config()
        |
        v
  compile_or_warm_up_model()
        |
        v
  execute_model()
        |
        v
  sample_tokens()
```

---

## 3.3 Executor 和 Worker 组件通信 Demo-RPC 过程

### 原理讲解

在多进程执行器里，`Executor` 和 `Worker` 之间的通信并不是传统意义上的 HTTP RPC，也不是 ZeroMQ，而是基于共享内存消息队列 `MessageQueue` 的**进程内控制面 RPC**。

核心思路非常直接：

1. `Executor` 把 `(method, args, kwargs, output_rank)` 广播给 Worker
2. 每个 Worker 进程在 `worker_busy_loop()` 里取出消息
3. Worker 根据方法名调用对应成员函数
4. 结果再写回响应队列，由 `Executor` 收集

因此，`collective_rpc()` 可以理解为：“一次面向全部 Worker 的控制命令广播”。

### 源码解析

**1. `MultiprocExecutor` 启动时会先拉起所有 Worker 进程**

```python
# vllm/vllm/v1/executor/multiproc_executor.py 第99-214行
def _init_executor(self) -> None:
    self.rpc_broadcast_mq: MessageQueue | None = None
    ...
    if self.parallel_config.node_rank_within_dp == 0:
        self.rpc_broadcast_mq = MessageQueue(...)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

    for local_rank in range(self.local_world_size):
        unready_workers.append(
            WorkerProc.make_worker_process(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=global_rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
                shared_worker_lock=shared_worker_lock,
            )
        )
```

这一段说明 Executor 初始化时就完成了两件事：

- 创建广播消息队列
- 拉起 Worker 子进程

**2. `collective_rpc()` 是真正的控制面入口**

```python
# vllm/vllm/v1/executor/multiproc_executor.py 第287-359行
def collective_rpc(...):
    if isinstance(method, str):
        send_method = method
    else:
        send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)

    self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

    def get_response():
        responses = []
        for mq in response_mqs:
            status, result = mq.dequeue(...)
            ...
            responses.append(result)
        return responses[0] if output_rank is not None else responses
```

这段代码里最关键的消息格式是：

```text
(method, args, kwargs, output_rank)
```

其中：

- `method`：方法名，或序列化后的 callable
- `args/kwargs`：要传给 Worker 的参数
- `output_rank`：如果只想要某个 Worker 的返回值，就指定它

**3. Worker 进程是在 `make_worker_process()` 里拉起的**

```python
# vllm/vllm/v1/executor/multiproc_executor.py 第568-606行
@staticmethod
def make_worker_process(...):
    proc = context.Process(
        target=WorkerProc.worker_main,
        kwargs=process_kwargs,
        name=f"VllmWorker-{rank}",
        daemon=True,
    )
    proc.start()
    return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)
```

这里很像一个“轻量 RPC Server”的启动过程：

- 父进程负责创建子进程
- 子进程入口函数固定为 `worker_main`

**4. Worker 启动完成后会回传 READY 信号**

```python
# vllm/vllm/v1/executor/multiproc_executor.py 第675-742行
@staticmethod
def worker_main(*args, **kwargs):
    worker = WorkerProc(*args, **kwargs)
    ...
    ready_writer.send(
        {
            "status": WorkerProc.READY_STR,
            "handle": worker.worker_response_mq.export_handle(),
            "peer_response_handles": worker.peer_response_handles,
        }
    )
    worker.worker_busy_loop(cancel=shutdown_event)
```

这说明 Executor 并不会“盲目认为 Worker 已经可用”，而是等待 Worker 完成：

- device 初始化
- 模型加载
- 消息队列准备

之后才正式进入 RPC 循环。

**5. `worker_busy_loop()` 就是 Worker 侧的 RPC 分发器**

```python
# vllm/vllm/v1/executor/multiproc_executor.py 第806-832行
def worker_busy_loop(self, cancel: threading.Event | None = None):
    while True:
        method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(...)
        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)

            output = func(*args, **kwargs)
        except Exception as e:
            ...
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)
            continue

        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)
```

这一段就是本章最重要的 RPC 核心：

- 从广播队列取命令
- 找到 Worker 上对应的方法
- 执行
- 把结果写回

### 代码示例

下面用 `multiprocessing.Queue` 写一个最小版 RPC demo：

```python
import multiprocessing as mp


def worker_loop(in_q, out_q):
    class SimpleWorker:
        def execute_model(self, payload):
            return payload * 2

    worker = SimpleWorker()

    while True:
        method, args = in_q.get()
        if method == "STOP":
            break
        fn = getattr(worker, method)
        out_q.put(fn(*args))


if __name__ == "__main__":
    in_q = mp.Queue()
    out_q = mp.Queue()

    p = mp.Process(target=worker_loop, args=(in_q, out_q))
    p.start()

    in_q.put(("execute_model", (21,)))
    print(out_q.get())  # 42

    in_q.put(("STOP", ()))
    p.join()
```

这个例子虽然非常简化，但已经完整体现出：

- Executor 发送方法名和参数
- Worker 查找方法并执行
- 结果通过响应队列回传

### 数据流图

```text
                MultiprocExecutor 的 RPC 通信流程
  ====================================================================

  MultiprocExecutor.collective_rpc()
          |
          | enqueue(method, args, kwargs, output_rank)
          v
  rpc_broadcast_mq
          |
          v
  WorkerProc.worker_busy_loop()
          |
          | getattr(self.worker, method)
          v
  Worker.execute_model() / Worker.sample_tokens()
          |
          v
  handle_output() -> worker_response_mq
          |
          v
  MultiprocExecutor.get_response()
```

---

## 3.4 Executor 和 Worker 组件的协作

### 原理讲解

前一节讲的是“怎么通信”，这一节讲的是“通信之后怎么真正协作完成一轮推理”。

从引擎视角看，一轮执行大概是这样：

1. Scheduler 产出 `SchedulerOutput`
2. Executor 把 `SchedulerOutput` 广播给 Worker
3. Worker 先做必要的 PP/TP 数据准备
4. Worker 调用 `GPUModelRunner.execute_model()` 做 forward
5. Worker 再调用 `GPUModelRunner.sample_tokens()` 做采样与后处理
6. 结果汇总成 `ModelRunnerOutput` 回到 Executor，再交还给上层引擎

这条协作链的关键点在于：**控制面只传“这一轮该执行哪些请求、各自要算多少 token、分到哪些 block”，执行面才真正构造 GPU 输入并计算。**

### 源码解析

**1. `Executor.execute_model()` 只是一个 RPC 包装层**

```python
# vllm/vllm/v1/executor/multiproc_executor.py 第254-264行
def execute_model(
    self, scheduler_output: SchedulerOutput, non_block: bool = False
) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
    return self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,
        non_block=non_block,
        timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        kv_output_aggregator=self.kv_output_aggregator,
    )
```

可以看到，Executor 自己并不解释 `SchedulerOutput` 的内部细节，它只负责转发。

**2. `Worker.execute_model()` 负责承接并行策略和 `ModelRunner`**

```python
# vllm/vllm/v1/worker/gpu_worker.py 第575-642行
def execute_model(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput | None:
    ...
    if forward_pass and not get_pp_group().is_first_rank:
        tensor_dict = get_pp_group().recv_tensor_dict(...)
        intermediate_tensors = IntermediateTensors(tensor_dict)

    output = self.model_runner.execute_model(
        scheduler_output, intermediate_tensors
    )
    if isinstance(output, (ModelRunnerOutput, NoneType)):
        return output

    get_pp_group().send_tensor_dict(output.tensors, ...)
    return None
```

这一段非常关键，因为它体现出 Worker 的“协作职责”：

- 如果有 Pipeline Parallel，就先接收前一阶段的张量
- 然后调用 `model_runner.execute_model()`
- 如果当前不是最后一个 PP stage，就把中间张量继续发给下一阶段

所以 Worker 不是单纯的“调用一层函数”，而是承担了并行协作胶水层的角色。

**3. `GPUModelRunner.execute_model()` 把 `SchedulerOutput` 转成 GPU 可执行输入**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第857-947行
def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
    cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp = (
        self.get_cudagraph_and_dp_padding(scheduler_output)
    )
    ...
    self.update_states(scheduler_output)
    input_batch = self.prepare_inputs(
        scheduler_output,
        num_tokens_after_padding,
    )
    ...
    hidden_states = self.model(
        input_ids=input_batch.input_ids,
        positions=input_batch.positions,
    )
    self.execute_model_state = hidden_states, input_batch, sampling_metadata
    return None
```

这一步就是把“调度器语言”翻译成“模型语言”的过程：

- 调度器只知道 request、token budget、block id
- ModelRunner 要把它们变成 `input_ids`、`positions`、`attn_metadata`

**4. `prepare_inputs()` 是协作中最具体的一步**

在 `prepare_inputs()` 中，ModelRunner 会根据 `SchedulerOutput`：

- 排序请求
- 计算每个请求本轮要执行多少 token
- 拼接 `query_start_loc`
- 组装 block table
- 计算 slot mapping
- 构造 attention metadata

这意味着：**Scheduler 和 Worker 的边界，就在 `SchedulerOutput -> InputBatch` 这里。**

**5. `sample_tokens()` 把前向结果转换回调度器能消费的输出**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py 第949-1006行
def sample_tokens(self, grammar_output):
    hidden_states, input_batch, sampling_metadata = self.execute_model_state
    sampler_output, num_sampled, num_rejected = self.sample(...)
    prompt_logprobs_dict = self.compute_prompt_logprobs(hidden_states, input_batch)

    model_runner_output = ModelRunnerOutput(...)
    async_output = AsyncOutput(...)
    self.postprocess(...)
    return async_output.get_output()
```

这一步完成了从“模型内部 hidden_states”到“调度器可消费结果”的逆向翻译：

- 采样 token
- 计算 prompt logprobs
- 更新请求状态
- 生成 `ModelRunnerOutput`

### 代码示例

下面的示例把 `SchedulerOutput -> Executor -> Worker -> ModelRunnerOutput` 压缩成一条最小路径：

```python
class SimpleModelRunner:
    def execute_model(self, scheduler_output):
        tokens = scheduler_output["tokens"]
        hidden_states = [token.upper() for token in tokens]
        self.state = hidden_states
        return None

    def sample_tokens(self):
        return {"sampled_tokens": self.state}


class SimpleWorker:
    def __init__(self):
        self.model_runner = SimpleModelRunner()

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self):
        return self.model_runner.sample_tokens()


class SimpleExecutor:
    def __init__(self, worker):
        self.worker = worker

    def run_step(self, scheduler_output):
        self.worker.execute_model(scheduler_output)
        return self.worker.sample_tokens()


executor = SimpleExecutor(SimpleWorker())
output = executor.run_step({"tokens": ["hello", "world"]})
print(output)
```

这个 demo 和真实 vLLM 相比省略了非常多内容，但保留了最关键的协作骨架。

### 完整数据流程图

```text
                 SchedulerOutput 到 ModelRunnerOutput 的协作链
  =========================================================================

  Scheduler
    |
    | schedule()
    v
  SchedulerOutput
    |
    | Executor.execute_model()
    v
  MultiprocExecutor.collective_rpc("execute_model")
    |
    v
  Worker.execute_model()
    |
    | (必要时处理 PP/TP 中间张量)
    v
  GPUModelRunner.execute_model()
    |
    | update_states()
    | prepare_inputs()
    | model(...)
    v
  execute_model_state = (hidden_states, input_batch, sampling_metadata)
    |
    | Worker.sample_tokens()
    v
  GPUModelRunner.sample_tokens()
    |
    | sample()
    | compute_prompt_logprobs()
    | postprocess()
    v
  ModelRunnerOutput
    |
    v
  Executor / Engine / Scheduler 后处理
```

---

## 总结

本章重点拆解了 vLLM 执行层中 `Executor` 和 `Worker` 的分工关系：

1. **角色边界**：`Executor` 负责控制面，`Worker` 负责设备执行面，`ModelRunner` 负责最底层模型运行细节
2. **抽象接口**：`Executor` 和 `WorkerBase` 通过抽象接口把上层调度逻辑与底层硬件实现解耦
3. **初始化流程**：`Worker` 会经历设备绑定、分布式初始化、模型加载、显存 profile、KV Cache 初始化和 warmup
4. **RPC 过程**：`MultiprocExecutor.collective_rpc()` 通过 `MessageQueue` 把命令广播给各个 Worker，`worker_busy_loop()` 负责分发执行
5. **协作主链**：`SchedulerOutput` 先进入 `Worker.execute_model()`，再下沉到 `GPUModelRunner.execute_model()/sample_tokens()` 完成真实前向与采样
6. **理解收益**：掌握这一层之后，后续再读调度器、PagedAttention、ModelRunner 或分布式推理章节时，都会更容易定位“控制逻辑”和“执行逻辑”分别落在哪一层

---

## 思考题

1. **为什么 vLLM 不直接让调度器去调用 `Worker`，而是中间再插一层 `Executor`？** 请从“后端可替换性”和“控制面抽象”两个角度回答。

2. **`WorkerWrapperBase` 存在的意义是什么？** 如果没有这一层，直接在子进程里构造 `Worker`，会在哪些初始化环节上变得更难管理？

3. **在 `collective_rpc()` 中，为什么 `method` 既可以是字符串，也可以是序列化后的 callable？** 这样设计相比只支持字符串方法名有什么好处和代价？

4. **`GPUModelRunner.execute_model()` 和 `GPUModelRunner.sample_tokens()` 为什么要拆成两步？** 如果把两步硬塞进一个函数里，会对异步调度或结构化输出带来什么限制？

5. **设计题**：如果你要给 vLLM 增加一个新的 Worker 后端（例如某种专用加速器），你认为最少需要实现 `WorkerBase` 的哪些方法？哪些逻辑可以继续复用 `Executor` 不动？
