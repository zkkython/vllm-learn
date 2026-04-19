# 课时7 - vLLM 架构-总览及模型推理过程详解

> **关键源码文件**：
> - `vllm/vllm/entrypoints/llm.py` -- `LLM` 离线推理入口，负责构造 `EngineArgs` 并创建 `LLMEngine`
> - `vllm/vllm/v1/engine/llm_engine.py` -- `LLMEngine`，负责请求接入、输出整理和前后处理
> - `vllm/vllm/v1/engine/core.py` -- `EngineCore`，负责 KV Cache 初始化、调度和执行主循环
> - `vllm/vllm/v1/engine/core_client.py` -- `EngineCoreClient`，定义 inproc / multiprocess 两类前后端通信方式
> - `vllm/vllm/v1/executor/abstract.py` / `vllm/vllm/v1/executor/multiproc_executor.py` -- `Executor` 抽象与多进程执行实现
> - `vllm/vllm/v1/worker/gpu_worker.py` -- `Worker`，负责设备初始化、模型加载和 worker 侧控制逻辑
> - `vllm/vllm/v1/worker/gpu/model_runner.py` -- `GPUModelRunner`，负责请求状态更新、输入构建、模型 forward 和采样

## 学习目标

1. 理解 vLLM V1 为什么要拆成 `LLM / LLMEngine / EngineCore / Executor / Worker / ModelRunner` 多层结构
2. 掌握一次离线推理请求从 `LLM.generate()` 进入到 `Scheduler`、再到 `ModelRunner` 的完整调用链
3. 理解前端控制面和后端执行面的边界，以及 `EngineCoreClient` / `Executor` 在中间起到的解耦作用
4. 掌握 `GPUModelRunner.execute_model()`、`update_states()`、`prepare_inputs()` 三个核心阶段的职责分工
5. 能够从源码角度解释 vLLM 为什么既能支持单进程，也能支持多进程、多卡和更复杂的并行形态

---

## 7.1 LLM v1 架构核心类介绍

### 原理讲解

第一次看 vLLM V1，最容易产生的疑问是：

**为什么一次推理要经过这么多层？**

这是因为 vLLM 想同时解决三件事：

1. **给用户一个简单入口**：离线场景下用户只想调用 `LLM.generate()`
2. **把控制逻辑和执行逻辑拆开**：前端负责请求接入、输出组织，后端负责调度和 GPU 执行
3. **把单机和多进程统一起来**：同一套上层 API，既能走 inproc，也能走 multiprocess

所以从外到内，它大致分成下面几层：

- `LLM`：给离线推理用户使用的高层入口
- `LLMEngine`：真正的前端调度器入口，负责 add_request / step
- `EngineCoreClient`：把前端调用转成“本地函数调用”或“跨进程消息”
- `EngineCore`：调度与执行主循环
- `Executor`：把一个 batch 分发到一个或多个 worker
- `Worker`：管理单个进程里的设备、模型和 cache
- `GPUModelRunner`：真正把 `SchedulerOutput` 翻译成模型 forward 输入并执行

可以先把它理解成一条分层流水线：**越靠外越偏控制，越靠里越偏执行。**

### 源码解析

**1. `LLM` 是离线推理入口，它先构造 `EngineArgs`，再创建 `LLMEngine`**

```python
# vllm/vllm/entrypoints/llm.py :: LLM.__init__
engine_args = EngineArgs(...)
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
```

这意味着 `LLM` 本身不直接做调度和执行，它更像一层“用户友好的包装器”。

**2. `LLMEngine` 是前端核心，它会把输入处理器、输出处理器和 EngineCoreClient 都接起来**

```python
# vllm/vllm/v1/engine/llm_engine.py :: LLMEngine.__init__
self.input_processor = InputProcessor(...)
self.output_processor = OutputProcessor(...)
self.engine_core = EngineCoreClient.make_client(...)
```

这里的关键点是：

- `InputProcessor` 负责把 prompt 变成 `EngineCoreRequest`
- `OutputProcessor` 负责把 `EngineCoreOutputs` 还原成 `RequestOutput`
- `EngineCoreClient` 负责把前端和后端接起来

**3. `EngineCore` 才是后端主循环的起点**

```python
# vllm/vllm/v1/engine/core.py :: EngineCore.__init__
self.model_executor = executor_class(vllm_config)
num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(...)
self.scheduler = Scheduler(...)
```

`EngineCore` 里至少完成了三件大事：

- 创建 `Executor`
- 完成 KV Cache 规格计算与初始化
- 创建 `Scheduler`

也就是说，`EngineCore` 才是真正把“模型执行资源”和“调度策略”组装到一起的地方。

**4. `Executor` 往下再交给 `Worker` 和 `ModelRunner`**

```python
# vllm/vllm/v1/executor/abstract.py :: Executor.execute_model
output = self.collective_rpc(
    "execute_model", args=(scheduler_output,), non_block=non_block
)
return output[0]
```

这里可以看到，`Executor` 并不自己做 forward，它只是把 `SchedulerOutput` 发送给 worker 侧的 `execute_model()`。

**5. `GPUModelRunner` 才是模型执行的最终落点**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py :: GPUModelRunner.execute_model
input_batch = self.prepare_inputs(...)
with set_forward_context(...):
    hidden_states = self.model(
        input_ids=input_batch.input_ids,
        positions=input_batch.positions,
    )
```

所以如果我们只问“模型到底在哪一层跑起来”，答案其实是：

**最终在 worker 进程里的 `GPUModelRunner`。**

### 代码示例

下面用一个最小分层 demo 把这几个类的职责关系抽出来：

```python
class LLM:
    def __init__(self):
        self.engine = LLMEngine()

    def generate(self, prompt):
        self.engine.add_request(prompt)
        return self.engine.step()


class LLMEngine:
    def __init__(self):
        self.core = EngineCore()

    def add_request(self, prompt):
        self.core.add_request(prompt)

    def step(self):
        return self.core.step()


class EngineCore:
    def __init__(self):
        self.executor = Executor()

    def add_request(self, prompt):
        self.prompt = prompt

    def step(self):
        return self.executor.execute_model(self.prompt)


class Executor:
    def __init__(self):
        self.worker = Worker()

    def execute_model(self, prompt):
        return self.worker.model_runner.execute_model(prompt)


class Worker:
    def __init__(self):
        self.model_runner = GPUModelRunner()


class GPUModelRunner:
    def execute_model(self, prompt):
        return f"run model on: {prompt}"


print(LLM().generate("hello vllm"))
```

---

## 7.2 vLLM 核心组件关系图解

### 原理讲解

理解 vLLM 架构，一个很重要的视角是区分：

1. **控制面（control plane）**
2. **数据面（data plane）**

在 vLLM 里：

- 控制面负责请求的进入、调度、RPC、状态同步
- 数据面负责 `SchedulerOutput`、`ModelRunnerOutput`、KV Cache 和真正的 GPU 计算

这样设计的好处是：

- 单进程模式下，控制面和数据面可以直接函数调用
- 多进程模式下，只要把中间接口换成消息队列 / socket，整体结构不需要重写

### 源码解析

**1. `EngineCoreClient` 是前后端边界**

```python
# vllm/vllm/v1/engine/core_client.py :: EngineCoreClient.make_client
if multiprocess_mode and asyncio_mode:
    return AsyncMPClient(...)
if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(...)
return InprocClient(...)
```

这段分发非常关键，它说明：

- 同一个 `LLMEngine` 上层接口
- 可以接本地 `InprocClient`
- 也可以接多进程 `SyncMPClient / AsyncMPClient`

也就是说，**前端不需要知道后端到底是“同进程函数调用”还是“跨进程消息通信”。**

**2. `MultiprocExecutor` 负责把调度结果广播到 worker，并收集执行结果**

```python
# vllm/vllm/v1/executor/multiproc_executor.py :: MultiprocExecutor._init_executor
self.rpc_broadcast_mq = MessageQueue(...)
self.response_mqs = [...]
```

这里能看出多进程执行的两个核心通道：

- `rpc_broadcast_mq`：把控制消息 / 调度结果发给 workers
- `response_mqs`：把 worker 的输出收回来

**3. `Worker` 负责进程内的设备和模型生命周期**

从 `gpu_worker.py` 可以看到，`Worker` 负责的事情很多：

- `init_device()`：初始化 CUDA 设备和分布式环境
- `initialize_cache()`：接收 `EngineCore` 计算好的 KV Cache 参数
- `sleep()` / `wake_up()`：管理显存休眠与恢复
- profiler、LoRA、KV transfer 等附加机制

因此，`Worker` 不是“一个简单的 forward 容器”，而是 **GPU 进程侧的运行时外壳**。

### 代码示例

下面用一个双队列例子模拟“控制面发任务，数据面回结果”的关系：

```python
from queue import Queue


control_q = Queue()
result_q = Queue()


def scheduler():
    control_q.put({"batch_id": 1, "tokens": 32})


def worker():
    task = control_q.get()
    result_q.put({"batch_id": task["batch_id"], "logits": "gpu_done"})


scheduler()
worker()
print(result_q.get())
```

### 架构总览

```text
                     vLLM V1 组件关系
========================================================================

用户代码
   |
   v
LLM
   |
   v
LLMEngine
   |
   +--> InputProcessor
   +--> OutputProcessor
   |
   v
EngineCoreClient  <----- 前后端边界 ----->  Inproc / SyncMP / AsyncMP
   |
   v
EngineCore
   |
   +--> Scheduler
   +--> StructuredOutputManager
   +--> Executor
            |
            v
        Worker 0 ... Worker N
            |
            v
      GPUModelRunner / KV Cache / Model
```

---

## 7.3 vLLM 整体运行流程图

### 原理讲解

从用户视角看，vLLM 的一次离线推理好像只有两步：

1. `llm.generate(prompts)`
2. 拿到输出

但在内部，它至少要经历下面这条链：

1. prompt 进入 `InputProcessor`
2. 生成 `EngineCoreRequest`
3. 请求进入 `Scheduler`
4. 调度器决定这一轮要跑哪些 token
5. `Executor` 把调度结果发给 worker
6. `ModelRunner` 构造输入并执行 forward
7. 采样结果回到调度器
8. 输出回到 `OutputProcessor`
9. `RequestOutput` 返回给用户

因此，vLLM 的执行单位不是“整个请求一次跑完”，而是：

**请求被拆成多轮 step，每一轮只调度本轮预算内的 token。**

### 源码解析

**1. `LLM.generate()` 并不直接 forward，它先加请求，再驱动 engine 循环**

```python
# vllm/vllm/entrypoints/llm.py :: LLM.generate
self._validate_and_add_requests(...)
outputs = self._run_engine(use_tqdm=use_tqdm)
```

这说明 `generate()` 更像一个批量提交 + 循环拉取结果的包装过程。

**2. `LLMEngine.add_request()` 会把输入变成 `EngineCoreRequest` 再送给 EngineCore**

```python
# vllm/vllm/v1/engine/llm_engine.py :: LLMEngine.add_request
request = self.input_processor.process_inputs(...)
self.output_processor.add_request(request, prompt_text, ...)
self.engine_core.add_request(request)
```

这三步分别对应：

- 输入规范化
- 前端状态登记
- 后端调度入队

**3. `LLMEngine.step()` 每次只拉一轮执行结果**

```python
# vllm/vllm/v1/engine/llm_engine.py :: LLMEngine.step
outputs = self.engine_core.get_output()
processed_outputs = self.output_processor.process_outputs(...)
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

这里可以看到，`step()` 的核心职责不是跑模型，而是：

- 拿后端结果
- 做输出后处理
- 处理停止条件和中止逻辑

**4. 真正“调度 + 执行”的主循环在 `EngineCore.step()`**

```python
# vllm/vllm/v1/engine/core.py :: EngineCore.step
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result() or self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

这段代码基本就是 vLLM 后端主循环的缩影：

1. `schedule()` 产出本轮执行计划
2. `execute_model()` 异步把 batch 发到 worker
3. 必要时做 `sample_tokens()`
4. `update_from_output()` 把执行结果写回请求状态

### 代码示例

下面用最小 step-loop 模型模拟 vLLM 的“多轮迭代执行”：

```python
class MiniEngine:
    def __init__(self):
        self.waiting = ["reqA", "reqB"]

    def schedule(self):
        return self.waiting[:1]

    def execute(self, batch):
        return {req: "next_token" for req in batch}

    def update(self, outputs):
        for req in outputs:
            print("finish one step:", req)
            self.waiting.remove(req)

    def run(self):
        while self.waiting:
            batch = self.schedule()
            outputs = self.execute(batch)
            self.update(outputs)


MiniEngine().run()
```

### 数据流图

```text
 prompt
   |
   v
InputProcessor
   |
   v
EngineCoreRequest
   |
   v
Scheduler ----> SchedulerOutput
                    |
                    v
                 Executor
                    |
                    v
              Worker / ModelRunner
                    |
                    v
              ModelRunnerOutput
                    |
                    v
                Scheduler.update_from_output
                    |
                    v
               OutputProcessor
                    |
                    v
               RequestOutput
```

---

## 7.4 模型的执行

### 原理讲解

架构总览看完以后，下一步就是回答一个更具体的问题：

**`SchedulerOutput` 到底是怎么被跑起来的？**

这一段可以拆成三层：

1. `EngineCore.step()`：发起一次执行
2. `Executor.execute_model()`：把执行请求发到 worker
3. `GPUModelRunner.execute_model()`：在 worker 内真正构造 batch 并调用模型

从这个角度看，`ModelRunner` 是执行面最核心的一层。

### 源码解析

**1. `EngineCore.step()` 通过 `Executor` 异步触发执行**

```python
# vllm/vllm/v1/engine/core.py :: EngineCore.step
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

这里返回 `Future` 很重要，它允许：

- CPU 侧继续做语法约束等准备
- 执行器在后台推进 worker 侧 forward

**2. `Executor.execute_model()` 本质上是一个 RPC 分发**

```python
# vllm/vllm/v1/executor/abstract.py :: Executor.execute_model
output = self.collective_rpc(
    "execute_model", args=(scheduler_output,), non_block=non_block
)
```

也就是说，`SchedulerOutput` 不是直接传给 PyTorch 模型，而是先通过 worker RPC 接口传入进程内的运行时。

**3. `GPUModelRunner.execute_model()` 才是真正的执行入口**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py :: GPUModelRunner.execute_model
cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp = (
    self.get_cudagraph_and_dp_padding(scheduler_output)
)
self.update_states(scheduler_output)
input_batch = self.prepare_inputs(scheduler_output, num_tokens_after_padding)
```

这三步非常关键：

- `get_cudagraph_and_dp_padding()`：决定本轮是否走 CUDA Graph，以及是否要做 DP 对齐 padding
- `update_states()`：先更新 worker 侧请求状态和 block table
- `prepare_inputs()`：再把这一轮的执行输入拼出来

**4. 之后才进入真正的模型 forward**

```python
# eager path
with set_forward_context(...):
    hidden_states = self.model(
        input_ids=input_batch.input_ids,
        positions=input_batch.positions,
    )
```

如果当前 batch 命中了 full CUDA graph，则会走：

```python
hidden_states = self.cudagraph_manager.run(
    input_batch.num_tokens_after_padding
)
```

否则走 eager forward。

这说明 `GPUModelRunner.execute_model()` 的本质并不是“直接调模型”，而是：

**先选执行路径，再准备输入，最后调用模型。**

### 代码示例

```python
class SimpleModelRunner:
    def execute_model(self, scheduler_output):
        mode = self.choose_mode(scheduler_output)
        self.update_states(scheduler_output)
        batch = self.prepare_inputs(scheduler_output)

        if mode == "graph":
            return self.run_graph(batch)
        return self.run_eager(batch)

    def choose_mode(self, scheduler_output):
        return "graph" if scheduler_output["num_tokens"] in [16, 32] else "eager"

    def update_states(self, scheduler_output):
        print("update request states")

    def prepare_inputs(self, scheduler_output):
        return {"input_ids": [1, 2, 3], "positions": [0, 1, 2]}

    def run_graph(self, batch):
        return f"graph forward: {batch}"

    def run_eager(self, batch):
        return f"eager forward: {batch}"


runner = SimpleModelRunner()
print(runner.execute_model({"num_tokens": 16}))
print(runner.execute_model({"num_tokens": 7}))
```

---

## 7.5 更新请求

### 原理讲解

在 `GPUModelRunner` 里，很多人最容易忽略的一步是 `update_states()`。

它看起来不像真正的计算代码，但实际上它是模型执行前的关键准备步骤。

因为 worker 侧必须知道：

- 哪些请求已经结束
- 哪些请求被抢占了
- 哪些是本轮刚进来的新请求
- 哪些老请求本轮又拿到了新的 KV blocks

如果这些状态不先更新，后面的 block table、slot mapping、positions 全都会错。

### 源码解析

**1. 先删除 finished / preempted 请求**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py :: GPUModelRunner.update_states
for req_id in scheduler_output.preempted_req_ids:
    self.req_states.remove_request(req_id)
for req_id in scheduler_output.finished_req_ids:
    self.req_states.remove_request(req_id)
```

这一步对应调度器上一轮的结果清理。

**2. 再把新请求写入 `req_states`**

```python
self.req_states.add_request(
    req_id=req_id,
    prompt_len=len(new_req_data.prompt_token_ids),
    prefill_token_ids=new_req_data.prefill_token_ids,
    num_computed_tokens=new_req_data.num_computed_tokens,
    sampling_params=new_req_data.sampling_params,
    lora_request=new_req_data.lora_request,
)
```

这里不是简单记一个 `req_id`，而是把后面 forward 需要的核心信息都放进去：

- prompt 长度
- prefill token ids
- 当前已经算到哪里
- sampling 参数
- LoRA 信息

**3. 最后更新 block table**

```python
self.block_tables.append_block_ids(
    req_indices=req_indices,
    cu_num_new_blocks=cu_num_new_blocks,
    new_block_ids=new_block_ids,
    overwrite=overwrite,
)
```

这一步非常重要，因为请求在 worker 侧真正访问 KV Cache 时，靠的就是 block table。

所以 `update_states()` 本质上是在做：

**把调度器给出的“逻辑执行计划”落成 worker 侧可以直接用的数据结构。**

### 代码示例

```python
class ReqStates:
    def __init__(self):
        self.reqs = {}

    def add_request(self, req_id, prompt_len):
        self.reqs[req_id] = {"prompt_len": prompt_len, "blocks": []}

    def remove_request(self, req_id):
        self.reqs.pop(req_id, None)


states = ReqStates()
states.add_request("req1", 8)
states.reqs["req1"]["blocks"].extend([10, 11])
print(states.reqs)
states.remove_request("req1")
print(states.reqs)
```

---

## 7.6 模型输入构建

### 原理讲解

`prepare_inputs()` 是 `GPUModelRunner` 里最值得精读的函数之一。

因为调度器给到 worker 的还只是“本轮哪些请求各跑多少 token”，而模型 forward 真正需要的是：

- `input_ids`
- `positions`
- `query_start_loc`
- `seq_lens`
- `block_tables`
- `slot_mappings`
- `attn_metadata`
- `logits_indices`

也就是说，`prepare_inputs()` 做的是：

**把调度器语言翻译成模型语言。**

### 源码解析

`prepare_inputs()` 的主线可以拆成 6 步。

**1. 根据本轮调度结果确定 batch 内请求顺序**

```python
req_ids = sorted(
    scheduler_output.num_scheduled_tokens.keys(),
    key=lambda k: scheduler_output.num_scheduled_tokens[k],
)
```

当前实现里，会根据本轮每个请求调度的 token 数进行排序。这样后面构造 `query_start_loc` 和输入布局会更稳定。

**2. 建立 `idx_mapping`，把 batch 内顺序映射到 `req_states`**

```python
idx_mapping_list = [
    self.req_states.req_id_to_index[req_id] for req_id in req_ids
]
```

后面几乎所有状态读取都会通过这个映射完成。

**3. 构造 block tables 与 query_start_loc**

```python
block_tables = self.block_tables.gather_block_tables(idx_mapping)
np.cumsum(
    num_scheduled_tokens,
    out=self.input_buffers.query_start_loc.np[1 : num_reqs + 1],
)
```

这一步决定了：

- 每个请求要去哪些物理 block 读写 KV
- batch 内每个请求在 token 维度上的起止位置

**4. 准备 prefill token、positions 和 seq_lens**

```python
prepare_prefill_inputs(...)
prepare_pos_seq_lens(...)
```

这是把输入 token 和位置编码真正写到 input buffer 里的地方。

**5. 计算 slot mapping 和 attention metadata**

```python
slot_mappings = self.block_tables.compute_slot_mappings(...)
attn_metadata = build_attn_metadata(...)
```

这一步是注意力后端最关心的部分。不同 attention backend 最终都会通过 `attn_metadata` 拿到：

- query 起点
- 序列长度
- block table
- slot mapping

**6. 返回 `InputBatch`**

最终 `prepare_inputs()` 会把所有这批张量和索引整理成一个 `InputBatch` 对象，供后续模型 forward、sample 和 postprocess 使用。

### 代码示例

下面用一个最小例子说明 `query_start_loc` 的作用：

```python
import numpy as np


num_scheduled_tokens = np.array([1, 3, 2], dtype=np.int32)
query_start_loc = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
np.cumsum(num_scheduled_tokens, out=query_start_loc[1:])

print("num_scheduled_tokens =", num_scheduled_tokens.tolist())
print("query_start_loc =", query_start_loc.tolist())
```

输出含义是：

- 第 0 个请求占 `[0, 1)`
- 第 1 个请求占 `[1, 4)`
- 第 2 个请求占 `[4, 6)`

### 完整流程图

```text
SchedulerOutput
   |
   +--> num_scheduled_tokens
   +--> scheduled_new_reqs / scheduled_cached_reqs
   |
   v
update_states()
   |
   v
prepare_inputs()
   |
   +--> idx_mapping
   +--> block_tables
   +--> query_start_loc
   +--> input_ids / positions
   +--> slot_mappings
   +--> attn_metadata
   v
InputBatch
   |
   v
model forward / sample / postprocess
```

---

## 总结

1. `LLM` 只是用户入口，真正的前端核心是 `LLMEngine`，真正的后端核心是 `EngineCore`
2. `EngineCoreClient` 把单进程和多进程统一到了同一套前端接口之下，这是 vLLM 分层设计的关键
3. `EngineCore.step()` 负责“调度 + 执行 + 回写”闭环，是后端执行主循环的中心
4. `Executor` 负责把调度结果送到 worker，`Worker` 负责设备和运行时，`GPUModelRunner` 负责真正的模型执行
5. `GPUModelRunner.execute_model()` 可以拆成三步：选执行模式、更新请求状态、构建模型输入
6. `update_states()` 和 `prepare_inputs()` 不是辅助函数，而是把调度器结果变成可执行 batch 的核心桥梁

## 思考题

1. 如果把 `InputProcessor`、`Scheduler`、`ModelRunner` 都塞进一个类里，代码会在哪些地方变得难以维护？
2. `EngineCoreClient` 为什么要把 inproc 和 multiprocess 封到同一个抽象层里？如果没有这一层，上层 API 会变成什么样？
3. `GPUModelRunner.execute_model()` 为什么先做 `update_states()`，再做 `prepare_inputs()`？如果顺序反过来会出什么问题？
4. `prepare_inputs()` 里为什么需要 `idx_mapping`、`query_start_loc`、`slot_mappings` 三种不同索引？它们分别解决的是什么问题？
5. 如果你要给 vLLM 新增一个“请求级 trace 采集模块”，你更倾向把它挂在 `LLMEngine`、`EngineCore` 还是 `ModelRunner`？为什么？
