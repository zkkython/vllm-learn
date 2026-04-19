# 课时1 - vLLM快速上手与环境搭建

> **关键源码文件**：
> - `vllm/vllm/entrypoints/llm.py` -- 离线推理入口 `LLM`，负责把用户参数装配成 `EngineArgs` 和 `LLMEngine`
> - `vllm/vllm/v1/engine/llm_engine.py` -- 离线引擎主控，负责请求注册、步进执行和结果回收
> - `vllm/vllm/entrypoints/cli/run_batch.py` / `vllm/vllm/entrypoints/openai/run_batch.py` -- 离线批处理 CLI 入口与 JSONL 批量执行流程
> - `vllm/vllm/entrypoints/cli/serve.py` -- 在线服务 CLI 入口，负责 headless / 单进程 / 多 API Server 模式选择
> - `vllm/vllm/entrypoints/openai/api_server.py` -- FastAPI 应用构建、AsyncLLM 客户端创建、HTTP 路由与服务启动
> - `vllm/vllm/benchmarks/latency.py` / `vllm/vllm/benchmarks/throughput.py` / `vllm/vllm/benchmarks/serve.py` -- 离线延迟、离线吞吐、在线服务压测

## 学习目标

1. 理解 vLLM 的两种基本使用方式：离线推理与在线服务部署
2. 掌握 `LLM`、`LLMEngine`、`serve`、`run-batch` 这些入口在源码中的职责分工
3. 能够从源码角度理解一次离线 `generate()` 调用和一次在线 HTTP 请求分别如何进入引擎
4. 掌握 vLLM 官方 benchmark 工具的三种典型用法：`latency`、`throughput`、`serve`
5. 能够独立完成最小可运行环境搭建、离线脚本部署、在线 API 服务部署和性能测试

---

## 1.1 安装与环境配置

### 原理讲解

从“快速上手”的角度看，vLLM 的环境配置并不只是把 Python 包装好那么简单，它实际上是在为两类入口准备同一套底层配置对象：

1. **离线入口**：用户在 Python 脚本里构造 `LLM(...)`
2. **在线入口**：用户在命令行里执行 `vllm serve ...`

这两个入口表面差别很大，但底层都要完成三件事：

- 收集用户参数，例如模型路径、张量并行数、显存占用比例、量化方式
- 把这些参数规范化成统一的配置对象
- 根据配置对象启动 `LLMEngine` / `AsyncLLM`，再进一步连接 `EngineCore`、`Executor`、`Worker`

因此，vLLM 的“环境配置”可以理解为两层：

- **运行环境层**：Python、PyTorch、CUDA、可见 GPU、依赖包
- **引擎配置层**：`EngineArgs` / `AsyncEngineArgs` / `VllmConfig`

很多同学第一次使用 vLLM 时只关心 `pip install`，但在读源码时更重要的是理解：**安装只是让模块可导入，配置对象才决定引擎最终如何工作。**

### 源码解析

**1. 离线入口把参数收敛到 `EngineArgs`**

在 `LLM.__init__` 中，可以看到用户传入的大部分参数最终都被收敛到 `EngineArgs`，然后继续构造 `LLMEngine`：

```python
# vllm/vllm/entrypoints/llm.py 第316-353行
engine_args = EngineArgs(
    model=model,
    runner=runner,
    convert=convert,
    tokenizer=tokenizer,
    tensor_parallel_size=tensor_parallel_size,
    dtype=dtype,
    quantization=quantization,
    gpu_memory_utilization=gpu_memory_utilization,
    kv_cache_memory_bytes=kv_cache_memory_bytes,
    compilation_config=compilation_config_instance,
    ...
)

self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
```

这里有两个关键点：

- `LLM` 不是直接操作 `Worker` 或 `ModelRunner`
- 用户级 API 先被标准化为 `EngineArgs`，再交给 `LLMEngine`

**2. `LLMEngine` 完成离线引擎核心组件装配**

`LLMEngine.__init__` 负责把 tokenizer、输入处理器、输出处理器和 EngineCore 客户端串起来：

```python
# vllm/vllm/v1/engine/llm_engine.py 第91-115行
self.input_processor = InputProcessor(self.vllm_config, tokenizer)
self.io_processor = get_io_processor(
    self.vllm_config,
    self.model_config.io_processor_plugin,
)

self.output_processor = OutputProcessor(
    self.tokenizer,
    log_stats=self.log_stats,
    stream_interval=self.vllm_config.scheduler_config.stream_interval,
)

self.engine_core = EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
)
```

也就是说，安装完成之后，离线 `LLM` 真正可运行的关键，不是某个单独函数，而是这组组件已经被正确装配出来。

**3. 在线入口和离线入口共用同一套思路**

`ServeSubcommand.cmd` 并不自己实现推理逻辑，而是根据部署模式选择不同启动路径：

```python
# vllm/vllm/entrypoints/cli/serve.py 第47-60行
@staticmethod
def cmd(args: argparse.Namespace) -> None:
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            uvloop.run(run_server(args))
```

这一段说明：在线服务也不是“另一套引擎”，而是“另一种外层壳”，底层仍然会进入 `AsyncEngineArgs -> VllmConfig -> AsyncLLM / EngineCore` 的装配过程。

### 代码示例

下面给出一个最小环境搭建流程。这个仓库里已经包含 `vllm/` 源码目录，所以最自然的方式是本地 editable 安装：

```bash
cd /home/kason/python_workspace/vllm-learn/vllm

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .

python -c "import vllm; print('vLLM import ok')"
python -c "import torch; print(torch.cuda.is_available())"
```

如果你想在开始前先验证 GPU 和 PyTorch 是否正常，可额外执行：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.device_count())"
```

### 架构总览

```text
                    vLLM 快速上手的统一装配路径
  =====================================================================

  用户参数
     |
     | Python: LLM(...)
     | CLI:    vllm serve / vllm run-batch
     v
  EngineArgs / AsyncEngineArgs
     |
     v
  VllmConfig
     |
     +--------------------------+
     |                          |
     v                          v
  LLMEngine                 AsyncLLM / API Server
     |                          |
     v                          v
  EngineCoreClient <------> EngineCore / Executor / Worker
     |
     v
  最终输出: RequestOutput / HTTP Response
```

---

## 1.2 离线推理的部署

### 原理讲解

离线推理的核心特点是：**没有常驻 HTTP 服务，脚本启动引擎、完成推理、拿到结果后直接退出。**

在 vLLM 里，离线推理至少有两种典型部署方式：

1. **Python API 模式**：在代码中直接使用 `LLM.generate()`
2. **批处理 CLI 模式**：使用 `vllm run-batch` 读取输入文件并输出结果文件

这两种方式的共同点是都围绕同一个离线引擎展开：

- `LLM` 负责接收用户 prompt 和采样参数
- `LLMEngine` 负责把 prompt 注册成内部请求
- `EngineCore` 负责调度与模型执行
- `OutputProcessor` 负责把内部输出整理为最终 `RequestOutput`

从学习角度看，Python API 模式更适合理解调用链，`run-batch` 更适合理解“工程化批量部署”。

### 源码解析

**1. `LLM.generate()` 是离线推理最核心的用户入口**

```python
# vllm/vllm/entrypoints/llm.py 第381-450行
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
) -> list[RequestOutput]:
    ...
    self._validate_and_add_requests(
        prompts=prompts,
        params=sampling_params,
        use_tqdm=use_tqdm,
        lora_request=lora_request,
        priority=priority,
    )

    outputs = self._run_engine(use_tqdm=use_tqdm)
    return self.engine_class.validate_outputs(outputs, RequestOutput)
```

这里可以把 `generate()` 理解为两段：

- 前半段：把外部 prompt 变成内部 request
- 后半段：驱动引擎循环运行，直到所有 request 完成

**2. `LLMEngine.add_request()` 把 prompt 正式注册到引擎**

```python
# vllm/vllm/v1/engine/llm_engine.py 第222-283行
def add_request(...):
    if isinstance(prompt, EngineCoreRequest):
        request = prompt
    else:
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            arrival_time,
            lora_request,
            tokenization_kwargs,
            trace_headers,
            priority,
        )

    self.output_processor.add_request(request, prompt_text, None, 0)
    self.engine_core.add_request(request)
```

这一步做了两件最关键的事：

- `input_processor.process_inputs(...)`：把用户输入整理成 `EngineCoreRequest`
- `engine_core.add_request(request)`：把请求送进真正的执行核心

**3. `LLMEngine.step()` 周期性拉取输出并做后处理**

```python
# vllm/vllm/v1/engine/llm_engine.py 第285-319行
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
    outputs = self.engine_core.get_output()

    processed_outputs = self.output_processor.process_outputs(
        outputs.outputs,
        engine_core_timestamp=outputs.timestamp,
        iteration_stats=iteration_stats,
    )

    self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    return processed_outputs.request_outputs
```

这说明离线推理不是“一次函数调用直接返回字符串”，而是：

1. 把请求加进引擎
2. 持续 `step()`
3. 逐步回收输出

**4. `run-batch` 复用了 OpenAI 兼容层，但依旧是离线批处理**

`vllm run-batch` 的 CLI 入口很薄，只负责把参数交给真正实现：

```python
# vllm/vllm/entrypoints/cli/run_batch.py 第27-46行
@staticmethod
def cmd(args: argparse.Namespace) -> None:
    from vllm.entrypoints.openai.run_batch import main as run_batch_main
    ...
    asyncio.run(run_batch_main(args))
```

真正的批量执行逻辑在 `openai/run_batch.py` 中：

```python
# vllm/vllm/entrypoints/openai/run_batch.py 第431-549行
async def run_batch(engine_client: EngineClient, args: Namespace) -> None:
    ...
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        request = BatchRequestInput.model_validate_json(request_json)

        if request.url == "/v1/chat/completions":
            response_futures.append(run_request(chat_handler_fn, request, tracker))
            tracker.submitted()
        elif request.url == "/v1/embeddings":
            response_futures.append(run_request(embed_handler_fn, request, tracker))
            tracker.submitted()
```

这个设计很有意思：`run-batch` 不是直接调用 `LLM.generate()`，而是复用 OpenAI 兼容协议对象，把 JSONL 文件里的每一行请求提交给对应 handler。这样做的好处是：

- 离线批处理与在线协议层尽量复用同一套请求格式
- 批量作业可以直接兼容 `/v1/chat/completions`、`/v1/embeddings` 这类接口风格

### 代码示例

**Python API：最小离线推理脚本**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=64,
)

outputs = llm.generate(
    ["请用三句话介绍 vLLM 的作用。"],
    sampling_params=sampling_params,
)

for output in outputs:
    print(output.outputs[0].text)
```

**CLI：批处理文件部署**

输入文件 `input.jsonl`：

```json
{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"请解释什么是 KV Cache。"}]}}
{"custom_id":"req-2","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"请解释什么是张量并行。"}]}}
```

执行命令：

```bash
vllm run-batch \
  -i input.jsonl \
  -o output.jsonl \
  --model Qwen/Qwen3-0.6B
```

### 架构总览

```text
                    离线推理的两种部署形态
  =====================================================================

  Python 脚本                                  JSONL 批处理
  LLM.generate()                               vllm run-batch
      |                                             |
      v                                             v
  LLM._validate_and_add_requests()            openai.run_batch.run_batch()
      |                                             |
      v                                             v
  LLMEngine.add_request()                     OpenAI Handler / EngineClient
      |                                             |
      +---------------------> EngineCore <----------+
                               |
                               v
                         Scheduler / Executor / Worker
                               |
                               v
                       OutputProcessor / BatchRequestOutput
```

---

## 1.3 在线推理服务部署与调用

### 原理讲解

在线推理与离线推理的最大区别，不是模型执行方式变了，而是外层多了一个**常驻 API Server**：

- 离线模式：脚本自己调用引擎，主动拉取结果
- 在线模式：HTTP 请求进入 FastAPI，再由服务层去调用异步引擎

在 vLLM 中，在线服务入口是 `vllm serve`。它至少做四件事：

1. 解析并校验 CLI 参数
2. 创建监听 socket，准备对外提供 HTTP 服务
3. 创建 `AsyncLLM` / `EngineClient`
4. 构建 FastAPI 应用，把 `/v1/chat/completions` 等路由挂到对应 handler 上

因此，“在线服务部署”本质上是在离线引擎外层再包一层 API Server 外壳。

### 源码解析

**1. `vllm serve` 先选择启动模式**

```python
# vllm/vllm/entrypoints/cli/serve.py 第47-60行
@staticmethod
def cmd(args: argparse.Namespace) -> None:
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            uvloop.run(run_server(args))
```

这里体现了 vLLM 在线部署的三个典型模式：

- `run_headless(args)`：只起引擎，不起 HTTP 服务
- `run_multi_api_server(args)`：多个 API Server 进程共用底层引擎
- `run_server(args)`：最常见的单服务进程模式

**2. `setup_server()` 先抢占端口，再初始化引擎**

```python
# vllm/vllm/entrypoints/openai/api_server.py 第1347-1388行
def setup_server(args):
    logger.info("vLLM API server version %s", VLLM_VERSION)
    log_non_default_args(args)
    validate_api_server_args(args)

    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)

    set_ulimit()
    ...
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock
```

这一步的关键设计是：**先绑定端口，再启动引擎**。这样可以减少多进程和 Ray 场景下的端口竞争问题。

**3. `run_server_worker()` 创建异步引擎并启动 HTTP 服务**

```python
# vllm/vllm/entrypoints/openai/api_server.py 第1401-1448行
async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    async with build_async_engine_client(
        args,
        client_config=client_config,
    ) as engine_client:
        app = build_app(args)
        await init_app_state(engine_client, app.state, args)

        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            ...
        )
```

这里可以非常清楚地看到在线服务的三层结构：

- `build_async_engine_client(...)`：创建异步推理引擎
- `build_app(args)`：构建 FastAPI 应用对象
- `serve_http(...)`：真正开始监听 HTTP 请求

**4. `build_app()` 负责组装 FastAPI 路由与中间件**

```python
# vllm/vllm/entrypoints/openai/api_server.py 第960-1070行
def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    ...
    register_sagemaker_routes(router)
    app.include_router(router)
    register_pooling_api_routers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
```

这说明在线服务层的职责不是做模型计算，而是：

- 注册路由
- 注入中间件
- 组织异常处理
- 把请求转交给具体的 serving handler

**5. `init_app_state()` 把模型服务对象挂到 `app.state` 上**

```python
# vllm/vllm/entrypoints/openai/api_server.py 第1073-1306行
async def init_app_state(engine_client: EngineClient, state: State, args: Namespace):
    state.engine_client = engine_client
    supported_tasks = await engine_client.get_supported_tasks()

    state.openai_serving_models = OpenAIServingModels(...)
    state.openai_serving_chat = OpenAIServingChat(...) if "generate" in supported_tasks else None
    state.openai_serving_completion = OpenAIServingCompletion(...) if "generate" in supported_tasks else None
    state.openai_serving_embedding = OpenAIServingEmbedding(...) if "embed" in supported_tasks else None
```

因此，当一个 HTTP 请求来到 `/v1/chat/completions` 时，真正处理请求的是已经挂在 `app.state` 上的 `OpenAIServingChat` 对象，而不是路由函数本身。

### 代码示例

**启动一个最小 OpenAI 兼容服务**

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

**用 `curl` 调用 `/v1/chat/completions`**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请解释什么是流式推理。"}
    ],
    "temperature": 0.7,
    "max_tokens": 64,
    "stream": false
  }'
```

**流式调用示例**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请按条目介绍 vLLM 的核心优势。"}
    ],
    "stream": true,
    "max_tokens": 64
  }'
```

### 架构总览

```text
                       在线服务的请求路径
  =====================================================================

  Client / curl / SDK
          |
          v
  FastAPI Router (/v1/chat/completions)
          |
          v
  OpenAIServingChat / OpenAIServingCompletion
          |
          v
  AsyncLLM / EngineClient
          |
          v
  EngineCore
          |
          v
  Scheduler -> Executor -> Worker -> ModelRunner
          |
          v
  RequestOutput / StreamingResponse
```

---

## 1.4 离线推理与在线推理的性能测试

### 原理讲解

当环境已经能跑起来后，下一步通常不是“继续堆功能”，而是先回答三个问题：

1. 单批次推理延迟是多少？
2. 批量离线推理吞吐是多少？
3. 在线服务在真实请求速率下的 TTFT、TPOT、ITL 是多少？

vLLM 官方把这三类问题分别交给三套 benchmark：

- `vllm bench latency`：测单批次延迟
- `vllm bench throughput`：测离线吞吐
- `vllm bench serve`：测在线服务吞吐与流式指标

这三者关注点不同：

- **latency**：更像“单次/单批次执行有多快”
- **throughput**：更像“离线大量请求时单位时间能跑多少 token / request”
- **serve**：更像“在线服务在给定请求速率下的用户体验如何”

### 源码解析

**1. benchmark 总入口只是做命令分发**

```python
# vllm/vllm/entrypoints/cli/benchmark/main.py 第17-52行
class BenchmarkSubcommand(CLISubcommand):
    name = "bench"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)
```

也就是说，`vllm bench ...` 的真正逻辑都在各个子命令实现里。

**2. 离线延迟测试：构造固定 batch，循环测 `llm.generate()`**

```python
# vllm/vllm/benchmarks/latency.py 第80-170行
def main(args: argparse.Namespace):
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(...)
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts = [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    def llm_generate():
        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
```

这个脚本的设计重点是：

- 先 warmup，再正式测量
- 统一 batch size、input len、output len
- 默认关闭 prefix caching，避免结果被缓存机制“美化”

**3. 离线吞吐测试：批量构造 request，再测总执行耗时**

```python
# vllm/vllm/benchmarks/throughput.py 第42-122行
def run_vllm(requests, n, engine_args, do_profile, disable_detokenize=False):
    llm = LLM(**dataclasses.asdict(engine_args))
    ...
    for request in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            )
        )

    start = time.perf_counter()
    outputs = llm.generate(
        prompts, sampling_params, lora_request=lora_requests, use_tqdm=True
    )
    end = time.perf_counter()
    return end - start, outputs
```

这里测的不是“单次响应时间”，而是“这一整批请求从开始到完成总共耗时多少”，因此更适合观察总吞吐。

**4. 在线服务压测：异步发请求，并统计 TTFT / TPOT / ITL**

```python
# vllm/vllm/benchmarks/serve.py 第137-235行
async def get_request(input_requests, request_rate, burstiness=1.0, ...):
    ...
    for request_index, request in enumerate(input_requests):
        ...
        if sleep_interval_s > 0:
            await asyncio.sleep(sleep_interval_s)
        yield request, request_rates[request_index]
```

这一步负责按给定请求速率“发流量”。

真正的 OpenAI 兼容请求函数在 `endpoint_request_func.py` 中：

```python
# vllm/vllm/benchmarks/lib/endpoint_request_func.py 第141-252行
async def async_request_openai_completions(...):
    payload = {
        "model": ...,
        "prompt": request_func_input.prompt,
        "max_tokens": request_func_input.output_len,
        "stream": True,
    }

    async with session.post(url=api_url, json=payload, headers=headers) as response:
        async for chunk_bytes in response.content.iter_any():
            ...
            if not first_chunk_received:
                output.ttft = time.perf_counter() - st
            else:
                output.itl.append(timestamp - most_recent_timestamp)
```

这正是在线 benchmark 能给出 `TTFT`、`ITL`、`TPOT` 的原因：它不是只看最终响应，而是逐个流式 chunk 统计时间。

**5. 在线 benchmark 的结果会沉淀成 JSON**

当前仓库已经有一份在线压测结果示例：

```json
{
  "completed": 100,
  "failed": 0,
  "request_throughput": 9.187555620238113,
  "output_throughput": 1176.0071193904785,
  "mean_ttft_ms": 32.65031902119517,
  "p99_ttft_ms": 55.31776260584593,
  "mean_tpot_ms": 7.466669174160545
}
```

这个结果来自 `bench_results/qwen3-1.7b-test-10.0qps-Qwen3-0.6B-20260406-183757.json`，可以把它理解为：

- `request_throughput`：每秒完成多少请求
- `output_throughput`：每秒输出多少 token
- `mean_ttft_ms`：首 token 平均时间
- `mean_tpot_ms`：后续 token 平均时间

### 代码示例

**1. 离线延迟测试**

```bash
vllm bench latency \
  --model Qwen/Qwen3-0.6B \
  --input-len 256 \
  --output-len 64 \
  --batch-size 8 \
  --num-iters-warmup 5 \
  --num-iters 20 \
  --output-json latency.json
```

**2. 离线吞吐测试**

```bash
vllm bench throughput \
  --model Qwen/Qwen3-0.6B \
  --dataset-name random \
  --num-prompts 128 \
  --input-len 256 \
  --output-len 64
```

**3. 在线服务压测**

先启动服务：

```bash
vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
```

再发压测：

```bash
vllm bench serve \
  --backend openai \
  --model Qwen/Qwen3-0.6B \
  --endpoint /v1/completions \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --num-prompts 100 \
  --request-rate 10.0 \
  --save-result
```

### 数据流图

```text
                    vLLM 三类性能测试的关注点
  =====================================================================

  vllm bench latency
      |
      v
  构造固定 dummy batch
      |
      v
  重复执行 llm.generate()
      |
      v
  平均延迟 / 分位数


  vllm bench throughput
      |
      v
  构造请求集合 requests[]
      |
      v
  一次性批量提交到 LLM
      |
      v
  总耗时 -> request throughput / token throughput


  vllm bench serve
      |
      v
  按 request_rate 异步发 HTTP 请求
      |
      v
  流式读取 SSE / chunk
      |
      v
  TTFT / ITL / TPOT / E2EL / 并发峰值
```

---

## 总结

本章完成了 vLLM 快速上手所需的四个基础动作：

1. **安装与环境配置**：理解了运行环境和引擎配置是两层概念，`LLM` / `serve` 最终都会收敛到统一配置对象
2. **离线推理部署**：掌握了 Python `LLM.generate()` 和 `vllm run-batch` 两种典型离线入口，以及它们如何进入 `LLMEngine`
3. **在线推理服务部署**：掌握了 `vllm serve` 的启动模式、FastAPI 应用构建过程，以及 HTTP 请求如何进入异步引擎
4. **性能测试**：掌握了 `latency`、`throughput`、`serve` 三种 benchmark 的区别和适用场景
5. **统一视角**：无论是离线脚本、批处理作业还是在线 API 服务，底层都依赖同一套引擎组件
6. **后续学习入口**：课时2会进一步拆开这套统一引擎，重点看 AsyncLLM 和 EngineCore 之间的通信与流式执行

---

## 思考题

1. **为什么说 vLLM 的安装和环境配置不只是 `pip install`？** 请结合 `LLM.__init__` 中的 `EngineArgs` 构造过程回答。

2. **`run-batch` 明明是离线批处理，为什么它却复用了 OpenAI 协议层对象？** 这种设计相比直接调用 `LLM.generate()` 有什么工程优势？

3. **在 `setup_server()` 中，vLLM 为什么要先绑定端口，再启动引擎？** 如果把这个顺序反过来，在多进程或 Ray 场景下可能出现什么问题？

4. **`vllm bench latency` 和 `vllm bench throughput` 都会调用 `LLM.generate()`，它们测出来的指标为什么仍然完全不同？** 请从“测量对象”和“请求组织方式”两个角度解释。

5. **设计题**：如果你要给团队写一个“一键验证部署是否正常”的脚本，你会如何串联 `导入验证 -> 离线最小推理 -> 在线服务启动 -> HTTP 调用 -> benchmark 烟雾测试` 这五步？
