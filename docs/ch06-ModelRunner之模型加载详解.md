# 课时6 - vLLM 核心组件-ModelRunner之模型加载详解

> **关键源码文件**：
> - `vllm/vllm/v1/worker/gpu/model_runner.py` -- `GPUModelRunner.load_model()`，是真正触发模型加载链的调用方
> - `vllm/vllm/model_executor/model_loader/base_loader.py` -- `BaseModelLoader`，定义模型加载总流程
> - `vllm/vllm/model_executor/model_loader/__init__.py` -- `get_model_loader()`、`register_model_loader()` 和 load format 分发
> - `vllm/vllm/model_executor/model_loader/default_loader.py` -- `DefaultModelLoader`，实现主流 HF/safetensors/bin/pt 权重加载
> - `vllm/vllm/model_executor/model_loader/utils.py` -- `initialize_model()`、`get_model_architecture()`、`process_weights_after_loading()`
> - `vllm/vllm/model_executor/models/registry.py` -- `ModelRegistry`，负责模型类注册与架构解析
> - `vllm/vllm/model_executor/models/utils.py` / `vllm/vllm/model_executor/models/qwen3.py` -- `AutoWeightsLoader` 与真实模型的 `load_weights()` 示例

## 学习目标

1. 理解 vLLM 模型加载为什么要拆成“选模型类”和“装权重”两个阶段
2. 掌握 `initialize_model()` 如何从 `ModelConfig` 找到最终模型类并实例化
3. 理解 `ModelRegistry` 如何把 HuggingFace `architectures` 映射到 vLLM 模型实现
4. 掌握 `DefaultModelLoader` 的权重准备、权重迭代和权重装载流程
5. 能够从源码层面理解“新增一个模型实现”和“自定义一个权重加载逻辑”分别应该改哪里

---

## 6.1 前言

### 原理讲解

很多同学第一次读 vLLM 的模型加载代码时，容易把两件事混在一起：

1. **加载哪个模型类**
2. **把 checkpoint 权重灌到这个模型类里**

实际上这两件事在 vLLM 里是明确拆开的。

原因很简单：

- 同一个 HuggingFace 架构名，可能对应 vLLM 自己实现的模型类
- 也可能退回到 Transformers backend
- 甚至还可能经过 convert wrapper、量化配置、pipeline parallel 包装

所以 vLLM 的加载主链大致是：

1. 先根据 `ModelConfig` 解析出模型类
2. 再实例化空模型
3. 再通过 `ModelLoader` 读取权重
4. 最后做量化后处理、attention 权重后处理等收尾

这章的重点，就是把这条链拆开。

### 源码解析

从调用关系上看，这条加载链真正是由 `ModelRunner` 发起的：

```python
# vllm/vllm/v1/worker/gpu/model_runner.py :: GPUModelRunner.load_model
with DeviceMemoryProfiler() as m:
    model_loader = get_model_loader(self.vllm_config.load_config)
    self.model = model_loader.load_model(
        vllm_config=self.vllm_config,
        model_config=self.vllm_config.model_config,
    )
```

所以虽然大部分源码落在 `model_loader/` 目录下，但它实际属于 `ModelRunner` 的模型装配阶段。

最顶层的统一入口其实就在 `BaseModelLoader.load_model()`：

```python
# vllm/vllm/model_executor/model_loader/base_loader.py :: BaseModelLoader.load_model
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(
            vllm_config=vllm_config, model_config=model_config
        )

    self.load_weights(model, model_config)
    process_weights_after_loading(model, model_config, target_device)
return model.eval()
```

这段代码已经把整个加载过程分成了三段：

- `initialize_model()`：构造模型骨架
- `load_weights()`：装 checkpoint 权重
- `process_weights_after_loading()`：做后处理

### 代码示例

```python
class SimpleLoader:
    def initialize_model(self):
        return {"layers": "empty model"}

    def load_weights(self, model):
        model["weights"] = "loaded"

    def process_after_loading(self, model):
        model["ready"] = True

    def load_model(self):
        model = self.initialize_model()
        self.load_weights(model)
        self.process_after_loading(model)
        return model


print(SimpleLoader().load_model())
```

### 架构总览

```text
                 vLLM 模型加载主链
  ====================================================================

  LoadConfig / ModelConfig
       |
       v
  get_model_loader()
       |
       v
  BaseModelLoader.load_model()
       |
       +--> initialize_model()
       +--> load_weights()
       +--> process_weights_after_loading()
       v
  可执行的 nn.Module
```

---

## 6.2 initialize_model初始化模型

### 原理讲解

`initialize_model()` 做的事情并不是“下载模型”或“读 checkpoint”，而是：

**根据当前配置，决定应该实例化哪个模型类，并把这个空模型对象构造出来。**

这里最容易忽略的一点是：vLLM 并不是简单地 `import 某个模型类然后 new 一下`。

它还会考虑：

- `hf_config.architectures`
- `model_impl` 是 `vllm`、`auto` 还是 `transformers`
- 是否需要 convert 成 embedding / classify 模型
- 是否要把 quant config 注入模型类
- 模型类是不是符合 vLLM 的“新式构造签名”

### 源码解析

**1. `initialize_model()` 先解析模型类**

```python
# vllm/vllm/model_executor/model_loader/utils.py :: initialize_model
if model_class is None:
    model_class, _ = get_model_architecture(model_config)

if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)
```

这说明模型类不是调用方直接塞进来的，而是通常由 `get_model_architecture()` 根据配置解析出来。

**2. 新式模型类要求接受 `vllm_config` 和 `prefix`**

```python
signatures = inspect.signature(model_class.__init__)
all_params = [param.name for param in signatures.parameters.values()]
if "vllm_config" in all_params and "prefix" in all_params:
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        return model_class(vllm_config=vllm_config, prefix=prefix)
```

这就是当前 vLLM 推荐的新模型写法。

**3. `get_model_architecture()` 会走到 `ModelRegistry.resolve_model_cls()`**

```python
# vllm/vllm/model_executor/model_loader/utils.py :: _get_model_architecture
architectures = getattr(model_config.hf_config, "architectures", [])
model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

也就是说，HuggingFace 配置里的 `architectures` 字段，最终会通过 registry 映射到 vLLM 的模型类。

**4. 构造完空模型后，还会有一个统一后处理阶段**

```python
# vllm/vllm/model_executor/model_loader/utils.py :: process_weights_after_loading
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
```

所以初始化不是终点，它只是后续权重加载和量化后处理的起点。

### 代码示例

```python
class MyModel:
    def __init__(self, *, vllm_config, prefix=""):
        self.cfg = vllm_config
        self.prefix = prefix


def initialize_model_like(model_class, vllm_config, prefix=""):
    return model_class(vllm_config=vllm_config, prefix=prefix)


print(initialize_model_like(MyModel, {"dtype": "bf16"}, prefix="model"))
```

---

## 6.3 注册新模型的过程

### 原理讲解

vLLM 解析模型类时，并不是去全仓库盲搜“哪个类名看起来像 Qwen”。它依赖一张注册表。

这张注册表的作用是：

- 把 HuggingFace 架构名映射到 `模块路径 + 类名`
- 支持懒加载，避免主进程过早导入模型代码
- 必要时退回到 Transformers backend

所以“注册模型”本质上就是告诉 vLLM：

**当 `architectures = [...]` 里出现某个名字时，你应该去加载哪个实现。**

### 源码解析

**1. 内置模型先在 `_VLLM_MODELS` 里注册**

```python
# vllm/vllm/model_executor/models/registry.py
_TEXT_GENERATION_MODELS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    ...
}

ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm.model_executor.models.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)
```

这里说明两件事：

- 静态注册表的键是 HuggingFace 架构名
- 值默认不是直接类对象，而是懒加载描述

**2. `register_model()` 允许运行时注册外部模型**

```python
# vllm/vllm/model_executor/models/registry.py :: _ModelRegistry.register_model
def register_model(self, model_arch: str, model_cls: type[nn.Module] | str) -> None:
    if isinstance(model_cls, str):
        model = _LazyRegisteredModel(*split_str)
    elif isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
        model = _RegisteredModel.from_model_cls(model_cls)
```

这意味着外部模型有两种注册方式：

- 直接传类对象
- 传 `<module>:<class>` 字符串做 lazy import

**3. 真正解析时走 `resolve_model_cls()`**

```python
# vllm/vllm/model_executor/models/registry.py :: resolve_model_cls
for arch in architectures:
    normalized_arch = self._normalize_arch(arch, model_config)
    model_cls = self._try_load_model_cls(normalized_arch)
    if model_cls is not None:
        return (model_cls, arch)
```

所以 registry 并不是“按单个字符串硬匹配”这么简单，它还会做：

- arch 归一化
- auto / transformers fallback
- dynamic module 解析

### 代码示例

```python
class SimpleRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, arch, model_cls):
        self.models[arch] = model_cls

    def resolve_model_cls(self, architectures):
        for arch in architectures:
            if arch in self.models:
                return self.models[arch]
        raise ValueError("unsupported architecture")


registry = SimpleRegistry()
registry.register_model("ToyForCausalLM", "toy_module:ToyForCausalLM")
print(registry.resolve_model_cls(["ToyForCausalLM"]))
```

---

## 6.4 让 vLLM 支持一个新的模型

### 原理讲解

如果你要让 vLLM 支持一个新模型，通常至少要补上三样东西：

1. **模型类本身**：能够 forward、compute logits、load weights
2. **模型注册**：让 `ModelRegistry` 能找到它
3. **权重映射逻辑**：让 checkpoint 参数名能正确落到 vLLM 模型参数上

换句话说，“支持一个新模型”绝不只是往注册表里加一行名字。

### 源码解析

**1. 新模型类建议采用新式构造签名**

`Qwen3Model` 和 `Qwen3ForCausalLM` 是一个很标准的例子：

```python
# vllm/vllm/model_executor/models/qwen3.py
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )
```

外层可执行模型也同样遵守这个签名：

```python
class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...
```

**2. 这个模型类至少要能 forward 和 load_weights**

```python
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    hidden_states = self.model(
        input_ids, positions, intermediate_tensors, inputs_embeds
    )
    return hidden_states

def compute_logits(self, hidden_states):
    return self.logits_processor(self.lm_head, hidden_states)
```

```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
    )
    return loader.load_weights(weights)
```

**3. 最后还要把架构名接到 registry 上**

```python
_TEXT_GENERATION_MODELS = {
    ...
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
}
```

这三步合起来，才算真正“让 vLLM 支持一个新的模型”。

### 代码示例

```python
class ToyForCausalLM:
    def __init__(self, *, vllm_config, prefix=""):
        self.cfg = vllm_config
        self.prefix = prefix

    def forward(self, input_ids, positions, **kwargs):
        return input_ids

    def compute_logits(self, hidden_states):
        return hidden_states

    def load_weights(self, weights):
        loaded = {name for name, _ in weights}
        return loaded


def add_toy_model(registry):
    registry["ToyForCausalLM"] = ("toy", "ToyForCausalLM")
```

### 架构总览

```text
              让 vLLM 支持新模型的最小闭环
  ====================================================================

  新模型类文件
    - __init__(vllm_config, prefix)
    - forward()
    - compute_logits()
    - load_weights()
        |
        v
  注册到 ModelRegistry
        |
        v
  initialize_model() 能实例化
        |
        v
  DefaultModelLoader 能把 checkpoint 权重灌进去
```

---

## 6.5 模型权重的加载流程

### 原理讲解

模型类找到之后，下一步才是读权重。

在 vLLM 里，这一段被 `ModelLoader` 抽象了。不同 load format 可以用不同 loader，但默认最常见的是 `DefaultModelLoader`。

它的主流程可以概括成：

1. 准备 checkpoint 文件列表
2. 选权重迭代器
3. 逐个产生 `(name, tensor)`
4. 交给模型自己的 `load_weights()`
5. 统计哪些权重已经成功加载

### 源码解析

**1. `GPUModelRunner.load_model()` 先拿到具体 loader**

```python
# vllm/vllm/v1/worker/gpu/model_runner.py :: GPUModelRunner.load_model
model_loader = get_model_loader(self.vllm_config.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config,
    model_config=self.vllm_config.model_config,
)
```

也就是说，ModelRunner 本身并不关心 `safetensors` 还是 `pt`，它只负责把 `LoadConfig` 交给 loader 工厂。

**2. 先由 `get_model_loader()` 选 loader**

```python
# vllm/vllm/model_executor/model_loader/__init__.py
_LOAD_FORMAT_TO_MODEL_LOADER = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "safetensors": DefaultModelLoader,
    "tensorizer": TensorizerLoader,
    ...
}

def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
```

**3. `DefaultModelLoader._prepare_weights()` 负责准备文件**

```python
# vllm/vllm/model_executor/model_loader/default_loader.py :: _prepare_weights
if load_format == "hf":
    allow_patterns = ["*.safetensors", "*.bin"]
elif load_format == "safetensors":
    allow_patterns = ["*.safetensors"]
elif load_format == "pt":
    allow_patterns = ["*.pt"]
```

然后它会：

- 判断本地/远端
- 必要时从 HF 下载
- 过滤重复 safetensors 文件
- 过滤推理不需要的权重文件

**4. `_get_weights_iterator()` 决定用哪种迭代器**

```python
if self.load_config.load_format == "npcache":
    weights_iterator = np_cache_weights_iterator(...)
elif use_safetensors:
    weights_iterator = safetensors_weights_iterator(...)
else:
    weights_iterator = pt_weights_iterator(...)
```

这里的重点是：vLLM 尽量把“读取权重文件”做成流式 iterator，而不是一次性全塞进内存。

**5. `load_weights()` 最终把 iterator 交给模型本身**

```python
# vllm/vllm/model_executor/model_loader/default_loader.py :: load_weights
weights_to_load = {name for name, _ in model.named_parameters()}
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

也就是说，loader 负责“产出 checkpoint 权重流”，真正知道“这个名字应该写到模型哪一层”的，是模型自己的 `load_weights()`。

### 代码示例

```python
def fake_weight_iterator():
    yield "model.layers.0.weight", "tensor0"
    yield "model.layers.1.weight", "tensor1"


class ToyModel:
    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            print("loading", name, tensor)
            loaded.add(name)
        return loaded


ToyModel().load_weights(fake_weight_iterator())
```

### 数据流图

```text
                 DefaultModelLoader 权重加载主链
  ====================================================================

  LoadConfig
     |
     v
  _prepare_weights()
     |
     v
  _get_weights_iterator()
     |
     v
  get_all_weights()
     |
     v
  model.load_weights(iterator)
     |
     v
  loaded_weights / strict check / post process
```

---

## 6.6 自定义模型权重加载函数

### 原理讲解

这一节最重要的结论是：

**vLLM 并不要求所有模型都用同一套权重装载逻辑。**

它给了两级自定义入口：

1. **模块级自定义**：模型类自己实现 `load_weights()`
2. **参数级自定义**：某个参数对象挂一个 `weight_loader`

这样做的原因是，不同模型在权重命名和参数打包方式上差异很大：

- 有的模型把 `q/k/v` 分开存
- 有的模型把它们打包成一个参数
- 有的模型权重名和 HF 完全一致
- 有的模型要做额外映射、跳过或拆分

### 源码解析

**1. 最常见的写法是模型类里用 `AutoWeightsLoader`**

```python
# vllm/vllm/model_executor/models/qwen3.py :: Qwen3ForCausalLM.load_weights
def load_weights(self, weights):
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
    )
    return loader.load_weights(weights)
```

这说明大多数模型不需要手写一个巨长的参数装载循环，只要把模型树交给 `AutoWeightsLoader` 即可。

**2. `AutoWeightsLoader` 会递归遍历 module / param**

```python
# vllm/vllm/model_executor/models/utils.py :: AutoWeightsLoader.load_weights
autoloaded_weights = set(self._load_module("", self.module, weights))
return autoloaded_weights
```

它内部会：

- 先按前缀分组权重名
- 找到对应 child module 或 parameter
- 递归下钻
- 必要时调用子模块自己的 `load_weights()`

**3. 参数级别还可以挂 `weight_loader`**

```python
# vllm/vllm/model_executor/models/utils.py :: AutoWeightsLoader._load_param
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, weight_data)
```

这给了 fused 参数、量化参数非常大的灵活性。

**4. 如果需要重命名，还可以先过 `WeightsMapper`**

```python
# vllm/vllm/model_executor/models/utils.py :: AutoWeightsLoader.load_weights
if mapper is not None:
    weights = mapper.apply(weights)
```

所以自定义权重加载的手段并不止一种：

- 模块级 `load_weights`
- `AutoWeightsLoader`
- `WeightsMapper`
- 参数级 `weight_loader`

### 代码示例

```python
class Param:
    def __init__(self):
        self.data = None


def custom_weight_loader(param, weight):
    param.data = f"loaded<{weight}>"


class ToyModel:
    def __init__(self):
        self.weight = Param()
        self.weight.weight_loader = custom_weight_loader

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            if name == "weight":
                self.weight.weight_loader(self.weight, tensor)
                loaded.add(name)
        return loaded


model = ToyModel()
model.load_weights([("weight", "ckpt_tensor")])
print(model.weight.data)
```

---

## 总结

本章重点拆解了 vLLM 的模型加载总链路：

1. **模型加载先选类，再装权重**：`initialize_model()` 和 `load_weights()` 是两个清晰分离的阶段
2. **`initialize_model()` 依赖 registry 解析模型类**：它会结合 `architectures`、`model_impl`、`convert_type`、量化配置等因素选出最终实现
3. **模型注册是新增模型支持的入口**：无论是静态修改 `_VLLM_MODELS`，还是运行时 `register_model()`，本质上都是在告诉 vLLM“这个架构名该找谁”
4. **`DefaultModelLoader` 负责 checkpoint 文件与权重 iterator**：它解决的是“从哪里读”“按什么格式读”，而不是“参数最终落到模型哪一层”
5. **模型自己的 `load_weights()` 负责最终落盘**：这使 vLLM 可以适配 packed 参数、量化参数、tie embeddings 等多种特殊情况
6. **自定义入口很多层**：`BaseModelLoader`、`ModelRegistry`、模型类 `load_weights()`、`AutoWeightsLoader`、参数级 `weight_loader` 都可以按需扩展

---

## 思考题

1. **为什么 vLLM 要把“解析模型类”和“读取权重文件”拆成两个阶段？** 如果把两者硬写在一起，会给扩展性带来什么问题？

2. **`ModelRegistry` 为什么默认使用 `_LazyRegisteredModel` 而不是直接导入所有模型类？** 这和多进程 / CUDA 初始化有什么关系？

3. **`DefaultModelLoader` 为什么要把权重读取做成 iterator？** 这种设计对大模型 checkpoint 有什么实际收益？

4. **模型类自己的 `load_weights()` 为什么仍然必要？** 如果完全靠 `DefaultModelLoader` 直接 `state_dict.load_state_dict()`，哪些模型会最先出问题？

5. **设计题**：如果你现在要给 vLLM 加一个新模型，且这个模型 checkpoint 里的参数名和 vLLM 实现不一致，你会优先选择哪种扩展方式：`WeightsMapper`、模型类 `load_weights()` 还是自定义 `ModelLoader`？为什么？
