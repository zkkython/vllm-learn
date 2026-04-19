import os

os.environ["VLLM_USE_V1"] = "1"  # 必须在 import vllm 之前！
# 只有在外部没有显式指定时，默认使用第一张可见 GPU。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "Write a poem about China:",
    "Who won the world series in 2020?",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B", enforce_eager=True
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
