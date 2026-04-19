python3 -m vllm.entrypoints.openai.api_server \
  --model "/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --port "13311" \
