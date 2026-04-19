#!/usr/bin/env bash
# 1. 查看可用模型
curl -s --noproxy '*' http://127.0.0.1:13311/v1/models | jq .

# 2. 走 chat/completions 生成文本
curl -s --noproxy '*' http://127.0.0.1:13311/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用20字介绍vLLM"}],
    "max_tokens": 30,
    "temperature": 0.6
  }' | jq -r '.choices[0].message.content'
