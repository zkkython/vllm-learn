#!/usr/bin/env bash


# 临时取消所有代理，运行 benchmark
# no_proxy="*" HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" \
vllm bench serve \
    --model /home/kason/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
    --host 127.0.0.1 \
    --random-input-len 128 \
    --port 13311 \
    --request-rate 10 \
    --num-prompts 100 \
    --save-result \
    --result-dir ./bench_results \
    --label "qwen3-1.7b-test"
        
