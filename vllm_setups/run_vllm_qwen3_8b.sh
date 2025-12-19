#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --dtype auto \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --enforce_eager \
    --gpu-memory-utilization "0.95" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning_parser qwen3 \
    --port 8102 \
    --disable-log-requests \
    --max_num_batched_tokens 16384