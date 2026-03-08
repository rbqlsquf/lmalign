#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# vLLM v0 API 사용 (더 안정적, Qwen3 호환성 좋음)
python -m vllm.entrypoints.openai.api_server \
    --model $1 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --port 12738 \
    --trust-remote-code