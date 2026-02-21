#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
vllm serve /data/ib-a100-cluster-a-pri-lmt_967/models/Qwen3-0.6B-Base \
    --data-parallel-size 2 \
    --max-model-len 32768 \
    --port 8000