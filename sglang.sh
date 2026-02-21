#!/bin/bash

python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --context-length 32768 \
    --host 0.0.0.0 \
    --port 6000 \
    --data-parallel-size 4
    # --reasoning-parser deepseek-r1
