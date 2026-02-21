#!/bin/bash

python3 -m sglang.launch_server \
    --model-path $1 \
    --context-length 32768 \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 4
    # --reasoning-parser deepseek-r1
