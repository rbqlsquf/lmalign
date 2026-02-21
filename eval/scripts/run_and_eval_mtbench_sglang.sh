#!/bin/bash

MODEL_DIR=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/sft
EVAL_SCRIPT=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/scripts/mt_beach_eval.sh
VLLM_SCRIPT=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/scripts/sglang.sh
RESULTS_BASE=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/sft_mtbench

VLLM_PID=""

cleanup_vllm() {
    if [ -n "$VLLM_PID" ]; then
        echo "Cleaning up vLLM processes..."
        pkill -P $VLLM_PID 2>/dev/null || true
        kill $VLLM_PID 2>/dev/null || true
        sleep 5
        pkill -9 -P $VLLM_PID 2>/dev/null || true
        kill -9 $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        fuser -k /dev/nvidia* 2>/dev/null || true
        sleep 5
        VLLM_PID=""
    fi

    echo "Waiting for port 8000 to be released..."
    while lsof -i :8000 > /dev/null 2>&1; do
        sleep 5
    done
    echo "Port 8000 released."
}

trap cleanup_vllm EXIT

EVALUATED=(checkpoint-1000)

while true; do
    NEW_FOUND=false
    for CP in $(ls -d $MODEL_DIR/checkpoint-* 2>/dev/null | sort -V); do
        MODEL_NAME=$(basename $CP)
        echo "DEBUG: MODEL_NAME='$MODEL_NAME'"
        if [[ " ${EVALUATED[@]} " =~ " ${MODEL_NAME} " ]]; then
            echo "DEBUG: Skipping $MODEL_NAME"
            continue
        fi

        echo "DEBUG: Evaluating $MODEL_NAME"
        NEW_FOUND=true
        MODEL_PATH=$CP
        RESULTS_DIR=$RESULTS_BASE
        cp /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/sft/tokenizer_config.json $MODEL_PATH/
        echo "===== Evaluating $MODEL_NAME ====="
        mkdir -p $RESULTS_DIR

        bash $VLLM_SCRIPT $MODEL_PATH > /tmp/vllm_${MODEL_NAME}.log 2>&1 &
        VLLM_PID=$!

        echo "Waiting for vLLM server..."
        until curl -s http://localhost:8000/health > /dev/null 2>&1; do
            sleep 5
        done
        echo "vLLM server ready."

        if ! bash $EVAL_SCRIPT $MODEL_NAME $RESULTS_DIR; then
            echo "ERROR: eval failed for $MODEL_NAME, exiting."
            exit 1
        fi

        cleanup_vllm

        echo "===== Done: $MODEL_NAME ====="
        EVALUATED+=("$MODEL_NAME")
        sleep 5
    done

    if [ "$NEW_FOUND" = false ]; then
        echo "No new checkpoints found, waiting..."
        sleep 1800
    fi
done