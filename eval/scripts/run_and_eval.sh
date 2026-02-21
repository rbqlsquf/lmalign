#!/bin/bash

# Ensure python command is available
#sudo ln -sf $(which python3) /usr/local/bin/python

export NEMO_SKILLS_CONFIG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/configs"
export NEMO_SKILLS_DATA_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/datasets"

MODEL_DIR=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/sft-qwen3-1.7b-tulu3-sft-mixture_jj_same_kbscripts_bf16
EVAL_SCRIPT=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/scripts/eval.sh
VLLM_SCRIPT=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/scripts/vllm.sh
RESULTS_BASE=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/sft-qwen3-1.7b-tulu3-sft-mixture_jj_same_kbscripts_bf16
PORT=12738
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

    echo "Waiting for port $PORT to be released..."
    while lsof -i :$PORT > /dev/null 2>&1; do
        sleep 5
    done
    echo "Port $PORT released."
}

trap cleanup_vllm EXIT

EVALUATED=()

while true; do
    NEW_FOUND=false
    for CP in $(ls -d $MODEL_DIR/checkpoint-* 2>/dev/null | sort -V); do
        MODEL_NAME=$(basename $CP)

        if [[ " ${EVALUATED[@]} " =~ " ${MODEL_NAME} " ]]; then
            continue
        fi

        NEW_FOUND=true
        MODEL_PATH=$CP
        RESULTS_DIR=$RESULTS_BASE/$MODEL_NAME
        # cp /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/sft/tokenizer_config.json $MODEL_PATH/
        echo "===== Evaluating $MODEL_NAME ====="
        mkdir -p $RESULTS_DIR

        bash $VLLM_SCRIPT $MODEL_PATH > /tmp/vllm_${MODEL_NAME}.log 2>&1 &
        VLLM_PID=$!

        echo "Waiting for vLLM server..."
        until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
            sleep 5
        done
        echo "vLLM server ready."

        if ! bash $EVAL_SCRIPT $MODEL_PATH $PORT $RESULTS_DIR; then
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