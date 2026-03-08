#!/bin/bash

# Ensure python command is available
#sudo ln -sf $(which python3) /usr/local/bin/python

export NEMO_SKILLS_CONFIG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/configs"
export NEMO_SKILLS_DATA_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/datasets"

MODEL_DIR=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/GROP/IF_open_instruct_check_verify_new
EVAL_SCRIPT=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-eval/scripts/eval.sh
SGLANG_SCRIPT=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-eval/scripts/sglang.sh
RESULTS_BASE=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-eval/results/GRPO/IF_open_instruct_check_verify_new
PORT=8000
SGLANG_PID=""

cleanup_sglang() {
    if [ -n "$SGLANG_PID" ]; then
        echo "Cleaning up SGLang processes..."
        pkill -P $SGLANG_PID 2>/dev/null || true
        kill $SGLANG_PID 2>/dev/null || true
        sleep 5
        pkill -9 -P $SGLANG_PID 2>/dev/null || true
        kill -9 $SGLANG_PID 2>/dev/null || true
        wait $SGLANG_PID 2>/dev/null || true
        fuser -k /dev/nvidia* 2>/dev/null || true
        sleep 5
        SGLANG_PID=""
    fi

    # Kill leftover eval child processes (e.g. infinite-loop sandbox code)
    echo "Cleaning up orphaned python /tmp/tmp*.py processes..."
    pkill -f 'python3 /tmp/tmp.*\.py' 2>/dev/null || true
    sleep 1
    pkill -9 -f 'python3 /tmp/tmp.*\.py' 2>/dev/null || true
    # Remove leftover temp python files
    rm -f /tmp/tmp*.py 2>/dev/null || true

    echo "Waiting for port $PORT to be released..."
    while lsof -i :$PORT > /dev/null 2>&1; do
        sleep 5
    done
    echo "Port $PORT released."
}

trap cleanup_sglang EXIT

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

        bash $SGLANG_SCRIPT $MODEL_PATH > /tmp/sglang_${MODEL_NAME}.log 2>&1 &
        SGLANG_PID=$!

        echo "Waiting for SGLang server..."
        until curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
            sleep 5
        done
        echo "SGLang server ready."

        if ! bash $EVAL_SCRIPT $MODEL_PATH $PORT $RESULTS_DIR; then
            echo "ERROR: eval failed for $MODEL_NAME, exiting."
            # Still clean up orphaned processes before exiting
            pkill -9 -f 'python3 /tmp/tmp.*\.py' 2>/dev/null || true
            rm -f /tmp/tmp*.py 2>/dev/null || true
            exit 1
        fi

        cleanup_sglang

        echo "===== Done: $MODEL_NAME ====="
        EVALUATED+=("$MODEL_NAME")
        sleep 5
    done

    if [ "$NEW_FOUND" = false ]; then
        echo "No new checkpoints found, waiting..."
        sleep 1800
    fi
done