#!/bin/bash
OUTPUT_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/GROP/IF"
LOG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/rl/IF/logs"
MODEL_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/qwen3-1.7b-sft-by-tulu3-subsets"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4


export WANDB_PROJECT="OpenRLFT"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Ctrl+C 시 vLLM 서버도 함께 종료
cleanup() {
    # cleanup 진입 후 signal trap 해제 → Ctrl+C 두 번째부터는 즉시 종료
    trap - SIGINT SIGTERM EXIT

    echo "Cleaning up..."
    if [ -n "$VLLM_PID" ]; then
        kill -9 $VLLM_PID 2>/dev/null || true
        pkill -9 -P $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        VLLM_PID=""
    fi
    # vLLM 엔진 코어 잔여 프로세스 정리
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 2
    echo "Cleanup done."
}
trap cleanup SIGINT SIGTERM EXIT

# 1. vLLM 서버를 백그라운드로 먼저 실행 (GPU 1장 할당)
CUDA_VISIBLE_DEVICES=7 trl vllm-serve \
    --model $MODEL_DIR \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 &
VLLM_PID=$!

echo "vLLM server started at $(date)" | tee -a $LOG_FILE
echo "Waiting for vLLM server..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
done
echo "vLLM server ready."

# 2. GRPO 학습 실행 (나머지 GPU 7개)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --num_processes=7 /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/rl/IF/grpo_train.py \
    --model_name $MODEL_DIR \
    --vllm_mode server \
    --vllm_model_impl vllm \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    2>&1 | tee -a $LOG_FILE

echo "Training completed at $(date)" | tee -a $LOG_FILE

# 3. 정리
cleanup
echo "All done at $(date)" | tee -a $LOG_FILE