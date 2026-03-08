#!/bin/bash
OUTPUT_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/GROP/IF_colocate"
LOG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/rl/IF/logs"
MODEL_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/qwen3-1.7b-sft-by-tulu3-subsets"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
BETA=0.01
WANDB_NAME="if-grpo-colocate"

# open_instruct 모듈 경로 추가
export PYTHONPATH="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/rl:$PYTHONPATH"
export WANDB_PROJECT="OpenRLFT"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Colocate 모드: vLLM이 trainer 내부에서 실행되므로 별도 서버 불필요
# GPU 8장 전부를 학습+생성에 활용
accelerate launch --num_processes=8 /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/rl/IF/grpo_train.py \
    --model_name $MODEL_DIR \
    --vllm_mode colocate \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --beta $BETA \
    --wandb_name $WANDB_NAME \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee -a $LOG_FILE

echo "Training completed at $(date)" | tee -a $LOG_FILE
echo "All done at $(date)" | tee -a $LOG_FILE