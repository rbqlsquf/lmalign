#!/bin/bash

MODEL_NAME="/data/ib-a100-cluster-a-pri-lmt_967/models/Qwen3-1.7B-Base"
DATASET_NAME="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/data/tulu3-sft-mixture"

OUTPUT_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/sft-qwen3-1.7b-tulu3-sft-mixture"
LOG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/sft/logs"
NUM_TRAIN_EPOCHS=1
WARMUP_STEPS=100
LOGGING_STEPS=10
SAVE_STEPS=500
MAX_STEPS=15000


PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5

LEARNING_RATE_SCHEDULER_TYPE="cosine"

MAX_LENGTH=4096
CHAT_TEMPLATE_PATH="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/chat_temp/qwen3-0_6.jinja"

# Create log directory
mkdir -p $LOG_DIR

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo "Starting training at $(date)" | tee -a $LOG_FILE

accelerate launch --num_processes=8 /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/sft/train.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --max_steps $MAX_STEPS \
    --bf16 \
    --lr_scheduler_type $LEARNING_RATE_SCHEDULER_TYPE \
    --gradient_checkpointing \
    --max_length $MAX_LENGTH \
    --assistant_only_loss \
    --chat_template_path $CHAT_TEMPLATE_PATH \
    2>&1 | tee -a $LOG_FILE

echo "Training finished at $(date)" | tee -a $LOG_FILE