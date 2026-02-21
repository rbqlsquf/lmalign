"""
Supervised Fine-Tuning (SFT) training script for Qwen3 base models.
Based on TRL SFTTrainer documentation.
"""

import argparse
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk
from peft import LoraConfig
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training script for Qwen3 base models")

    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base",
                        help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="allenai/Dolci-Instruct-SFT",
                        help="Dataset name or path")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--output_dir", type=str,
                        default="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/sft",
                        help="Output directory for checkpoints")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (use 1e-4 for LoRA)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint frequency")
    # Precision and optimization
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer type")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")

    # SFT specific arguments
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length (None for VLMs)")
    parser.add_argument("--packing", action="store_true",
                        help="Enable example packing")
    parser.add_argument("--assistant_only_loss", action="store_true",
                        help="Train only on assistant messages")
    parser.add_argument("--chat_template_path", type=str, default="Qwen/Qwen3-0.6B",
                        help="Chat template path for instruction tuning")
    parser.add_argument("--eos_token", type=str, default=None,
                        help="EOS token (e.g., '<|endoftext|>' for Qwen2.5)")

    # Evaluation
    parser.add_argument("--eval_dataset_name", type=str, default=None,
                        help="Evaluation dataset name")
    parser.add_argument("--eval_dataset_split", type=str, default="validation",
                        help="Evaluation dataset split")

    # Model loading
    parser.add_argument("--model_dtype", type=str, default=None,
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype for loading")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    train_dataset = load_from_disk(args.dataset_name)
    train_dataset = train_dataset[args.dataset_split]
    train_dataset = train_dataset.select(range(min(1000000, len(train_dataset))))
    # Load eval dataset if provided
    eval_dataset = None
    if args.eval_dataset_name:
        print(f"Loading eval dataset: {args.eval_dataset_name}")
        eval_dataset = load_from_disk(args.eval_dataset_name, split=args.eval_dataset_split)

    # Prepare model_init_kwargs
    model_init_kwargs = {}
    if args.model_dtype:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model_init_kwargs["dtype"] = dtype_map[args.model_dtype]

    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        max_length=args.max_length if args.max_length > 0 else None,
        packing=args.packing,
        assistant_only_loss=args.assistant_only_loss,
        chat_template_path=args.chat_template_path if args.chat_template_path else None,
        eos_token=args.eos_token,
        model_init_kwargs=model_init_kwargs if model_init_kwargs else None,
        report_to="tensorboard",
        dataset_num_proc=32
    )

    # Initialize trainer
    print(f"Initializing SFTTrainer with model: {args.model_name}")
    trainer = SFTTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()

    # Optional: Push to hub
    # trainer.push_to_hub()


if __name__ == "__main__":
    main()

