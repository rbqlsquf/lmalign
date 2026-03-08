import json

from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, concatenate_datasets
import argparse

from verify import check_ifeval_old, check_ifeval_multi


# ============================================================
# Dataset 전처리
# ============================================================

SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["messages"][0]["content"]},
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/GROP/IF")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="OpenRLFT")
    parser.add_argument("--wandb_name", type=str, default="if-grpo")
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--use_vllm", type=bool, default=True)
    parser.add_argument("--vllm_mode", type=str, default="server")
    parser.add_argument("--vllm_model_impl", type=str, default="vllm")
    parser.add_argument("--beta", type=float, default=0.01)

    args = parser.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    per_device_train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    max_completion_length = args.max_completion_length
    num_generations = args.num_generations
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    push_to_hub = args.push_to_hub
    save_strategy = args.save_strategy
    save_steps = args.save_steps
    use_vllm = args.use_vllm
    vllm_mode = args.vllm_mode
    vllm_model_impl = args.vllm_model_impl
    logging_steps = args.logging_steps
    beta = args.beta
    # --- 데이터셋 로드 ---
    ds1_id = "allenai/RLVR-IFeval"
    ds2_id = "allenai/IF_multi_constraints_upto5"

    ds1_train = load_dataset(ds1_id, split="train")
    ds2_train = load_dataset(ds2_id, split="train")

    ds1_train = ds1_train.map(make_conversation)
    ds2_train = ds2_train.map(make_conversation)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{%- if tools %}", "{%- if false %}"
    )
    # 공통 컬럼만 남기고 합치기
    common_cols = ["prompt", "ground_truth"]
    ds1_train = ds1_train.select_columns(common_cols)
    ds2_train = ds2_train.select_columns(common_cols)
    train_dataset = concatenate_datasets([ds1_train, ds2_train]).shuffle(seed=42)
        # 극단적으로 긴 프롬프트 제거
    def filter_long_prompts(example):
        content = example["prompt"][-1]["content"]  # user message
        return len(tokenizer.encode(content, add_special_tokens=False)) <= 4096

    train_dataset = train_dataset.filter(filter_long_prompts, num_proc=1)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,

        # Data preprocessing
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,

        # Reporting and saving
        report_to=["wandb"],
        run_name=wandb_name,

        push_to_hub=push_to_hub,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        # vLLM
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_model_impl=vllm_model_impl,
        vllm_server_base_url="http://localhost:8000",
        chat_template_kwargs={"enable_thinking": False},

        beta=beta
        )

    # 두 데이터셋 합쳤으므로 통합 reward 사용
    # ground_truth 형식으로 자동 분기: JSON이면 RLVR, Python literal이면 multi
    def unified_reward(completions, **kwargs):
        """두 데이터셋의 ground_truth 형식에 따라 자동 분기."""
        ground_truths = kwargs["ground_truth"]
        completion_contents = [c[0]["content"] for c in completions]
        rewards = []
        for content, gt_str in zip(completion_contents, ground_truths):
            try:
                # RLVR-IFeval: JSON dict with "func_name"
                gt = json.loads(gt_str)
                if isinstance(gt, dict) and "func_name" in gt:
                    rewards.append(check_ifeval_old(content, gt_str) * 10.0)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # IF_multi_constraints: Python literal list (instruction_id 체계)
            try:
                rewards.append(check_ifeval_multi(content, gt_str) * 10.0)
            except Exception:
                rewards.append(0.0)

        return rewards

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[unified_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if push_to_hub:
        trainer.push_to_hub(dataset_name=f"{ds1_id}+{ds2_id}", repo_id=output_dir)
