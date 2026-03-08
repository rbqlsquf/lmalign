from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import argparse

from verify import check_ifeval_multi


# ============================================================
# Reward 함수
# ============================================================

def ifeval_multi_reward(completions, **kwargs):
    """IF_multi_constraints 데이터셋용 reward (다중 constraint, instruction_id 체계).

    만족한 constraint 비율 × 10.0을 반환한다.
    """
    ground_truths = kwargs["ground_truth"]
    completion_contents = [c[0]["content"] for c in completions]
    rewards = []
    for content, gt_str in zip(completion_contents, ground_truths):
        try:
            rewards.append(check_ifeval_multi(content, gt_str) * 10.0)
        except Exception:
            rewards.append(0.0)
    return rewards


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
    ds_id = "allenai/IF_multi_constraints_upto5"
    full_dataset = load_dataset(ds_id, split="train")
    full_dataset = full_dataset.map(make_conversation)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{%- if tools %}", "{%- if false %}"
    )

    full_dataset = full_dataset.select_columns(["prompt", "ground_truth"])

    # 극단적으로 긴 프롬프트 제거
    def filter_long_prompts(example):
        content = example["prompt"][-1]["content"]  # user message
        return len(tokenizer.encode(content, add_special_tokens=False)) <= 4096

    full_dataset = full_dataset.filter(filter_long_prompts, num_proc=1)

    # train/eval split (eval 200샘플)
    split = full_dataset.train_test_split(test_size=200, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

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
        eval_strategy="steps",
        eval_steps=save_steps,
        # vLLM
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_model_impl=vllm_model_impl,
        vllm_server_base_url="http://localhost:8000",
        chat_template_kwargs={"enable_thinking": False},

        beta=beta
        )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[ifeval_multi_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if push_to_hub:
        trainer.push_to_hub(dataset_name=ds_id, repo_id=output_dir)
