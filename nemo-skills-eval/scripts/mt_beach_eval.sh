
export MY_API_KEY="your_api_key_here"
# nemo-evaluator-launcher run \
#     --config /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/rl/sglang_eval_qwen.yaml \
#     -o execution.output_dir=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/rl/results

# nemo-evaluator run_eval \
#   --eval_type ifeval \
#   --model_id Qwen/Qwen3-0.6B \
#   --model_type chat \
#   --model_url http://localhost:6000/v1/chat/completions \
#   --api_key_name MY_API_KEY \
#   --output_dir /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/rl/results/nemo_eval/ifeval \
#   --overrides "config.params.request_timeout=300,config.params.max_new_tokens=1024,target.api_endpoint.adapter_config.mode=client"


# nemo-evaluator run_eval \
#   --eval_type gsm8k \
#   --model_id Qwen/Qwen3-0.6B \
#   --model_type completions \
#   --model_url http://localhost:6000/v1/completions \
#   --api_key_name MY_API_KEY \
#   --output_dir /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/rl/results/nemo_eval/gsm8k \
#   --overrides "config.params.request_timeout=300,config.params.max_new_tokens=1024,target.api_endpoint.adapter_config.mode=client"

# nemo-evaluator run_eval \
#   --eval_type humaneval \
#   --model_id Qwen/Qwen3-0.6B \
#   --model_type chat \
#   --model_url http://localhost:6000/v1/chat/completions \
#   --api_key_name MY_API_KEY \
#   --output_dir /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/rl/results/nemo_eval/humaneval \
#   --overrides "config.params.request_timeout=300,config.params.max_new_tokens=1024,target.api_endpoint.adapter_config.mode=client"

export OPENAI_MODEL_URL=https://api.openai.com/v1

OVERRIDES="
config.params.extra.judge.model_id=gpt-4o,
config.params.extra.judge.api_key=OPENAI_API_KEY,
config.params.extra.judge.url=https://api.openai.com/v1,
config.params.request_timeout=300,
config.params.max_new_tokens=1024
"

nemo-evaluator run_eval \
  --eval_type mtbench \
  --model_id $1 \
  --model_type chat \
  --model_url http://localhost:8000/v1/chat/completions \
  --api_key_name MY_API_KEY \
  --output_dir $2 \
  --overrides "$OVERRIDES"