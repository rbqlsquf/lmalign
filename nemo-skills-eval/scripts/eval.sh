PORT=$2
export NEMO_SKILLS_CONFIG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/configs"
export NEMO_SKILLS_DATA_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/datasets"
bash /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-eval/scripts/openai_key.sh
export OPENAI_MODEL_URL=https://api.openai.com/v1

ns eval \
	--server_type=vllm \
	--model=$1 \
	--server_address=http://localhost:$PORT/v1 \
	--benchmarks=gsm8k,hendrycks_math,human-eval,ifeval,ifbench,mbpp \
	--output_dir=$3 \
	++inference.temperature=0.6 \
	++inference.top_p=0.95 \
	++inference.tokens_to_generate=2048