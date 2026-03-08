PORT=$2
export NEMO_SKILLS_CONFIG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/configs"
export NEMO_SKILLS_DATA_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/datasets"

ns eval \
	--server_type=vllm \
	--model=$1 \
	--server_address=http://localhost:$PORT/v1 \
	--benchmarks=arena-hard \
	--output_dir=$3 \
	++inference.temperature=0.6 \
	++inference.top_p=0.95 \
	++inference.tokens_to_generate=4096 \
	++max_concurrent_requests=5