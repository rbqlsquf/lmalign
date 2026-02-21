# ifeval 평가

# export NEMO_SKILLS_CONFIG_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/configs"
# export NEMO_SKILLS_DATA_DIR="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/datasets"

# # Set Python executable - ensure python command is available
# export PYTHON=python3
# export PYTHON_EXECUTABLE=python3
# # Add ~/bin to PATH if python symlink exists there
# if [ -f "$HOME/bin/python" ]; then
#     export PATH="$HOME/bin:$PATH"
# fi

# if [ ! -d $NEMO_SKILLS_DATA_DIR ]; then
#     mkdir -p $NEMO_SKILLS_DATA_DIR
# fi

ns eval \
	--server_type=vllm \
	--model=/data/ib-a100-cluster-a-pri-lmt_967/models/Qwen3-0.6B-Base \
	--server_address=http://localhost:8000/v1 \
	--benchmarks=mmlu \
	--output_dir=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/Qwen3-0.6B-Base \
	++inference.temperature=0.6 \
	++inference.top_p=0.95 \
	++inference.tokens_to_generate=1024