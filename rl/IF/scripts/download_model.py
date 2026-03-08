from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="junseojang/Qwen3-1.7B-IFEval-RLVR",
    local_dir="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/checkpoints/lmalign/jjs",
)
