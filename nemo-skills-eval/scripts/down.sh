#!/bin/bash

benchmark_dir=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/nemo-skills-harness/datasets
benchmarks=gsm8k,humaneval
for bench in $benchmarks; do
    ns prepare_data --data_dir $benchmark_dir $bench
    cp /usr/local/lib/python3.12/dist-packages/nemo_skills/dataset/${bench}/*.jsonl $benchmark_dir/$bench/
done