#!/bin/bash

for step in $(seq 500 500 8500)
do
  echo "Summarizing results for checkpoint-${step}"
  results_dir=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/sft-qwen3-1.7b-tulu3-sft-mixture/checkpoint-${step}/eval-results
  ns summarize_results \
    $results_dir \
    --save_metrics_path $results_dir/../metrics.json
done
# Convert JSON to CSV for model comparison
# CSV will be saved in a common location for comparing multiple models
# comparison_csv=$(dirname $(dirname $results_dir))/model_comparison.csv
python3 /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/scripts/json_to_csv.py --batch /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/sft-qwen3-1.7b-tulu3-sft-mixture /data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/sft-qwen3-1.7b-tulu3-sft-mixture/combined_metrics.csv



# results_dir=/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/lmalign/eval/sft/checkpoint-2000/eval-results
# rm -rf $results_dir/summary
# ns summarize_results \
#   $results_dir \
#   --save_metrics_path $results_dir/../metrics.json