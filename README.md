# LMAlign

LLM Alignment 연구를 위한 워크스페이스. SFT 학습, RL 학습, 멀티 벤치마크 평가 파이프라인을 포함한다.

## Project Structure

```
lmalign/
├── sft-scripts/                 # SFT 학습 스크립트 (TRL + accelerate)
│   ├── train.py                 # 기본 SFT trainer
│   ├── train_packing.py         # Packing 지원 SFT trainer (flash_attention_2)
│   └── run_sft.sh               # 학습 실행 스크립트
├── eval/                        # 평가 파이프라인
│   ├── scripts/
│   │   ├── eval.sh              # 기본 벤치마크 평가 (ifeval, gsm8k, human-eval, mbpp)
│   │   ├── eval_arena-hard.sh   # Arena-Hard 평가
│   │   ├── mt_beach_eval.sh     # MT-Bench 평가 (GPT-4o judge)
│   │   ├── run_and_eval.sh      # 체크포인트 자동 감지 + 연속 평가
│   │   ├── run_and_eval_mtbench_sglang.sh
│   │   ├── vllm.sh              # vLLM 서버 (8 GPU, TP=8)
│   │   ├── sglang.sh            # SGLang 서버 (DP=4)
│   │   ├── save_result.sh       # 결과 요약 + CSV 변환
│   │   └── json_to_csv.py       # metrics.json → CSV 유틸
│   └── results/                 # 평가 결과
├── rl/                          # RL 학습 (GRPO 등)
├── data/                        # 학습 데이터셋
│   ├── tulu3-sft-mixture/
│   └── Dolci-Instruct-SFT/
├── chat_temp/                   # Chat template (Jinja)
│   ├── qwen3.jinja
│   └── qwen3_nonthinking.jinja
├── NeMo-Skills/                 # NVIDIA NeMo-Skills 프레임워크
├── nemo-skills-harness/         # NeMo-Skills 커스텀 래퍼
├── vllm.sh                      # vLLM 서버 런처
├── sglang.sh                    # SGLang 서버 런처
├── basic_eval.py                # NeMo Evaluator API 평가 예시
└── GUIDE.md                     # SFT & Eval 상세 가이드
```

## Prerequisites

- Python 3.12+
- CUDA (A100 GPU x 8)
- 주요 패키지: `trl`, `transformers`, `datasets`, `accelerate`, `peft`, `vllm`, `sglang`

## Quick Start

### 1. 환경 설정

```bash
# NeMo-Skills 평가 환경
cd nemo-skills-harness
conda create -n nemo-skills-harness python=3.13
conda activate nemo-skills-harness
bash setup.sh
```

### 2. 데이터 준비

```bash
# HuggingFace에서 데이터셋 다운로드
python data/down.py

# 평가용 벤치마크 데이터 준비
export NEMO_SKILLS_DATA_DIR=$(pwd)/nemo-skills-harness/datasets
ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR ifeval gsm8k human-eval mbpp
```

### 3. SFT 학습

`sft-scripts/run_sft.sh` 내 경로를 수정한 후 실행:

```bash
bash sft-scripts/run_sft.sh
```

핵심 설정:
- 8 GPU 분산 학습 (`accelerate launch --num_processes=8`)
- TRL `SFTTrainer` 기반
- `--assistant_only_loss`: assistant 응답에만 loss 적용
- `--bf16 --gradient_checkpointing`: 메모리 최적화
- Cosine LR scheduler, warmup 100 steps

### 4. 평가

```bash
# 1) 추론 서버 실행
bash eval/scripts/vllm.sh /path/to/checkpoint

# 2) 벤치마크 평가
bash eval/scripts/eval.sh /path/to/checkpoint 12738 eval/results/checkpoint-name

# 3) 결과 요약
ns summarize_results eval/results/checkpoint-name/eval-results \
    --save_metrics_path eval/results/checkpoint-name/metrics.json

# 4) CSV 변환 (체크포인트 비교용)
python eval/scripts/json_to_csv.py --batch eval/results/ combined_metrics.csv
```

학습과 동시에 자동 평가를 돌리려면:
```bash
bash eval/scripts/run_and_eval.sh
```

## Supported Benchmarks

| Category | Benchmarks |
|----------|-----------|
| Math | GSM8K, MATH, AIME24/25, MMLU |
| Code | Human-Eval, MBPP, LiveCodeBench, SWE-Bench |
| Instruction Following | IFEval |
| Chat Quality | Arena-Hard, MT-Bench |
| Science | GPQA, SciCode |

NeMo-Skills 프레임워크를 통해 80개 이상의 벤치마크를 지원한다.

## Key References

- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [NVIDIA NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)
- [GRPO + vLLM Online Training](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)

## Detailed Guide

SFT 학습 인자, 평가 스크립트 상세 사용법, 전체 워크플로우는 [GUIDE.md](GUIDE.md) 참고.
