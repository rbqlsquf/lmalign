# LMAlign

LLM Alignment 연구를 위한 End-to-End 파이프라인. **SFT(Supervised Fine-Tuning)**, **RL(Reinforcement Learning, GRPO)**, **멀티 벤치마크 평가**를 포함한다.

## Features

- **SFT Training** — TRL `SFTTrainer` + `accelerate` 기반 멀티 GPU 분산 학습. Packing, assistant-only loss, 커스텀 chat template 지원.
- **RL Training (GRPO)** — Group Relative Policy Optimization으로 instruction-following 능력 강화. vLLM 온라인 생성 + 54개 제약조건 리워드 함수.
- **Evaluation Pipeline** — NeMo-Skills 기반 80개 이상의 벤치마크 자동 평가. 학습 중 체크포인트 자동 감지 + 연속 평가 지원.
- **Inference Serving** — vLLM (Tensor Parallel) / SGLang (Data Parallel) 서버 런처 내장.

## Project Structure

```
lmalign/
├── sft-scripts/                 # SFT 학습
│   ├── train.py                 # 기본 SFT trainer
│   ├── train_packing.py         # Packing 지원 SFT trainer (flash_attention_2)
│   └── run_sft.sh               # 학습 실행 스크립트
├── rl/                          # RL 학습
│   └── IF/                      # Instruction-Following GRPO
│       ├── grpo_train.py        # GRPO trainer (dual dataset, vLLM online)
│       ├── extended_instructions.py  # 29개 커스텀 제약조건 체커
│       ├── run_grop_if.sh       # vLLM 서버 + GRPO 학습 런처
│       └── download_model.py    # 모델 다운로드 유틸
├── eval/                        # 평가 파이프라인
│   ├── scripts/
│   │   ├── eval.sh              # 기본 벤치마크 평가 (ifeval, gsm8k, human-eval, mbpp)
│   │   ├── eval_arena-hard.sh   # Arena-Hard 평가
│   │   ├── mt_beach_eval.sh     # MT-Bench 평가 (GPT-4o judge)
│   │   ├── run_and_eval.sh      # 체크포인트 자동 감지 + 연속 평가 (vLLM)
│   │   ├── run_and_eval_sglang.sh  # 체크포인트 자동 감지 + 연속 평가 (SGLang)
│   │   ├── vllm.sh              # vLLM 서버 (8 GPU, TP=8)
│   │   ├── sglang.sh            # SGLang 서버 (DP=4)
│   │   ├── save_result.sh       # 결과 요약 + CSV 변환
│   │   └── json_to_csv.py       # metrics.json → CSV 유틸
│   └── results/                 # 평가 결과
├── data/                        # 학습 데이터셋
│   ├── tulu3-sft-mixture/
│   └── down.py                  # 데이터셋 다운로드
├── chat_temp/                   # Chat template (Jinja2)
│   ├── qwen3.jinja              # Qwen3 (tool call 포함)
│   └── qwen3_nonthinking.jinja  # Qwen3 non-thinking 모드
├── NeMo-Skills/                 # NVIDIA NeMo-Skills 프레임워크
├── nemo-skills-harness/         # NeMo-Skills 커스텀 래퍼
├── vllm.sh                      # 루트 vLLM 서버 런처
├── sglang.sh                    # 루트 SGLang 서버 런처
├── basic_eval.py                # NeMo Evaluator API 예시
└── GUIDE.md                     # SFT & Eval 상세 가이드
```

## Prerequisites

- Python 3.12+
- CUDA (A100 GPU × 8 권장)
- 주요 패키지:

```bash
pip install trl==1.0.0.dev0 transformers datasets accelerate peft
pip install vllm sglang
```

## Quick Start

### 1. 환경 설정

```bash
# NeMo-Skills 평가 환경
cd nemo-skills-harness
conda create -n nemo-skills-harness python=3.13
conda activate nemo-skills-harness
bash setup.sh

# NLTK 리소스 (IFEval에 필요)
python3 -c "import nltk; nltk.download('punkt_tab')"
```

### 2. 데이터 준비

```bash
# HuggingFace에서 학습 데이터셋 다운로드
python data/down.py

# 평가용 벤치마크 데이터 준비
export NEMO_SKILLS_DATA_DIR=$(pwd)/nemo-skills-harness/datasets
ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR ifeval gsm8k human-eval mbpp
```

### 3. SFT 학습

`sft-scripts/run_sft.sh` 내 모델/데이터 경로를 수정한 후 실행:

```bash
bash sft-scripts/run_sft.sh
```

핵심 설정:
- **8 GPU 분산 학습** (`accelerate launch --num_processes=8`)
- TRL `SFTTrainer` 기반, `--assistant_only_loss`로 assistant 응답에만 loss 적용
- `--bf16 --gradient_checkpointing`으로 메모리 최적화
- Cosine LR scheduler, warmup 100 steps

<details>
<summary>커스텀 학습 예시</summary>

```bash
accelerate launch --num_processes=8 sft-scripts/train_packing.py \
    --model_name /path/to/your/model \
    --dataset_name /path/to/your/dataset \
    --output_dir /path/to/output \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_length 4096 \
    --max_steps 10000 \
    --save_steps 1000 \
    --warmup_steps 200 \
    --bf16 \
    --gradient_checkpointing \
    --packing \
    --assistant_only_loss \
    --chat_template_path chat_temp/qwen3.jinja
```

</details>

### 4. RL 학습 (GRPO)

SFT 학습 후 GRPO(Group Relative Policy Optimization)로 instruction-following 능력을 강화한다.

```bash
bash rl/IF/run_grop_if.sh
```

동작 방식:
- **GPU 7**: vLLM 서버 (온라인 생성용)
- **GPU 0–6**: GRPO 학습
- 2개 데이터셋 사용: `allenai/RLVR-IFeval` + `allenai/IF_multi_constraints_upto5`
- **54개 제약조건 리워드 함수** (Google IFEval 25개 + AllenAI 확장 29개)

주요 하이퍼파라미터:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `learning_rate` | 1e-5 | 학습률 |
| `num_generations` | 8 | 생성 횟수 (per prompt) |
| `beta` | 0.01 | KL penalty |
| `max_completion_length` | 2048 | 최대 생성 길이 |
| `per_device_train_batch_size` | 4 | 배치 크기 |
| `gradient_accumulation_steps` | 4 | Gradient 축적 |

### 5. 평가

```bash
# 1) 추론 서버 실행
bash eval/scripts/vllm.sh /path/to/checkpoint    # vLLM (8 GPU, TP=8)
# 또는
bash eval/scripts/sglang.sh /path/to/checkpoint  # SGLang (4 GPU, DP=4)

# 2) 벤치마크 평가
bash eval/scripts/eval.sh /path/to/checkpoint 12738 eval/results/checkpoint-name

# 3) 결과 요약
ns summarize_results eval/results/checkpoint-name/eval-results \
    --save_metrics_path eval/results/checkpoint-name/metrics.json

# 4) CSV 변환 (체크포인트 간 비교)
python eval/scripts/json_to_csv.py --batch eval/results/ combined_metrics.csv
```

#### 학습 중 자동 연속 평가

학습과 동시에 새 체크포인트를 자동 감지하여 평가를 실행한다:

```bash
# vLLM 기반
bash eval/scripts/run_and_eval.sh

# SGLang 기반
bash eval/scripts/run_and_eval_sglang.sh
```

동작: 체크포인트 디렉토리 탐색 → 서버 자동 실행 → 평가 → 서버 종료 → 다음 체크포인트. 새 체크포인트가 없으면 30분 대기 후 재탐색.

#### Arena-Hard / MT-Bench

```bash
# Arena-Hard
bash eval/scripts/eval_arena-hard.sh <model_path> <port> <output_dir>

# MT-Bench (GPT-4o judge, OPENAI_API_KEY 필요)
bash eval/scripts/mt_beach_eval.sh <model_name> <output_dir>
```

#### IFBench 평가 시 패키지 패치 (필수)

NeMo-Skills의 `ifbench.py`에 파일명 버그가 있다. IFBench `run_eval`은 결과 파일에 입력 파일명을 접두사로 붙이지만 (`output-eval_results_loose.jsonl`), nemo_skills는 접두사 없이 (`eval_results_loose.jsonl`) 찾는다. 아래 명령으로 패치:

```bash
sudo sed -i \
  's|output_dir / "eval_results_loose.jsonl"|output_dir / f"{jsonl_path.stem}-eval_results_loose.jsonl"|; s|output_dir / "eval_results_strict.jsonl"|output_dir / f"{jsonl_path.stem}-eval_results_strict.jsonl"|' \
  /usr/local/lib/python3.12/dist-packages/nemo_skills/evaluation/evaluator/ifbench.py
```

## Supported Benchmarks

| Category | Benchmarks | Script |
|----------|-----------|--------|
| Math | GSM8K, MATH, AIME24/25, MMLU | `eval.sh` |
| Code | Human-Eval, MBPP, LiveCodeBench, SWE-Bench | `eval.sh` |
| Instruction Following | IFEval | `eval.sh` |
| Chat Quality | Arena-Hard | `eval_arena-hard.sh` |
| Chat Quality | MT-Bench (GPT-4o judge) | `mt_beach_eval.sh` |
| Science | GPQA, SciCode | `eval.sh` (커스텀) |

NeMo-Skills 프레임워크를 통해 `aime24`, `aime25`, `math-odyssey`, `gpqa`, `livecodebench`, `swe-bench`, `ruler` 등 **80개 이상의 벤치마크**를 지원한다.

## Workflow Summary

```
1. 데이터 준비        data/down.py → save_to_disk
        │
2. SFT 학습          run_sft.sh → accelerate 8 GPU
        │
3. RL 학습 (GRPO)    run_grop_if.sh → vLLM + 7 GPU
        │
4. 평가              eval.sh / run_and_eval.sh (자동)
        │
5. 결과 분석         ns summarize_results → json_to_csv.py
```

## Key References

- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [GRPO + vLLM Online Training](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [NVIDIA NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)
- [AllenAI Open-Instruct](https://github.com/allenai/open-instruct)

## Detailed Guide

SFT 학습 인자, 평가 스크립트 상세 사용법, 전체 워크플로우는 [GUIDE.md](GUIDE.md) 참고.
