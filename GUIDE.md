# LMAlign - SFT & Evaluation Guide

## Project Structure

```
lmalign/
├── sft-scripts/           # SFT 학습 스크립트
│   ├── train.py           # 기본 SFT 학습
│   ├── train_packing.py   # Packing 지원 SFT 학습 (flash_attention_2)
│   └── run_sft.sh         # 학습 실행 쉘 스크립트
├── eval/
│   ├── scripts/
│   │   ├── eval.sh              # 벤치마크 평가 (ifeval, gsm8k, human-eval, mbpp)
│   │   ├── eval_arena-hard.sh   # Arena-Hard 평가
│   │   ├── run_and_eval.sh      # 체크포인트 자동 감지 + 연속 평가 (vLLM)
│   │   ├── run_and_eval_mtbench_sglang.sh  # MT-Bench 연속 평가 (SGLang)
│   │   ├── mt_beach_eval.sh     # MT-Bench 평가 (GPT-4o judge)
│   │   ├── vllm.sh              # 평가용 vLLM 서버 (8 GPU, TP=8)
│   │   ├── sglang.sh            # 평가용 SGLang 서버 (DP=4)
│   │   ├── save_result.sh       # 결과 요약 + CSV 변환
│   │   └── json_to_csv.py       # metrics.json → CSV 변환 유틸
│   └── results/                 # 평가 결과 저장 디렉토리
├── data/                        # 학습 데이터셋
│   ├── tulu3-sft-mixture/
│   └── Dolci-Instruct-SFT/
├── chat_temp/                   # Chat template (Jinja)
│   ├── qwen3.jinja
│   └── qwen3_nonthinking.jinja
├── NeMo-Skills/                 # NVIDIA NeMo-Skills 프레임워크
├── nemo-skills-harness/         # NeMo-Skills 커스텀 래퍼
├── vllm.sh                      # 루트 vLLM 서버 런처
└── sglang.sh                    # 루트 SGLang 서버 런처
```

---

## 1. SFT 학습

### 1.1 학습 스크립트

두 가지 학습 스크립트가 존재한다:

| 스크립트 | 설명 |
|---------|------|
| `sft-scripts/train.py` | 기본 SFT 학습 |
| `sft-scripts/train_packing.py` | Packing 모드 시 `flash_attention_2` 자동 활성화 |

둘 다 TRL의 `SFTTrainer` 기반이며, `accelerate`로 멀티 GPU 분산 학습을 수행한다.

### 1.2 빠른 시작

```bash
bash sft-scripts/run_sft.sh
```

`run_sft.sh`의 기본 설정:

```bash
MODEL_NAME="/data/.../models/Qwen3-1.7B-Base"
DATASET_NAME="/data/.../data/tulu3-sft-mixture"
OUTPUT_DIR="/data/.../checkpoints/lmalign/sft-qwen3-1.7b-tulu3-sft-mixture"

accelerate launch --num_processes=8 sft-scripts/train.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_steps 15000 \
    --bf16 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --max_length 4096 \
    --assistant_only_loss \
    --chat_template_path chat_temp/qwen3-0_6.jinja
```

### 1.3 주요 학습 인자

| 인자 | 기본값 | 설명 |
|-----|--------|------|
| `--model_name` | `Qwen/Qwen3-0.6B-Base` | 모델 경로 또는 HuggingFace ID |
| `--dataset_name` | `allenai/Dolci-Instruct-SFT` | 데이터셋 경로 (load_from_disk) |
| `--output_dir` | - | 체크포인트 저장 경로 |
| `--max_length` | 2048 | 최대 시퀀스 길이 |
| `--assistant_only_loss` | False | assistant 토큰에만 loss 적용 |
| `--chat_template_path` | `Qwen/Qwen3-0.6B` | Jinja chat template 경로 |
| `--packing` | False | Example packing 활성화 |
| `--bf16` | False | bfloat16 precision |
| `--gradient_checkpointing` | False | 메모리 절약을 위한 gradient checkpointing |
| `--learning_rate` | 2e-5 | 학습률 (LoRA 사용 시 1e-4 권장) |
| `--lr_scheduler_type` | cosine | LR 스케줄러 |
| `--warmup_steps` | 0 | Warmup 스텝 수 |
| `--save_steps` | 500 | 체크포인트 저장 간격 |
| `--eos_token` | None | EOS 토큰 (Qwen2.5: `<\|endoftext\|>`) |
| `--model_dtype` | None | 모델 로딩 dtype (`bfloat16`, `float16`, `float32`) |

### 1.4 커스텀 학습 예시

```bash
# 다른 모델/데이터셋으로 학습
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

### 1.5 데이터셋 형식

데이터셋은 `load_from_disk`으로 로드하며, HuggingFace `datasets` arrow 형식이어야 한다. 최대 1,000,000개 샘플까지 사용된다.

새 데이터셋 다운로드 예시 (`data/down.py` 참고):
```python
from datasets import load_dataset
dataset = load_dataset("allenai/tulu-3-sft-mixture")
dataset.save_to_disk("/path/to/save")
```

---

## 2. Evaluation

### 2.1 사전 준비: NeMo-Skills 환경

```bash
cd nemo-skills-harness
conda create -n nemo-skills-harness python=3.13
conda activate nemo-skills-harness
bash setup.sh
```

환경 변수 설정:
```bash
export NEMO_SKILLS_CONFIG_DIR="$(pwd)/nemo-skills-harness/configs"
export NEMO_SKILLS_DATA_DIR="$(pwd)/nemo-skills-harness/datasets"
```

벤치마크 데이터 준비:
```bash
ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR ifeval
ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR gsm8k
ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR human-eval
ns prepare_data --cluster=local --data_dir $NEMO_SKILLS_DATA_DIR mbpp
```

### 2.2 추론 서버 실행

평가 전에 모델 서빙 서버를 먼저 띄워야 한다.

**vLLM (8 GPU, Tensor Parallel)**
```bash
# eval/scripts/vllm.sh - 8 GPU TP 방식
bash eval/scripts/vllm.sh /path/to/model
# → port 12738, TP=8, max_model_len=32768
```

**SGLang (4 GPU, Data Parallel)**
```bash
# eval/scripts/sglang.sh - 4 GPU DP 방식
bash eval/scripts/sglang.sh /path/to/model
# → port 8000, DP=4, context_length=32768
```

**루트의 vllm.sh / sglang.sh**은 특정 모델이 하드코딩되어 있으므로, 평가 시에는 `eval/scripts/` 하위의 스크립트를 사용할 것.

### 2.3 단일 평가 실행

```bash
bash eval/scripts/eval.sh <model_path> <port> <output_dir>
```

기본 벤치마크: `ifeval`, `gsm8k`, `human-eval`, `mbpp`

추론 설정: `temperature=0.6`, `top_p=0.95`, `tokens_to_generate=4096`

**예시:**
```bash
bash eval/scripts/eval.sh \
    /data/.../checkpoints/lmalign/sft-qwen3-1.7b/checkpoint-1000 \
    12738 \
    eval/results/checkpoint-1000
```

**Arena-Hard 평가:**
```bash
bash eval/scripts/eval_arena-hard.sh <model_path> <port> <output_dir>
# max_concurrent_requests=5
```

**MT-Bench 평가:**
```bash
bash eval/scripts/mt_beach_eval.sh <model_name> <output_dir>
# GPT-4o를 judge로 사용, port 8000 고정
# OPENAI_API_KEY 환경변수 필요
```

### 2.4 연속 자동 평가 (학습 중 체크포인트 자동 감지)

학습과 동시에 새 체크포인트를 자동으로 감지하여 평가를 돌리는 스크립트이다.

```bash
bash eval/scripts/run_and_eval.sh
```

동작 방식:
1. `MODEL_DIR`에서 `checkpoint-*` 디렉토리를 탐색
2. 미평가 체크포인트 발견 시 vLLM 서버를 자동으로 실행
3. 서버 health check 후 평가 수행 (`ifeval`, `gsm8k`, `human-eval`, `mbpp`)
4. 평가 완료 후 vLLM 서버 종료 및 다음 체크포인트로 진행
5. 새 체크포인트가 없으면 30분 대기 후 재탐색

사용 전 스크립트 내부의 경로를 수정해야 한다:
```bash
MODEL_DIR=<체크포인트 상위 디렉토리>
RESULTS_BASE=<결과 저장 디렉토리>
EVAL_SCRIPT=<eval.sh 경로>
VLLM_SCRIPT=<vllm.sh 경로>
```

### 2.5 결과 확인

**결과 요약 (ns summarize_results)**

평가가 끝나면 `eval-results/` 폴더에 벤치마크별 raw 결과가 저장된다. 이를 요약하려면:

```bash
ns summarize_results \
    eval/results/checkpoint-1000/eval-results \
    --save_metrics_path eval/results/checkpoint-1000/metrics.json
```

**여러 체크포인트 일괄 요약 (save_result.sh 참고)**

```bash
for step in $(seq 500 500 8500); do
    results_dir=eval/results/checkpoint-${step}/eval-results
    ns summarize_results $results_dir --save_metrics_path $results_dir/../metrics.json
done
```

**metrics.json 예시:**
```json
{
  "gsm8k": {"pass@1": {"symbolic_correct": 69.07, "no_answer": 7.96}},
  "human-eval": {"pass@1": {"passing_base_tests": 55.49, "passing_plus_tests": 49.39}},
  "ifeval": {"pass@1": {"average_score": 48.88, "instruction_strict_accuracy": 52.76}},
  "mbpp": {"pass@1": {"passing_base_tests": 49.47, "passing_plus_tests": 43.12}}
}
```

**CSV 변환 (체크포인트 간 비교용)**

```bash
# 단일 파일
python eval/scripts/json_to_csv.py metrics.json output.csv [model_name]

# 여러 체크포인트 일괄 변환
python eval/scripts/json_to_csv.py --batch eval/results/ combined_metrics.csv
```

### 2.6 지원 벤치마크 요약

| 벤치마크 | 주요 메트릭 | 스크립트 |
|---------|-----------|---------|
| GSM8K | `symbolic_correct` | eval.sh |
| Human-Eval | `passing_base_tests`, `passing_plus_tests` | eval.sh |
| MBPP | `passing_base_tests`, `passing_plus_tests` | eval.sh |
| IFEval | `average_score`, `instruction_strict_accuracy` | eval.sh |
| Arena-Hard | judge 기반 평가 | eval_arena-hard.sh |
| MT-Bench | GPT-4o judge | mt_beach_eval.sh |
| MMLU | `symbolic_correct` | ifeval.sh (별도) |

NeMo-Skills가 지원하는 전체 벤치마크: `aime24`, `aime25`, `math-odyssey`, `gpqa`, `livecodebench`, `swe-bench`, `ruler` 등 80개 이상.

---

## 3. 전체 워크플로우 요약

```
1. 데이터 준비
   └─ data/down.py 또는 HuggingFace datasets로 다운로드 → save_to_disk

2. SFT 학습
   └─ run_sft.sh 경로 수정 후 실행
      └─ accelerate launch --num_processes=8 train.py ...

3. 추론 서버 실행
   └─ bash eval/scripts/vllm.sh /path/to/checkpoint

4. 평가 실행
   ├─ 단일: bash eval/scripts/eval.sh <model> <port> <output>
   └─ 자동: bash eval/scripts/run_and_eval.sh (학습과 병렬로)

5. 결과 확인
   ├─ ns summarize_results → metrics.json
   └─ json_to_csv.py --batch → combined_metrics.csv
```
