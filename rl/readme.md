# RL Training — GRPO for Instruction Following

GRPO(Group Relative Policy Optimization)를 활용하여 SFT 모델의 instruction-following 능력을 강화하는 파이프라인.

## Overview

SFT 체크포인트를 기반으로, IFEval 스타일의 제약조건(대소문자 변환, 단어 수 제한, JSON 포맷 등)을 만족하는지 자동 검증하여 리워드를 부여하고 정책을 최적화한다.

```
SFT 모델 → vLLM 서버 (온라인 생성) → 54개 제약조건 리워드 함수 → GRPO 학습
```

## Prerequisites

```bash
# TRL은 PyPI 릴리즈에 해당 기능이 없으므로 main 브랜치에서 직접 설치해야 한다
pip install git+https://github.com/huggingface/trl.git@main
pip install transformers datasets accelerate vllm wandb
```

> TRL main 브랜치(1.0.0.dev0)가 필요하다. PyPI의 stable 릴리즈에는 `GRPOTrainer`의 `vllm_mode="server"` 및 tool calling 관련 기능이 아직 포함되어 있지 않아 설치 시 에러가 발생한다.

Google Research IFEval 검증 코드가 `/opt/benchmarks/google-research`에 설치되어 있어야 한다.

## Quick Start

```bash
bash rl/IF/run_grop_if.sh
```

실행 전 `run_grop_if.sh` 내부의 경로를 수정:

```bash
OUTPUT_DIR="<체크포인트 저장 경로>"
MODEL_DIR="<SFT 체크포인트 경로>"
```

## Architecture

### GPU 할당

| GPU | 역할 |
|-----|------|
| GPU 7 | vLLM 서버 (`trl vllm-serve`, memory utilization 0.9) |
| GPU 0–6 | GRPO 학습 (`accelerate launch --num_processes=7`) |

### 학습 흐름

1. **vLLM 서버 실행** — `trl vllm-serve`로 SFT 모델 서빙 (GPU 7, port 8000)
2. **서버 헬스체크** — `/health` 엔드포인트가 응답할 때까지 대기
3. **GRPO 학습** — 7 GPU로 분산 학습. vLLM 서버에서 온라인으로 응답 생성 → 리워드 함수로 평가 → 정책 업데이트
4. **자동 정리** — Ctrl+C 또는 학습 종료 시 vLLM 프로세스 자동 종료

## Datasets

두 개의 IFEval 데이터셋을 합쳐서 사용한다:

| 데이터셋 | 설명 | 제약조건 |
|---------|------|---------|
| [allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) | 단일 제약조건, `func_name` 체계 | 1개/샘플 |
| [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) | 다중 제약조건, `instruction_id` 체계 | 1–5개/샘플 |

두 데이터셋은 `ground_truth` 형식(JSON dict vs Python literal list)으로 자동 분기되어 통합 리워드 함수에서 처리된다.

## Reward Function

### 54개 제약조건 체커

Google Research IFEval 원본 25개 + AllenAI 확장 29개 = **총 54개** 제약조건을 지원한다.

**Google IFEval 원본 (25개):**

| 카테고리 | 제약조건 |
|---------|---------|
| Change Case | `english_lowercase`, `english_capital`, `capital_word_frequency` |
| Keywords | `existence`, `frequency`, `letter_frequency`, `forbidden_words` |
| Length | `number_words`, `number_sentences`, `number_paragraphs`, `nth_paragraph_first_word` |
| Format | `json_format`, `title`, `multiple_sections`, `number_highlighted_sections`, `number_bullet_lists`, `constrained_response` |
| Content | `number_placeholders`, `postscript` |
| Punctuation | `no_comma` |
| Start/End | `end_checker`, `quotation` |
| Combination | `two_responses`, `repeat_prompt` |

**AllenAI 확장 (29개, `extended_instructions.py`):**

| 카테고리 | 제약조건 |
|---------|---------|
| Copy/Repeat | `copy`, `copying_simple`, `copying_multiple`, `repeat_phrase` |
| Word Count | `count_increment_word`, `count_unique`, `counting_composition` |
| Format/Structure | 기타 22개 확장 체커 |

### 리워드 계산

- **단일 제약조건** (RLVR-IFeval): 만족 시 `10.0`, 불만족 시 `0.0`
- **다중 제약조건** (IF_multi_constraints): `(만족 비율) × 10.0` (proportional)

#### ×10 스케일링이 결과에 영향을 주지 않는 이유

리워드 값에 `× 10.0`이 곱해져 있지만, GRPO의 advantage 계산 과정에서 **group-wise normalization**이 적용되므로 절대적인 스케일은 무의미하다.

```python
# TRL GRPOTrainer 내부 (grpo_trainer.py)
advantages = rewards - mean_grouped_rewards          # 그룹 평균 차감
advantages = advantages / (std_rewards + 1e-4)       # 표준편차로 정규화
```

같은 프롬프트에서 생성된 `num_generations`개의 응답들 간 상대적 차이만 학습에 반영된다. `× 10.0` 이든 `× 1.0` 이든 정규화 후 advantage는 동일하다. 스케일링은 단지 wandb 로그에서 리워드 값을 직관적으로 읽기 위한 용도이다.

#### EOS 토큰 패널티를 추가하지 못한 이유

`max_completion_length`에 도달하여 truncation된 응답에 패널티를 주면 모델이 EOS를 적절히 생성하도록 유도할 수 있을 것으로 기대했으나, 다음의 이유로 적용하지 않았다.

**1. GRPO는 sequence-level reward만 지원한다**

GRPO의 리워드 함수는 시퀀스 전체에 대한 스칼라 값 하나만 반환한다. PPO처럼 value head를 통한 per-token reward shaping이 불가능하므로, "EOS를 빨리 생성하라"는 세밀한 시그널을 줄 수 없다.

```python
# reward function은 시퀀스당 스칼라 1개만 반환
def unified_reward(completions, **kwargs) -> list[float]:
    ...
    return rewards  # [scalar, scalar, ...]
```

**2. Reward hacking 위험**

Sequence-level에서 truncation 여부에 대해 단순 패널티를 부여하면, 모델이 instruction-following 품질과 무관하게 **짧은 응답만 생성**하여 항상 EOS를 달성하는 방향으로 학습될 수 있다. 제약조건 만족도보다 응답 길이를 줄이는 것이 패널티를 피하는 더 쉬운 전략이 되기 때문이다.

**3. Trainer 내부에서 이미 EOS 후 마스킹을 처리한다**

TRL `GRPOTrainer`는 내부적으로 EOS 이후 토큰을 자동 마스킹하고, truncation 비율(`completions/clipped_ratio`)을 wandb에 로깅한다. 따라서 EOS 관련 처리는 trainer 레벨에서 이미 수행되고 있으며, 리워드 함수에서 중복 처리할 필요가 없다.

```python
# TRL GRPOTrainer 내부 — EOS 이후 자동 마스킹
is_eos = completion_ids == self.eos_token_id
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
```

> 참고: 리워드 함수에서 `kwargs["completion_ids"]`로 토큰 ID에 접근하는 것 자체는 가능하지만, 위의 이유들로 인해 실효성이 낮다.

## Hyperparameters

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model_name` | `Qwen/Qwen2-0.5B-Instruct` | 모델 경로 |
| `--learning_rate` | `1e-5` | 학습률 |
| `--num_generations` | `8` | 프롬프트당 생성 횟수 |
| `--beta` | `0.01` | KL penalty 계수 |
| `--max_completion_length` | `2048` | 최대 생성 길이 |
| `--per_device_train_batch_size` | `4` | GPU당 배치 크기 |
| `--gradient_accumulation_steps` | `4` | Gradient 축적 |
| `--save_steps` | `250` | 체크포인트 저장 간격 |
| `--num_train_epochs` | `1` | 에폭 수 |
| `--wandb_project` | `OpenRLFT` | W&B 프로젝트명 |

## File Structure

```
rl/
└── IF/
    ├── grpo_train.py              # GRPO trainer (메인 학습 스크립트)
    ├── extended_instructions.py   # 29개 확장 제약조건 체커
    ├── run_grop_if.sh             # vLLM 서버 + 학습 런처
    ├── download_model.py          # 모델 다운로드 유틸
    └── logs/                      # 학습 로그
```

## Evaluation

학습 중 체크포인트를 자동으로 평가하려면:

```bash
# SGLang 기반 연속 평가 (rl/IF 체크포인트용)
bash eval/scripts/run_and_eval_sglang.sh
```

수동 평가:

```bash
# 1) 추론 서버
bash eval/scripts/sglang.sh /path/to/rl-checkpoint

# 2) 벤치마크 평가
bash eval/scripts/eval.sh /path/to/rl-checkpoint 8000 eval/results/grpo-checkpoint-name
```

## References

- [GRPO + vLLM Online Training (HuggingFace Cookbook)](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [AllenAI Open-Instruct](https://github.com/allenai/open-instruct)
- [Google Research IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- [AllenAI RLVR-IFeval Dataset](https://huggingface.co/datasets/allenai/RLVR-IFeval)
- [AllenAI IF_multi_constraints Dataset](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)
