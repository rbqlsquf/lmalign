# RL Training

강화학습(GRPO)을 활용한 모델 정렬 파이프라인.

## Prerequisites

### 1. 패키지 설치

```bash
pip install -r rl/requirements.txt
```

> **왜 TRL main 브랜치인가?** Qwen3의 chat template에는 tool calling 관련 분기(`{%- if tools %}`)가 포함되어 있다. PyPI stable 버전의 TRL은 이 template을 처리할 때 tool calling 경로를 잘못 타면서 에러가 발생한다. 이 버그가 수정된 main 브랜치를 설치해야 한다.

### 2. AllenAI Open-Instruct 검증 모듈

`verify.py`가 AllenAI [open-instruct](https://github.com/allenai/open-instruct)의 검증 모듈에 의존한다. `.gitignore`에 포함되어 있으므로 직접 가져와야 한다.

```bash
git clone https://github.com/allenai/open-instruct.git /tmp/open-instruct
cp -r /tmp/open-instruct/open_instruct rl/open_instruct
rm -rf /tmp/open-instruct
```

### 3. NLTK 데이터

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## Tasks

### Instruction Following (IF)

IFEval 스타일의 54개 제약조건을 자동 검증하여 리워드를 부여하고 정책을 최적화한다.

자세한 내용은 [rl/IF/readme.md](IF/readme.md) 참조.

```bash
bash rl/IF/run_grop_if.sh
```

## File Structure

```
rl/
├── readme.md
├── requirements.txt
├── IF/                            # Instruction Following GRPO
│   ├── readme.md                  # IF 학습 상세 문서
│   ├── grpo_train.py              # GRPO trainer — IF_multi_constraints only
│   ├── grpo_train_old.py          # GRPO trainer — 두 데이터셋 혼용 (Old+New verifier)
│   ├── verify.py                  # 경량 verifier (IFEval/Math/Code 검증)
│   ├── run_grop_if.sh             # vLLM 서버 + 학습 런처
│   └── download_model.py          # 모델 다운로드 유틸
└── open_instruct/                 # AllenAI Open-Instruct 검증 모듈
    ├── if_functions.py            # IFEval 원본 25개 함수
    ├── IFEvalG/                   # 확장 IFEval 검증 클래스
    │   ├── instructions.py        # 54개 제약조건 체커 클래스
    │   └── instructions_registry.py
    ├── math_utils.py              # Math 검증 유틸
    └── ground_truth_utils.py      # 통합 verifier 래퍼
```

## References

- [GRPO + vLLM Online Training (HuggingFace Cookbook)](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [AllenAI Open-Instruct](https://github.com/allenai/open-instruct)
- [Google Research IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
