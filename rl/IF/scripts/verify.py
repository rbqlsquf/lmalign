"""
경량 Verifier 모듈.

ground_truth_utils.py를 통째로 import하면 beaker, litellm 등 불필요한
의존성이 연쇄 로드되므로, 필요한 하위 모듈만 직접 import하여
IFEval / Math / Code 검증 로직을 제공한다.

의존성 체인:
  open_instruct.if_functions        → json, re, langdetect          ✅
  open_instruct.IFEvalG.*           → (자체 완결)                    ✅
  open_instruct.math_utils          → re, sympy 등                  ✅
  requests                          → (code 검증용 HTTP 호출)        ✅
"""

import ast
import json
import re
import logging
import requests

# ── IFEval 검증 모듈 ──
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.IFEvalG import instructions_registry

# ── Math 검증 모듈 ──
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT

logger = logging.getLogger(__name__)


# ============================================================
# 공통 유틸
# ============================================================

def remove_thinking_section(prediction: str) -> str:
    """<think>...</think> 및 <answer> 태그 제거."""
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


# ============================================================
# IFEval 검증
# ============================================================

def check_ifeval_old(prediction: str, label) -> float:
    """RLVR-IFeval: func_name 기반 단일 constraint 검증.

    (ground_truth_utils.IFEvalVerifierOld 로직)
    """
    answer = remove_thinking_section(prediction)
    constraint = json.loads(label) if isinstance(label, str) else dict(label)
    if "func_name" not in constraint:
        return 0.0
    func_name = constraint.pop("func_name")
    if func_name not in IF_FUNCTIONS_MAP:
        return 0.0
    func = IF_FUNCTIONS_MAP[func_name]
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    if not non_none_args:
        return float(func(answer))
    return float(func(answer, **non_none_args))


def check_ifeval_multi(prediction: str, label: str) -> float:
    """IF_multi_constraints: instruction_id + kwargs 기반 다중 constraint 검증.

    만족한 constraint 비율을 반환한다.
    (ground_truth_utils.IFEvalVerifier 로직)
    """
    answer = remove_thinking_section(prediction)
    if not prediction.strip() or not answer:
        return 0.0
    constraint_dict = ast.literal_eval(label)
    constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]
    rewards = []
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        instruction_cls = INSTRUCTION_DICT[instruction_key]
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)
        if instruction_instance.check_following(answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return sum(rewards) / len(rewards) if rewards else 0.0


# ============================================================
# Math 검증
# ============================================================

def check_gsm8k(prediction: str, label: str) -> float:
    """GSM8K: 마지막 숫자 추출 후 정답 비교.

    (ground_truth_utils.GSM8KVerifier 로직)
    """
    response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    extracted = numbers[-1] if numbers else response
    return float(str(extracted).lower() == str(label).lower())


def check_math(prediction: str, label: str) -> float:
    r"""Math: \\boxed{}, Minerva, LaTeX 등 다양한 추출 후 정답 비교.

    (ground_truth_utils.MathVerifier 로직)
    """
    raw_answer = prediction
    all_answers = []

    # \\boxed{} 추출
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)

    # Minerva format 추출
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    # LaTeX ($...$) 추출
    if not all_answers:
        dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
        if len(dollars) > 1:
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)

    # Fallback
    if not all_answers:
        all_answers.append(normalize_final_answer(prediction))
        all_answers.append(prediction)

    for answer in all_answers:
        if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
            return 1.0
    return 0.0


def check_strict_math(prediction: str, label: str) -> float:
    """Strict Math: Minerva format만 사용하는 엄격 검증.

    (ground_truth_utils.StrictMathVerifier 로직)
    """
    all_answers = []
    minerva_answer = normalize_final_answer(get_unnormalized_answer(prediction))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    if not all_answers:
        all_answers.append(normalize_final_answer(prediction))
    for answer in all_answers:
        if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
            return 1.0
    return 0.0


# ============================================================
# Code 검증
# ============================================================

def check_code(
    prediction: str,
    label,
    api_url: str,
    max_execution_time: float = 10.0,
    pass_rate_threshold: float = 0.0,
) -> float:
    """Code: 외부 API를 통한 테스트 케이스 실행 검증.

    (ground_truth_utils.CodeVerifier 로직 — 동기 버전)

    Args:
        prediction: 모델 출력 (코드 블록 포함)
        label: 테스트 케이스 리스트
        api_url: 코드 실행 API URL (e.g. "http://host:port/test_program")
        max_execution_time: 최대 실행 시간 (초)
        pass_rate_threshold: 이 비율 미만이면 0점
    """
    # 마지막 코드 블록 추출
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, prediction, re.DOTALL)
    python_code = matches[-1].strip() if matches else prediction

    payload = {
        "program": python_code,
        "tests": label,
        "max_execution_time": max_execution_time,
    }
    try:
        http_timeout = max(30, min(300, max_execution_time * 10))
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=http_timeout,
        )
        response.raise_for_status()
        result = response.json()
        passes = result["results"]
        pass_rate = sum(passes) / len(passes) if passes else 0.0
        return 0.0 if pass_rate < pass_rate_threshold else pass_rate
    except Exception as e:
        logger.warning(f"Error verifying code sample: {e}")
        return 0.0
