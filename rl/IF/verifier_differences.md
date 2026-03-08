# Old vs New Verifier 차이 분석

## 두 데이터셋 개요

| 데이터셋 | ground_truth 형식 | verifier |
|---------|------------------|----------|
| `allenai/RLVR-IFeval` | `{"func_name": "verify_keywords", ...}` | Old (`IF_FUNCTIONS_MAP`, `if_functions.py`) |
| `allenai/IF_multi_constraints_upto5` | `[{"instruction_id": [...], "kwargs": [...]}]` | New (`INSTRUCTION_DICT`, `IFEvalG/instructions.py`) |

두 데이터셋은 ground_truth 형식이 다르고, 일부 constraint에서 검증 로직 자체가 다르다. 대부분의 constraint는 포맷 변환만으로 통합 가능하지만, 아래 나열된 constraint들에서 차이가 존재한다.

---

## Constraint별 구현 차이

### 1. paragraph_count — 구분자가 상호 배타적

| | Old (`if_functions.py`) | New (`ParagraphChecker`) |
|---|---|---|
| **분리 방식** | `text.split("* * *")` (공백 포함 리터럴) | `re.split(r"\s?\*\*\*\s?", value)` (regex) |

Old는 정확히 `* * *` (별 사이 공백 포함) 문자열로 split하고, New는 regex `\s?\*\*\*\s?`로 연속된 `***` (공백 없음, 앞뒤 공백만 옵셔널) 기준으로 split한다. **양방향 불일치**:

| 모델 출력 | Old (`split("* * *")`) | New (`re.split(r"\s?\*\*\*\s?")`) |
|---|---|---|
| `* * *` (공백 포함) | ✅ | ❌ (별 사이 공백은 regex 매치 안 됨) |
| `***` (공백 없음) | ❌ | ✅ |

RLVR-IFeval 프롬프트는 `"the markdown divider: * * *"`, New 프롬프트는 `"the markdown divider: ***"`로 안내하므로, 모델이 각 프롬프트를 따르면 해당 verifier에서만 통과한다. **각 verifier에 맞는 데이터만 사용해야 한다.**

참고: New verifier에는 paragraph 관련 클래스가 3개 있다:

| 클래스 | instruction_id | 구분자 |
|--------|---------------|--------|
| `ParagraphChecker` | `length_constraints:number_paragraphs` | `***` (regex) |
| `ParagraphBasicChecker` | `paragraphs:paragraphs` | `***` (regex) |
| `ParagraphBasicChecker2` | `paragraphs:paragraphs2` | `\n\n` (빈 줄) |

`\n\n`을 쓰는 건 `ParagraphBasicChecker2`뿐이며, 이는 별도의 instruction_id로 구분된다.

### 2. keyword_frequency — 비교 연산 및 매칭 방식 차이

| | Old | New (`KeywordFrequencyChecker`) |
|---|---|---|
| **비교 연산** | 항상 `== N` (정확히 N번) | `< N` (`less than`) 또는 `>= N` (`at least`) |
| **매칭 방식** | `\b\w+\b`로 단어 분리 → exact word match | `re.findall(keyword, value, IGNORECASE)` → substring match |
| **파라미터** | `word`, `N` | `keyword`, `frequency`, `relation` |

**비교 연산 차이**: Old는 정확히 N번을 요구하지만, New는 부등식만 지원한다. RLVR-IFeval의 프롬프트가 "should appear N times"라고 명시하므로 의도는 exact match인데, New의 `"at least" N`으로 매핑하면 N번 이상이 모두 통과한다. 다만 실제로 모델이 keyword를 정확히 N번 넣는 경우가 대부분이라 false positive 빈도는 낮다.

**매칭 방식 차이**: Old는 단어 경계(`\b\w+\b`) 기준 exact word match이므로 "cat"을 찾을 때 "category"는 매치하지 않지만, New는 substring regex match이므로 "category" 안의 "cat"도 카운트한다. 단, RLVR-IFeval에서 사용되는 keyword가 대부분 독립적인 단어이므로 실질적 영향은 제한적이다.

### 3. letter_frequency — 비교 연산 차이

| | Old | New (`LetterFrequencyChecker`) |
|---|---|---|
| **비교 연산** | `text.count(letter) == N` (정확히 N번) | `< N` 또는 `>= N` (relation 기반) |
| **파라미터** | `letter`, `N` | `letter`, `let_frequency`, `let_relation` |

keyword_frequency와 동일한 구조의 차이. Old는 exact match, New는 부등식만 지원.

### 4. word_constraint — `around` quantifier 미지원

| | Old | New (`NumberOfWords`) |
|---|---|---|
| **지원 relation** | `at least`, `around`, `at most` | `less than`, `at least` |
| **`around` 처리** | ±10% 허용 | **ValueError 발생** |
| **`at most` 처리** | `<= N` | 미지원 (ValueError) |

RLVR-IFeval에 `"quantifier": "around"` 또는 `"at most"`인 데이터가 있으면 New verifier에서 에러가 발생한다.

### 5. sentence_constraint — 문장 분리 방식 및 quantifier 차이

| | Old | New (`NumberOfSentences`) |
|---|---|---|
| **문장 분리** | 정규식 `(?<=\.|\?)\s` | `instructions_util.count_sentences()` |
| **지원 relation** | `at least`, `around`, `at most` | `less than`, `at least` |
| **`around` 처리** | `abs(actual - N) <= 1` | 미지원 (ValueError) |

문장 분리 방식이 다르면 같은 텍스트라도 문장 수가 다르게 카운트될 수 있다. 또한 `around`, `at most` quantifier는 New에서 지원하지 않는다.

### 6. validate_choice — 비교 방향 차이

| | Old | New (`ConstrainedResponseChecker`) |
|---|---|---|
| **로직** | `any(text in option for option in options)` — 응답이 선택지 안에 포함되는지 | `any(option in value for option in self._constrained_responses)` — 선택지가 응답 안에 포함되는지 |

비교 방향이 반대다. Old는 "응답 ⊂ 선택지"를 확인하고, New는 "선택지 ⊂ 응답"을 확인한다. 짧은 응답에서는 결과가 다를 수 있다.

---

## 차이 없는 것들

파라미터 없는 단순 체커들은 동일하게 동작한다:

- `validate_lowercase` / `validate_uppercase`
- `validate_no_commas`
- `validate_quotation`
- `validate_json_format`
- `validate_title`
- `validate_two_responses`

---

## 요약

| 차이 유형 | 해당 constraint | 실질적 영향 |
|----------|----------------|------------|
| **exact match vs 부등식** | `keyword_frequency`, `letter_frequency` | false positive 가능 (N+1번도 통과) |
| **미지원 quantifier** | `word_constraint`, `sentence_constraint` | `around`, `at most` 시 ValueError 크래시 |
| **매칭 방식** | `keyword_frequency` | word match vs substring match 차이 |
| **비교 방향** | `validate_choice` | 짧은 응답에서 결과 차이 가능 |
| **문장 분리 엔진** | `sentence_constraint` | 문장 수 카운트 차이 가능 |
| **구분자 표기** | `paragraph_count` | `***` vs `* * *` 양방향 불일치, 프롬프트 따르면 한쪽만 통과 |

## 결론

대부분의 constraint는 포맷 변환만으로 통합 가능하며, 실질적으로 문제가 되는 건 소수다. 그러나 `around`/`at most` quantifier가 포함된 샘플은 New verifier에서 **크래시**가 발생하므로, RLVR-IFeval을 New verifier로 처리하려면 해당 샘플을 사전에 필터링하거나 매핑해야 한다. AllenAI 공식 문서(tulu3.md)에서도 "The new IFEvalVerifier class is not compatible with the old data format"이라고 명시하고 있으며, **New 데이터(`IF_multi_constraints_upto5`) + New verifier 조합을 권장**한다.

## 현재 선택

- `grpo_train.py` — `IF_multi_constraints_upto5`만 사용 (New verifier only) ← **권장**
- `grpo_train_old.py` — 두 데이터셋 혼용 (Old/New verifier 자동 분기)