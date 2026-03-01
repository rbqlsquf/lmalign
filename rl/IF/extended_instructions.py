"""AllenAI IF_multi_constraints_upto5 데이터셋용 확장 constraint checker.

Google Research IFEval의 Instruction 클래스 패턴을 따르며,
INSTRUCTION_DICT에 없는 29개 instruction_id를 구현.
"""
import re
import collections
from instruction_following_eval.instructions import Instruction
from instruction_following_eval import instructions_util


# ============================================================
# copy: 프롬프트 복사/반복 관련
# ============================================================

class CopyChecker(Instruction):
    """Copy the instruction verbatim."""

    def build_description(self, *, prompt_to_repeat=None, **kw):
        self._prompt_to_repeat = prompt_to_repeat or ""

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        return ["prompt_to_repeat"]

    def check_following(self, value):
        return self._prompt_to_repeat.strip().lower() in value.strip().lower()


class CopyingSimpleChecker(Instruction):
    """Repeat the request without change."""

    def build_description(self, *, prompt_to_repeat=None, **kw):
        self._prompt_to_repeat = prompt_to_repeat or ""

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        return ["prompt_to_repeat"]

    def check_following(self, value):
        return value.strip().lower().startswith(self._prompt_to_repeat.strip().lower())


class CopyingMultipleChecker(Instruction):
    """Repeat the request N times, separated by ******."""

    def build_description(self, *, prompt_to_repeat=None, N=None, **kw):
        self._prompt_to_repeat = prompt_to_repeat or ""
        self._n = N or 1

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat, "N": self._n}

    def get_instruction_args_keys(self):
        return ["prompt_to_repeat", "N"]

    def check_following(self, value):
        parts = value.split("******")
        valid = [p for p in parts if p.strip()]
        if len(valid) != self._n:
            return False
        prompt_lower = self._prompt_to_repeat.strip().lower()
        return all(prompt_lower in p.strip().lower() for p in valid)


class RepeatPhraseChecker(Instruction):
    """Repeat a phrase N times with slight transformations."""

    def build_description(self, *, phrase=None, small_n=None, **kw):
        self._phrase = phrase or ""
        self._n = small_n or 1

    def get_instruction_args(self):
        return {"phrase": self._phrase, "small_n": self._n}

    def get_instruction_args_keys(self):
        return ["phrase", "small_n"]

    def check_following(self, value):
        # 원래 phrase의 단어들 중 일부가 변형되어도 등장 횟수 체크
        phrase_words = self._phrase.lower().split()
        if not phrase_words:
            return False
        # 최소 phrase 핵심 단어(첫/마지막)가 N번 이상 등장
        key_word = phrase_words[0]
        count = len(re.findall(re.escape(key_word), value.lower()))
        return count >= self._n


# ============================================================
# count: 단어/문장 수 관련
# ============================================================

class CountIncrementWordChecker(Instruction):
    """keyword1 once, keyword2 twice."""

    def build_description(self, *, keyword1=None, keyword2=None, **kw):
        self._keyword1 = keyword1 or ""
        self._keyword2 = keyword2 or ""

    def get_instruction_args(self):
        return {"keyword1": self._keyword1, "keyword2": self._keyword2}

    def get_instruction_args_keys(self):
        return ["keyword1", "keyword2"]

    def check_following(self, value):
        v = value.lower()
        c1 = len(re.findall(r"\b" + re.escape(self._keyword1.lower()) + r"\b", v))
        c2 = len(re.findall(r"\b" + re.escape(self._keyword2.lower()) + r"\b", v))
        return c1 >= 1 and c2 >= 2


class CountUniqueChecker(Instruction):
    """Only unique words in response."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        words = re.findall(r"\w+", value.lower())
        return len(words) == len(set(words))


class CountingCompositionChecker(Instruction):
    """N paragraphs with M sentences each with K words."""

    def build_description(self, *, n_sent=None, n_words=None, **kw):
        self._n_sent = n_sent or 3
        self._n_words = n_words or 2

    def get_instruction_args(self):
        return {"n_sent": self._n_sent, "n_words": self._n_words}

    def get_instruction_args_keys(self):
        return ["n_sent", "n_words"]

    def check_following(self, value):
        paragraphs = re.split(r"\s?\*\s?\*\s?\*\s?", value)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if len(paragraphs) != 3:
            return False
        for para in paragraphs:
            sentences = instructions_util.split_into_sentences(para)
            sentences = [s for s in sentences if s.strip()]
            if len(sentences) != self._n_sent:
                return False
            for sent in sentences:
                words = re.findall(r"\w+", sent)
                if len(words) != self._n_words:
                    return False
        return True


class LowercaseCountingChecker(Instruction):
    """All lowercase words should appear at most N times."""

    def build_description(self, *, N=None, **kw):
        self._n = N or 3

    def get_instruction_args(self):
        return {"N": self._n}

    def get_instruction_args_keys(self):
        return ["N"]

    def check_following(self, value):
        words = re.findall(r"\b[a-z]+\b", value)
        counter = collections.Counter(words)
        return all(c <= self._n for c in counter.values())


# ============================================================
# detectable_format: 포맷 관련
# ============================================================

class SentenceHyphensChecker(Instruction):
    """All sentences connected using hyphens, no spaces."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return "-" in value and "- " not in value and " -" not in value


class SquareBracketsChecker(Instruction):
    """Every word enclosed in square brackets."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        bracketed = re.findall(r"\[[\w']+\]", value)
        words = re.findall(r"\w+", value)
        # 대부분의 단어가 bracket 안에 있으면 OK
        return len(bracketed) >= len(words) * 0.8 if words else False


class BigramWrappingChecker(Instruction):
    """Wrap word bigrams in <<>>."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        bigrams = re.findall(r"<<[^>]+>>", value)
        return len(bigrams) >= 2


# ============================================================
# first_word / last_word
# ============================================================

class FirstWordAnswerChecker(Instruction):
    """First word of response should be X."""

    def build_description(self, *, first_word=None, **kw):
        self._first_word = (first_word or "").lower()

    def get_instruction_args(self):
        return {"first_word": self._first_word}

    def get_instruction_args_keys(self):
        return ["first_word"]

    def check_following(self, value):
        words = value.strip().split()
        if not words:
            return False
        # 구두점 제거 후 비교
        first = re.sub(r"[^\w]", "", words[0]).lower()
        return first == self._first_word


class FirstWordSentChecker(Instruction):
    """First word of each sentence should be X."""

    def build_description(self, *, first_word=None, **kw):
        self._first_word = (first_word or "").lower()

    def get_instruction_args(self):
        return {"first_word": self._first_word}

    def get_instruction_args_keys(self):
        return ["first_word"]

    def check_following(self, value):
        sentences = instructions_util.split_into_sentences(value)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return False
        for sent in sentences:
            words = sent.strip().split()
            if not words:
                return False
            first = re.sub(r"[^\w]", "", words[0]).lower()
            if first != self._first_word:
                return False
        return True


class LastWordAnswerChecker(Instruction):
    """Last word of response should be X."""

    def build_description(self, *, last_word=None, **kw):
        self._last_word = (last_word or "").lower()

    def get_instruction_args(self):
        return {"last_word": self._last_word}

    def get_instruction_args_keys(self):
        return ["last_word"]

    def check_following(self, value):
        words = value.strip().split()
        if not words:
            return False
        last = re.sub(r"[^\w]", "", words[-1]).lower()
        return last == self._last_word


class LastWordSentChecker(Instruction):
    """Last word of each sentence (before punctuation) should be X."""

    def build_description(self, *, last_word=None, **kw):
        self._last_word = (last_word or "").lower()

    def get_instruction_args(self):
        return {"last_word": self._last_word}

    def get_instruction_args_keys(self):
        return ["last_word"]

    def check_following(self, value):
        sentences = instructions_util.split_into_sentences(value)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return False
        for sent in sentences:
            words = sent.strip().split()
            if not words:
                return False
            last = re.sub(r"[^\w]", "", words[-1]).lower()
            if last != self._last_word:
                return False
        return True


# ============================================================
# keywords: 키워드 관련 확장
# ============================================================

class ExcludeWordHarderChecker(Instruction):
    """Do not include keyword in response."""

    def build_description(self, *, keyword=None, **kw):
        self._keyword = keyword or ""

    def get_instruction_args(self):
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        return ["keyword"]

    def check_following(self, value):
        return self._keyword.lower() not in value.lower()


class KeywordSpecificPositionChecker(Instruction):
    """Include keyword in n-th sentence as m-th word."""

    def build_description(self, *, keyword=None, n=None, m=None, **kw):
        self._keyword = (keyword or "").lower()
        self._n = n or 1
        self._m = m or 1

    def get_instruction_args(self):
        return {"keyword": self._keyword, "n": self._n, "m": self._m}

    def get_instruction_args_keys(self):
        return ["keyword", "n", "m"]

    def check_following(self, value):
        sentences = instructions_util.split_into_sentences(value)
        sentences = [s for s in sentences if s.strip()]
        if self._n > len(sentences):
            return False
        words = sentences[self._n - 1].strip().split()
        if self._m > len(words):
            return False
        word = re.sub(r"[^\w]", "", words[self._m - 1]).lower()
        return word == self._keyword


class NoAdjacentConsecutiveChecker(Instruction):
    """No two adjacent words start with consecutive alphabet letters."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        words = re.findall(r"[a-zA-Z]+", value)
        for i in range(len(words) - 1):
            a = words[i][0].lower()
            b = words[i + 1][0].lower()
            if abs(ord(a) - ord(b)) == 1:
                return False
        return True


class PalindromeChecker(Instruction):
    """Include a palindrome in response."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        words = re.findall(r"[a-zA-Z]+", value.lower())
        for word in words:
            if len(word) >= 3 and word == word[::-1]:
                return True
        return False


class StartEndChecker(Instruction):
    """Start and end response with the same word."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        words = re.findall(r"\w+", value.strip())
        if len(words) < 2:
            return False
        return words[0].lower() == words[-1].lower()


class WordCountDifferentNumbersChecker(Instruction):
    """Word should appear exactly N times."""

    def build_description(self, *, keyword=None, frequency=None, relation=None, **kw):
        self._keyword = (keyword or "").lower()
        self._frequency = frequency or 1
        self._relation = relation or "less than"

    def get_instruction_args(self):
        return {"keyword": self._keyword, "frequency": self._frequency, "relation": self._relation}

    def get_instruction_args_keys(self):
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        count = len(re.findall(r"\b" + re.escape(self._keyword) + r"\b", value.lower()))
        if self._relation == "less than":
            return count < self._frequency
        else:
            return count >= self._frequency


class WordOnceChecker(Instruction):
    """Include keyword at least once."""

    def build_description(self, *, keyword=None, **kw):
        self._keyword = (keyword or "").lower()

    def get_instruction_args(self):
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        return ["keyword"]

    def check_following(self, value):
        return self._keyword in value.lower()


# ============================================================
# letters: 글자 수 관련
# ============================================================

class LetterCountingChecker(Instruction):
    """Answer with at least/most N letters."""

    def build_description(self, *, N=None, relation=None, **kw):
        self._n = N or 1
        self._relation = relation or "at least"

    def get_instruction_args(self):
        return {"N": self._n, "relation": self._relation}

    def get_instruction_args_keys(self):
        return ["N", "relation"]

    def check_following(self, value):
        letter_count = len(re.findall(r"[a-zA-Z]", value))
        if self._relation == "at least":
            return letter_count >= self._n
        else:
            return letter_count < self._n


class LetterCounting2Checker(Instruction):
    """Letter X appears at least/less than N times."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None, **kw):
        self._letter = (letter or "a").lower()
        self._frequency = let_frequency or 1
        self._relation = let_relation or "at least"

    def get_instruction_args(self):
        return {"letter": self._letter, "let_frequency": self._frequency, "let_relation": self._relation}

    def get_instruction_args_keys(self):
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        count = value.lower().count(self._letter)
        if self._relation == "less than":
            return count < self._frequency
        else:
            return count >= self._frequency


# ============================================================
# paragraphs: 문단 관련 (Google IFEval과 다른 변형)
# ============================================================

class ParagraphsChecker(Instruction):
    """2 paragraphs separated by ***."""

    def build_description(self, **kw):
        self._num_paragraphs = 2

    def get_instruction_args(self):
        return {}

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        valid = [p for p in paragraphs if p.strip()]
        return len(valid) == self._num_paragraphs


class Paragraphs2Checker(Instruction):
    """2 paragraphs separated by double newlines."""

    def build_description(self, **kw):
        self._num_paragraphs = 2

    def get_instruction_args(self):
        return {}

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        paragraphs = re.split(r"\n\n", value)
        valid = [p for p in paragraphs if p.strip()]
        return len(valid) == self._num_paragraphs


# ============================================================
# punctuation: 구두점 관련
# ============================================================

class PunctuationDotChecker(Instruction):
    """No dots allowed."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return "." not in value


class PunctuationExclamationChecker(Instruction):
    """No exclamation marks allowed."""

    def build_description(self, **kw):
        pass

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return "!" not in value


# ============================================================
# new: 기타
# ============================================================

class CopySpanIdxChecker(Instruction):
    """Copy the span between character index n_start and n_end."""

    def build_description(self, *, n_start=None, n_end=None, prompt_to_repeat=None, **kw):
        self._n_start = n_start or 0
        self._n_end = n_end or 0
        self._prompt = prompt_to_repeat or ""

    def get_instruction_args(self):
        return {"n_start": self._n_start, "n_end": self._n_end, "prompt_to_repeat": self._prompt}

    def get_instruction_args_keys(self):
        return ["n_start", "n_end", "prompt_to_repeat"]

    def check_following(self, value):
        span = self._prompt[self._n_start:self._n_end + 1]
        return span.lower() in value.lower()


# ============================================================
# Registry: instruction_id → checker class
# ============================================================

EXTENDED_INSTRUCTION_DICT = {
    "copy:copy": CopyChecker,
    "copy:copying_simple": CopyingSimpleChecker,
    "copy:copying_multiple": CopyingMultipleChecker,
    "copy:repeat_phrase": RepeatPhraseChecker,
    "count:count_increment_word": CountIncrementWordChecker,
    "count:count_unique": CountUniqueChecker,
    "count:counting_composition": CountingCompositionChecker,
    "count:lowercase_counting": LowercaseCountingChecker,
    "detectable_format:sentence_hyphens": SentenceHyphensChecker,
    "detectable_format:square_brackets": SquareBracketsChecker,
    "detectable_format:bigram_wrapping": BigramWrappingChecker,
    "first_word:first_word_answer": FirstWordAnswerChecker,
    "first_word:first_word_sent": FirstWordSentChecker,
    "last_word:last_word_answer": LastWordAnswerChecker,
    "last_word:last_word_sent": LastWordSentChecker,
    "keywords:exclude_word_harder": ExcludeWordHarderChecker,
    "keywords:keyword_specific_position": KeywordSpecificPositionChecker,
    "keywords:no_adjacent_consecutive": NoAdjacentConsecutiveChecker,
    "keywords:palindrome": PalindromeChecker,
    "keywords:start_end": StartEndChecker,
    "keywords:word_count_different_numbers": WordCountDifferentNumbersChecker,
    "keywords:word_once": WordOnceChecker,
    "letters:letter_counting": LetterCountingChecker,
    "letters:letter_counting2": LetterCounting2Checker,
    "paragraphs:paragraphs": ParagraphsChecker,
    "paragraphs:paragraphs2": Paragraphs2Checker,
    "punctuation:punctuation_dot": PunctuationDotChecker,
    "punctuation:punctuation_exclamation": PunctuationExclamationChecker,
    "new:copy_span_idx": CopySpanIdxChecker,
}
