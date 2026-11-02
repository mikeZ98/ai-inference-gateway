"""Lexical verification of semantic cache hits.

Cosine similarity measures topical overlap, not meaning. With the feature-hashing
embedder, two long prompts that differ only in "30" vs "60 days", "shall pay" vs
"shall not pay" or "z+1" vs "z-1" score above 0.99, so one prompt would be served
the other's cached answer. Every hit that clears the similarity threshold is
therefore re-checked here, deterministically, before it is served.

A hit survives only if both prompts carry, in the same order:

- the same numbers: digit literals (thousands separators and trailing decimal
  zeros normalized, currency symbol kept) and spelled-out number words;
- the same math operators next to a number, a one-letter variable or a bracket;
- the same negation cues, each paired with the word it governs ("not pay");

and neither prompt negates a word of the other with a polarity prefix
("lawful" / "unlawful"). Any difference forces a cache miss: the guard gives up
some hit rate so that a cached answer is never served for a different question.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final


class LexicalMismatch(StrEnum):
    """Why a semantically similar cached prompt was refused."""

    NUMBERS = "numbers"
    OPERATORS = "operators"
    NEGATION = "negation"
    POLARITY_PREFIX = "polarity_prefix"


_CHAR_MAP: Final = str.maketrans(
    {
        "\N{MINUS SIGN}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": " ",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{MULTIPLICATION SIGN}": "*",
        "\N{DIVISION SIGN}": "/",
    }
)
_CONTRACTIONS: Final = (
    (re.compile(r"\bcan't\b|\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"),
)
_NUMBER_RE: Final = re.compile(r"([$€£¥]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_WORD_RE: Final = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")
_OPERATOR_RE: Final = re.compile(r"[+\-*/^=<>≤≥%]")

_NUMBER_WORDS: Final = frozenset(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "half",
        "twice",
        "double",
        "triple",
    ]
)
_NEGATIONS: Final = frozenset(
    [
        "not",
        "no",
        "never",
        "none",
        "nor",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
        "without",
        "unless",
        "except",
    ]
)
# Skipped when finding the word a negation governs: "not the loan" governs "loan".
_FILLERS: Final = frozenset(
    [
        "a",
        "an",
        "the",
        "to",
        "be",
        "been",
        "any",
        "all",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "his",
        "her",
        "our",
        "your",
        "my",
    ]
)
_NEGATIVE_PREFIXES: Final = ("non-", "non", "un", "in", "im", "il", "ir", "dis")
_MIN_STEM: Final = 3

type _NumberToken = tuple[str, Decimal | str]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).lower()
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _numbers(text: str) -> tuple[_NumberToken, ...]:
    found: list[tuple[int, _NumberToken]] = [
        (m.start(), (m.group(1), Decimal(m.group(2).replace(",", ""))))
        for m in _NUMBER_RE.finditer(text)
    ]
    found += [
        (m.start(), ("", m.group())) for m in _WORD_RE.finditer(text) if m.group() in _NUMBER_WORDS
    ]
    return tuple(token for _, token in sorted(found, key=lambda item: item[0]))


def _is_operand_edge(text: str, index: int, step: int) -> bool:
    """Whether the operand next to an operator (``step`` = -1 left, +1 right) is mathy."""
    i = index + step
    while 0 <= i < len(text) and text[i] == " ":
        i += step
    if not 0 <= i < len(text):
        return False
    char = text[i]
    if char.isdigit() or char in "()[]":
        return True
    beyond = i + step
    # A lone letter ("z" in "z+1") is a variable; a letter inside a word is prose.
    return char.isalpha() and not (0 <= beyond < len(text) and text[beyond].isalpha())


def _operators(text: str) -> tuple[str, ...]:
    return tuple(
        m.group()
        for m in _OPERATOR_RE.finditer(text)
        if _is_operand_edge(text, m.start(), -1) or _is_operand_edge(text, m.start(), +1)
    )


def _negations(words: list[str]) -> tuple[tuple[str, str], ...]:
    """Each negation cue with the first non-filler word after it ("" at the end)."""
    pairs: list[tuple[str, str]] = []
    for i, word in enumerate(words):
        if word in _NEGATIONS:
            following = (words[j] for j in range(i + 1, len(words)))
            governed = next((w for w in following if w not in _FILLERS), "")
            pairs.append((word, governed))
    return tuple(pairs)


def _negates(words: Counter[str], others: Counter[str]) -> bool:
    """Whether a word in ``words`` is a word in ``others`` behind a negative prefix."""
    return any(
        word.startswith(prefix)
        and len(word) - len(prefix) >= _MIN_STEM
        and word[len(prefix) :] in others
        for word in words
        for prefix in _NEGATIVE_PREFIXES
    )


@dataclass(frozen=True, slots=True)
class _Signature:
    numbers: tuple[_NumberToken, ...]
    operators: tuple[str, ...]
    negations: tuple[tuple[str, str], ...]
    words: Counter[str]

    @classmethod
    def of(cls, prompt: str) -> "_Signature":
        text = _normalize(prompt)
        words = _WORD_RE.findall(text)
        return cls(_numbers(text), _operators(text), _negations(words), Counter(words))


class LexicalCacheVerifier:
    """``CacheHitVerifier`` refusing hits whose numbers, operators or negations differ.

    Pure, deterministic and linear in prompt length; holds no state.
    """

    def mismatch(self, prompt: str, cached_prompt: str) -> LexicalMismatch | None:
        """Return the first critical difference between the prompts, or ``None``."""
        if prompt == cached_prompt:
            return None
        new, cached = _Signature.of(prompt), _Signature.of(cached_prompt)
        if new.numbers != cached.numbers:
            return LexicalMismatch.NUMBERS
        if new.operators != cached.operators:
            return LexicalMismatch.OPERATORS
        if new.negations != cached.negations:
            return LexicalMismatch.NEGATION
        only_new, only_cached = new.words - cached.words, cached.words - new.words
        if _negates(only_new, only_cached) or _negates(only_cached, only_new):
            return LexicalMismatch.POLARITY_PREFIX
        return None
