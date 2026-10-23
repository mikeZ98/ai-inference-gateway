"""Rule-based prompt complexity assessment.

Cheap, deterministic signals (length, reasoning vocabulary, code, dialogue
structure, enumerated steps, math) are summed into a score in ``[0, 1]``.
Scores at or above the policy threshold are ``HIGH`` complexity (Claude tiers);
the rest are ``LOW`` (Tier 1). All patterns are linear-time (no nested
quantifiers) to stay ReDoS-safe on large prompts.
"""

import re
from typing import Final, Self

from pydantic import Field, model_validator

from ai_inference_gateway.schemas import ComplexityAssessment, ComplexityTier, StrictModel

CHARS_PER_TOKEN: Final = 4

WEIGHT_LONG_PROMPT: Final = 0.2
WEIGHT_VERY_LONG_PROMPT: Final = 0.4
WEIGHT_REASONING: Final = 0.25
WEIGHT_REASONING_DENSE: Final = 0.5
WEIGHT_CODE: Final = 0.3
WEIGHT_MULTI_TURN: Final = 0.2
WEIGHT_MULTI_STEP: Final = 0.15
WEIGHT_MATH: Final = 0.15

DENSE_REASONING_MIN_TERMS: Final = 3
MULTI_TURN_MIN_TURNS: Final = 2
MULTI_STEP_MIN_ITEMS: Final = 3
CODE_MIN_SYNTAX_LINES: Final = 2

_REASONING_RE: Final = re.compile(
    r"\b(prove|proof|derive|derivation|analy[sz]e|analysis|step[- ]by[- ]step|reasoning"
    r"|trade-?offs?|compare|contrast|evaluate|architect(?:ure)?|design|optimi[sz]e"
    r"|debug|refactor|explain why|justify|algorithm|complexity)\b",
    re.IGNORECASE,
)
_CODE_FENCE_RE: Final = re.compile(r"```|~~~")
_CODE_SYNTAX_RE: Final = re.compile(
    r"^[ \t]*(?:def |class |import |from [\w.]+ import |function |const |let |fn "
    r"|#include|public |private |SELECT |INSERT |CREATE TABLE )",
    re.MULTILINE,
)
_TURN_RE: Final = re.compile(
    r"^[ \t]*(?:user|assistant|human|system|ai)[ \t]*:", re.IGNORECASE | re.MULTILINE
)
_LIST_ITEM_RE: Final = re.compile(r"^[ \t]*(?:\d{1,3}[.)]|[-*•])[ \t]+\S", re.MULTILINE)
_MATH_RE: Final = re.compile(r"\$[^$\n]+\$|\\(?:frac|sum|int|lim|sqrt)\b|[∑∫∂√≤≥≠]")


class ComplexityPolicy(StrictModel):
    """Tunable thresholds for :class:`HeuristicComplexityAssessor`."""

    high_threshold: float = Field(default=0.45, gt=0.0, le=1.0)
    long_prompt_tokens: int = Field(default=512, ge=1)
    very_long_prompt_tokens: int = Field(default=2048, ge=1)

    @model_validator(mode="after")
    def _ordered_lengths(self) -> Self:
        if self.very_long_prompt_tokens <= self.long_prompt_tokens:
            raise ValueError("very_long_prompt_tokens must exceed long_prompt_tokens")
        return self


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer (max of chars/4 and word count)."""
    return max(len(text) // CHARS_PER_TOKEN, len(text.split()))


class HeuristicComplexityAssessor:
    """Implements ``ComplexityAssessor`` with weighted regex/length rules."""

    def __init__(self, policy: ComplexityPolicy | None = None) -> None:
        """Use ``policy`` thresholds, or the defaults."""
        self._policy = policy or ComplexityPolicy()

    def assess(self, prompt: str) -> ComplexityAssessment:
        """Score ``prompt`` and bucket it into a complexity tier."""
        policy = self._policy
        signals: list[tuple[str, float]] = []

        tokens = estimate_tokens(prompt)
        if tokens >= policy.very_long_prompt_tokens:
            signals.append(("very_long_prompt", WEIGHT_VERY_LONG_PROMPT))
        elif tokens >= policy.long_prompt_tokens:
            signals.append(("long_prompt", WEIGHT_LONG_PROMPT))

        reasoning_terms = {match.lower() for match in _REASONING_RE.findall(prompt)}
        if len(reasoning_terms) >= DENSE_REASONING_MIN_TERMS:
            signals.append(("dense_reasoning_keywords", WEIGHT_REASONING_DENSE))
        elif reasoning_terms:
            signals.append(("reasoning_keywords", WEIGHT_REASONING))

        if (
            _CODE_FENCE_RE.search(prompt)
            or len(_CODE_SYNTAX_RE.findall(prompt)) >= CODE_MIN_SYNTAX_LINES
        ):
            signals.append(("code_block", WEIGHT_CODE))

        if len(_TURN_RE.findall(prompt)) >= MULTI_TURN_MIN_TURNS:
            signals.append(("multi_turn", WEIGHT_MULTI_TURN))

        if len(_LIST_ITEM_RE.findall(prompt)) >= MULTI_STEP_MIN_ITEMS:
            signals.append(("multi_step_instructions", WEIGHT_MULTI_STEP))

        if _MATH_RE.search(prompt):
            signals.append(("math_notation", WEIGHT_MATH))

        score = round(min(1.0, sum(weight for _, weight in signals)), 4)
        tier = ComplexityTier.HIGH if score >= policy.high_threshold else ComplexityTier.LOW
        return ComplexityAssessment(
            score=score,
            tier=tier,
            matched_heuristics=tuple(name for name, _ in signals),
        )
