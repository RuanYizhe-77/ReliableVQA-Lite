from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log

from src.utils.text import clean_short_answer, normalize_answer


DIRECT_PROMPT = (
    "Question: {question}\n"
    "Answer the question based on the image. Give a short answer only."
)


@dataclass(frozen=True)
class SelfConsistencyResult:
    answer: str
    confidence: float
    normalized_counts: dict[str, int]
    entropy: float


def answer_entropy(normalized_answers: list[str]) -> float:
    if not normalized_answers:
        return 0.0
    counts = Counter(normalized_answers)
    total = len(normalized_answers)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * log(p)
    return entropy


def majority_vote(raw_answers: list[str]) -> SelfConsistencyResult:
    cleaned = [clean_short_answer(answer) for answer in raw_answers]
    normalized = [normalize_answer(answer) for answer in cleaned]
    counts = Counter(answer for answer in normalized if answer)
    if not counts:
        return SelfConsistencyResult("", 0.0, {}, 0.0)

    first_index = {answer: normalized.index(answer) for answer in counts}
    best_norm, best_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], first_index[item[0]]),
    )[0]
    best_answer = next(
        cleaned[i] for i, norm in enumerate(normalized) if norm == best_norm
    )
    return SelfConsistencyResult(
        answer=best_answer,
        confidence=best_count / len(raw_answers),
        normalized_counts=dict(counts),
        entropy=answer_entropy(normalized),
    )

