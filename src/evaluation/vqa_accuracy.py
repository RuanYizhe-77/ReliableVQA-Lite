from __future__ import annotations

from typing import Any, Iterable

from src.utils.text import normalize_answer


def extract_answer_texts(answers: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for answer in answers:
        if isinstance(answer, dict) and "answer" in answer:
            texts.append(str(answer["answer"]))
        else:
            texts.append(str(answer))
    return texts


def vqa_soft_accuracy(pred_answer: str, human_answers: Iterable[Any]) -> float:
    pred = normalize_answer(pred_answer)
    answers = [normalize_answer(answer) for answer in extract_answer_texts(human_answers)]
    matches = sum(1 for answer in answers if answer == pred)
    return min(1.0, matches / 3.0)

