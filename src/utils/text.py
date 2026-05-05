from __future__ import annotations

import math
import re
import string
from collections.abc import Mapping


ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def clean_short_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^answer\s*[:=-]\s*", "", text, flags=re.I).strip()
    if "\n" in text:
        text = text.splitlines()[0].strip()
    text = text.strip("` ")
    return text


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = answer.replace("\n", " ").replace("\t", " ")
    translator = str.maketrans({p: " " for p in string.punctuation if p not in {"."}})
    answer = answer.translate(translator)
    tokens = []
    for token in answer.split():
        if token in ARTICLES:
            continue
        tokens.append(NUMBER_MAP.get(token, token))
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def question_type(question: str) -> str:
    q = question.strip().lower()
    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ")):
        return "yes/no"
    if q.startswith(("how many", "what number", "what is the number")):
        return "number"
    return "other"


def answer_entropy_from_counts(counts: Mapping[str, int | float]) -> float:
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        p = float(value) / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

