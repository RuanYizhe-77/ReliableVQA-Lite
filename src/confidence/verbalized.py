from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.utils.text import clean_short_answer


CONFIDENCE_PROMPT = (
    "You are answering a visual question about the image.\n"
    "Question: {question}\n"
    "Return exactly one valid JSON object and no other text.\n"
    "The JSON schema is: {{\"answer\": \"short answer\", \"confidence\": 0.0}}.\n"
    "The confidence value must be a number between 0 and 1, where 1 means completely certain.\n"
    "Do not wrap the JSON in markdown."
)


@dataclass(frozen=True)
class VerbalizedPrediction:
    answer: str
    confidence: float
    parse_ok: bool


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    return candidates


def parse_verbalized_output(raw_output: str) -> VerbalizedPrediction:
    for candidate in _json_candidates(raw_output):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed and "confidence" in parsed:
            try:
                confidence = _clamp_confidence(float(parsed["confidence"]))
            except (TypeError, ValueError):
                continue
            return VerbalizedPrediction(
                answer=clean_short_answer(str(parsed["answer"])),
                confidence=confidence,
                parse_ok=True,
            )

    answer_match = re.search(r"answer\s*[:=]\s*['\"]?([^,'\"\n}]+)", raw_output, flags=re.I)
    conf_match = re.search(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", raw_output, flags=re.I)
    answer = clean_short_answer(answer_match.group(1) if answer_match else raw_output)
    confidence = 0.0
    if conf_match:
        confidence = _clamp_confidence(float(conf_match.group(1)))
    return VerbalizedPrediction(answer=answer, confidence=confidence, parse_ok=False)
