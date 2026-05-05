from __future__ import annotations

import json
import re
from dataclasses import dataclass


VERIFIER_PROMPT = (
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Given the image, question, and proposed answer, decide whether the proposed answer "
    "is visually supported by the image. Return JSON with keys: supported, confidence, explanation."
)


@dataclass(frozen=True)
class VerifierResult:
    supported: bool
    confidence: float
    explanation: str
    parse_ok: bool


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_confidence(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        word_map = {
            "very low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "moderate": 0.5,
            "high": 0.8,
            "very high": 0.95,
        }
        if normalized in word_map:
            return word_map[normalized]
        match = re.search(r"([01](?:\.\d+)?)", normalized)
        if match:
            return _clamp(float(match.group(1)))
        return 0.0
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return 0.0


def parse_verifier_output(raw_output: str) -> VerifierResult:
    candidates = [raw_output.strip()]
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", raw_output, re.DOTALL))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "supported" in parsed:
            supported = parsed["supported"]
            if isinstance(supported, str):
                supported = supported.strip().lower() in {"yes", "true", "supported"}
            confidence = _parse_confidence(parsed.get("confidence", 0.0))
            return VerifierResult(
                supported=bool(supported),
                confidence=confidence,
                explanation=str(parsed.get("explanation", "")),
                parse_ok=True,
            )

    lower = raw_output.lower()
    unsupported = bool(re.search(r"\b(no|not supported|false|unsupported)\b", lower))
    supported = bool(re.search(r"\b(yes|supported|true)\b", lower))
    if unsupported:
        supported = False
    conf_match = re.search(
        r"confidence[\"']?\s*[:=]\s*[\"']?([01](?:\.\d+)?|[a-z]+(?:\s+[a-z]+)?)",
        raw_output,
        re.I,
    )
    confidence = _parse_confidence(conf_match.group(1)) if conf_match else 0.0
    return VerifierResult(
        supported=supported,
        confidence=confidence,
        explanation=raw_output.strip(),
        parse_ok=False,
    )


def combine_base_and_verifier(base_confidence: float, verifier: VerifierResult) -> float:
    base = _clamp(base_confidence)
    if verifier.supported:
        return _clamp(base * verifier.confidence)
    return _clamp(base * (1.0 - verifier.confidence))
