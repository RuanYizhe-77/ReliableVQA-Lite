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
            confidence = _clamp(float(parsed.get("confidence", 0.0)))
            return VerifierResult(
                supported=bool(supported),
                confidence=confidence,
                explanation=str(parsed.get("explanation", "")),
                parse_ok=True,
            )

    lower = raw_output.lower()
    supported = bool(re.search(r"\b(yes|supported|true)\b", lower))
    unsupported = bool(re.search(r"\b(no|not supported|false|unsupported)\b", lower))
    if unsupported and not supported:
        supported = False
    conf_match = re.search(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", raw_output, re.I)
    confidence = _clamp(float(conf_match.group(1))) if conf_match else 0.0
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

