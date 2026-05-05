from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.text import answer_entropy_from_counts, question_type


@dataclass(frozen=True)
class SelectorArtifacts:
    model: Pipeline
    feature_names: list[str]


FEATURE_NAMES = [
    "confidence",
    "verbalized_confidence",
    "self_consistency_score",
    "verifier_score",
    "answer_length",
    "sample_entropy",
    "is_yes_no",
    "is_number",
    "is_other",
]


def features_from_record(record: dict[str, Any]) -> list[float]:
    raw = record.get("raw_outputs", {}) or {}
    pred_answer = str(record.get("pred_answer", ""))
    qtype = question_type(str(record.get("question", "")))
    counts = raw.get("normalized_counts") or raw.get("self_consistency", {}).get("normalized_counts") or {}
    verifier = raw.get("verifier", {}) or {}
    verbalized = raw.get("verbalized", {}) or {}
    return [
        float(record.get("confidence", 0.0)),
        float(verbalized.get("confidence", record.get("confidence", 0.0))),
        float(raw.get("self_consistency_score", record.get("confidence", 0.0))),
        float(verifier.get("confidence", 0.0)),
        float(len(pred_answer.split())),
        float(answer_entropy_from_counts(counts)),
        1.0 if qtype == "yes/no" else 0.0,
        1.0 if qtype == "number" else 0.0,
        1.0 if qtype == "other" else 0.0,
    ]


def train_selector(records: list[dict[str, Any]], labels: list[int]) -> SelectorArtifacts:
    if len(set(labels)) < 2:
        raise ValueError("Selector training needs both correct and incorrect examples.")
    x = np.asarray([features_from_record(record) for record in records], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x, y)
    return SelectorArtifacts(model=model, feature_names=FEATURE_NAMES)


def predict_selector_confidence(artifacts: SelectorArtifacts, records: list[dict[str, Any]]) -> list[float]:
    x = np.asarray([features_from_record(record) for record in records], dtype=np.float32)
    return artifacts.model.predict_proba(x)[:, 1].astype(float).tolist()


def save_selector(artifacts: SelectorArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(artifacts, f)
