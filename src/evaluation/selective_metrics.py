from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SelectiveMetrics:
    threshold: float
    coverage: float
    selective_accuracy: float
    risk: float
    effective_reliability_score: float
    answered: int
    total: int


def compute_selective_metrics(records: list[dict], threshold: float) -> SelectiveMetrics:
    total = len(records)
    if total == 0:
        return SelectiveMetrics(threshold, 0.0, 0.0, 0.0, 0.0, 0, 0)

    answered_records = [
        record for record in records if float(record.get("confidence", 0.0)) >= threshold
    ]
    answered = len(answered_records)
    answered_score_sum = sum(float(record.get("vqa_score", 0.0)) for record in answered_records)
    coverage = answered / total
    selective_accuracy = answered_score_sum / answered if answered else 0.0
    risk = 1.0 - selective_accuracy if answered else 0.0
    effective_reliability_score = answered_score_sum / total
    return SelectiveMetrics(
        threshold=threshold,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        risk=risk,
        effective_reliability_score=effective_reliability_score,
        answered=answered,
        total=total,
    )


def threshold_sweep(records: list[dict], num_thresholds: int = 101) -> list[dict]:
    grid = set(np.linspace(0.0, 1.0, num_thresholds).round(6).tolist())
    grid.update(round(float(record.get("confidence", 0.0)), 6) for record in records)
    return [
        asdict(compute_selective_metrics(records, threshold))
        for threshold in sorted(grid)
    ]

