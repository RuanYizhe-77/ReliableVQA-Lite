from __future__ import annotations

import numpy as np


def expected_calibration_error(records: list[dict], num_bins: int = 10) -> float:
    if not records:
        return 0.0
    confidences = np.asarray([float(r.get("confidence", 0.0)) for r in records], dtype=float)
    scores = np.asarray([float(r.get("vqa_score", 0.0)) for r in records], dtype=float)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        if end == 1.0:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences >= start) & (confidences < end)
        if not mask.any():
            continue
        ece += mask.mean() * abs(confidences[mask].mean() - scores[mask].mean())
    return float(ece)

