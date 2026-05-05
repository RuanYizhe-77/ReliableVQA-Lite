from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.vqav2 import load_vqav2_subset
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.selective_metrics import compute_selective_metrics, threshold_sweep
from src.evaluation.vqa_accuracy import vqa_soft_accuracy
from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reliable/selective VQA predictions.")
    parser.add_argument("--pred", required=True, help="Predictions JSONL.")
    parser.add_argument("--input", required=True, help="Reference subset JSONL.")
    parser.add_argument("--out", required=True, help="Output eval JSON.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Global abstention threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = {str(ex.question_id): ex for ex in load_vqav2_subset(args.input)}
    preds = read_jsonl(args.pred)
    records = []
    missing = 0
    for pred in preds:
        qid = str(pred["question_id"])
        ref = refs.get(qid)
        if ref is None:
            missing += 1
            continue
        score = vqa_soft_accuracy(str(pred.get("pred_answer", "")), ref.answers)
        confidence = max(0.0, min(1.0, float(pred.get("confidence", 0.0))))
        enriched = dict(pred)
        enriched.update(
            {
                "confidence": confidence,
                "answers": ref.answers,
                "vqa_score": score,
                "correct": score >= 0.6,
                "answered_at_gamma": confidence >= args.gamma,
            }
        )
        records.append(enriched)

    gamma_metrics = compute_selective_metrics(records, args.gamma)
    summary = {
        "num_predictions": len(preds),
        "num_evaluated": len(records),
        "num_missing_references": missing,
        "gamma": args.gamma,
        "mean_vqa_score": statistics.mean([r["vqa_score"] for r in records]) if records else 0.0,
        "mean_confidence": statistics.mean([r["confidence"] for r in records]) if records else 0.0,
        "ece_10_bins": expected_calibration_error(records, num_bins=10),
        "metrics_at_gamma": gamma_metrics.__dict__,
    }
    output = {
        "summary": summary,
        "examples": records,
        "threshold_sweep": threshold_sweep(records),
    }
    write_json(output, args.out)
    print(f"Evaluated {len(records)} predictions. Saved to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()

