from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.selector import (
    features_from_record,
    predict_selector_confidence,
    save_selector,
    train_selector,
)
from src.data.vqav2 import load_vqav2_subset
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.selective_metrics import compute_selective_metrics, threshold_sweep
from src.evaluation.vqa_accuracy import vqa_soft_accuracy
from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny reliability selector/calibrator.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL with confidence features.")
    parser.add_argument("--input", required=True, help="Reference subset JSONL.")
    parser.add_argument("--out", default="results/selector.pkl")
    parser.add_argument("--metrics-out", default="results/selector_metrics.json")
    parser.add_argument("--calibrated-out", default=None, help="Held-out selector predictions JSONL.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-threshold", type=float, default=0.6)
    return parser.parse_args()


def _split_indices(labels: list[int], train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")
    indices = np.arange(len(labels))
    stratify = labels if len(set(labels)) == 2 and min(labels.count(0), labels.count(1)) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_idx.astype(int).tolist(), test_idx.astype(int).tolist()


def _classification_metrics(labels: list[int], probs: list[float]) -> dict[str, Any]:
    preds = [1 if prob >= 0.5 else 0 for prob in probs]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "brier": float(brier_score_loss(labels, probs)),
        "positive_labels": int(sum(labels)),
        "num_examples": len(labels),
    }
    if len(set(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["roc_auc"] = None
    return metrics


def _make_calibrated_records(
    records: list[dict[str, Any]],
    labels: list[int],
    probs: list[float],
    split_name: str,
    feature_names: list[str],
    label_threshold: float,
) -> list[dict[str, Any]]:
    calibrated = []
    for record, label, prob in zip(records, labels, probs):
        raw_outputs = dict(record.get("raw_outputs", {}) or {})
        raw_outputs["selector"] = {
            "base_method": record.get("method", ""),
            "base_confidence": float(record.get("confidence", 0.0)),
            "features": dict(zip(feature_names, features_from_record(record))),
            "label_correct": int(label),
            "label_threshold": float(label_threshold),
            "split": split_name,
        }
        calibrated.append(
            {
                "question_id": record["question_id"],
                "image_path": record.get("image_path", ""),
                "question": record.get("question", ""),
                "pred_answer": record.get("pred_answer", ""),
                "confidence": float(prob),
                "method": "selector",
                "raw_outputs": raw_outputs,
            }
        )
    return calibrated


def main() -> None:
    args = parse_args()
    refs = {str(ex.question_id): ex for ex in load_vqav2_subset(args.input)}
    records = []
    labels = []
    for pred in read_jsonl(args.pred):
        ref = refs.get(str(pred["question_id"]))
        if ref is None:
            continue
        record = dict(pred)
        score = vqa_soft_accuracy(record.get("pred_answer", ""), ref.answers)
        record["vqa_score"] = score
        records.append(record)
        labels.append(1 if score >= args.label_threshold else 0)
    if not records:
        raise ValueError("No predictions matched the reference subset question IDs.")

    train_idx, test_idx = _split_indices(labels, args.train_frac, args.seed)
    train_records = [records[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    artifacts = train_selector(train_records, train_labels)
    save_selector(artifacts, args.out)

    train_probs = predict_selector_confidence(artifacts, train_records)
    test_probs = predict_selector_confidence(artifacts, test_records)
    test_calibrated = _make_calibrated_records(
        test_records,
        test_labels,
        test_probs,
        split_name="test",
        feature_names=artifacts.feature_names,
        label_threshold=args.label_threshold,
    )
    if args.calibrated_out:
        write_jsonl(test_calibrated, args.calibrated_out)

    test_eval_records = []
    for record, prob in zip(test_records, test_probs):
        enriched = dict(record)
        enriched["confidence"] = float(prob)
        test_eval_records.append(enriched)
    selector_metrics_at_gamma = compute_selective_metrics(test_eval_records, threshold=0.5)

    write_json(
        {
            "num_examples": len(records),
            "positive_labels": int(sum(labels)),
            "train_frac": args.train_frac,
            "seed": args.seed,
            "label_threshold": args.label_threshold,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train": _classification_metrics(train_labels, train_probs),
            "test": _classification_metrics(test_labels, test_probs),
            "test_ece_10_bins": expected_calibration_error(test_eval_records, num_bins=10),
            "test_metrics_at_gamma_0_5": selector_metrics_at_gamma.__dict__,
            "test_threshold_sweep": threshold_sweep(test_eval_records),
            "feature_names": artifacts.feature_names,
        },
        args.metrics_out,
    )
    print(f"Saved selector to {args.out}")
    if args.calibrated_out:
        print(f"Saved held-out calibrated predictions to {args.calibrated_out}")


if __name__ == "__main__":
    main()
