from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.selector import features_from_record, save_selector, train_selector
from src.data.vqav2 import load_vqav2_subset
from src.evaluation.vqa_accuracy import vqa_soft_accuracy
from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny reliability selector/calibrator.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL with confidence features.")
    parser.add_argument("--input", required=True, help="Reference subset JSONL.")
    parser.add_argument("--out", default="results/selector.pkl")
    parser.add_argument("--metrics-out", default="results/selector_metrics.json")
    return parser.parse_args()


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
        labels.append(1 if score >= 0.6 else 0)

    artifacts = train_selector(records, labels)
    save_selector(artifacts, args.out)
    x = np.asarray([features_from_record(record) for record in records], dtype=np.float32)
    probs = artifacts.model.predict_proba(x)[:, 1].tolist()
    train_acc = float(np.mean((np.asarray(probs) >= 0.5) == np.asarray(labels)))
    write_json(
        {
            "num_examples": len(records),
            "positive_labels": int(sum(labels)),
            "train_accuracy": train_acc,
            "feature_names": artifacts.feature_names,
        },
        args.metrics_out,
    )
    print(f"Saved selector to {args.out}")


if __name__ == "__main__":
    main()

