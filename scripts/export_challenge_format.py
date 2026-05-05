from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export challenge-style Reliable VQA predictions.")
    parser.add_argument("--pred", required=True, help="Internal predictions JSONL.")
    parser.add_argument("--out", required=True, help="Output .json or .jsonl.")
    parser.add_argument("--threshold", type=float, required=True, help="Global gamma threshold.")
    parser.add_argument("--method", default=None, help="Override method name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for pred in read_jsonl(args.pred):
        confidence = max(0.0, min(1.0, float(pred.get("confidence", 0.0))))
        abstain = confidence < args.threshold
        rows.append(
            {
                "question_id": pred["question_id"],
                "answer": pred.get("pred_answer", ""),
                "confidence": confidence,
                "threshold": args.threshold,
                "abstain": abstain,
                "decision": "abstain" if abstain else "answer",
                "method": args.method or pred.get("method", ""),
            }
        )

    out_path = Path(args.out)
    if out_path.suffix == ".jsonl":
        write_jsonl(rows, out_path)
    elif out_path.suffix == ".json":
        write_json(rows, out_path)
    else:
        raise SystemExit("Output must end with .json or .jsonl")
    print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

