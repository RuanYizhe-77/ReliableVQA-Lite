from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export internal predictions to standard VQA/EvalAI answer-only JSON."
    )
    parser.add_argument("--pred", required=True, help="Internal prediction JSONL.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--zip",
        default=None,
        help="Optional zip path containing the JSON file for upload portals that expect archives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    seen = set()
    for pred in read_jsonl(args.pred):
        question_id = pred["question_id"]
        if question_id in seen:
            raise ValueError(f"Duplicate question_id in predictions: {question_id}")
        seen.add(question_id)
        rows.append(
            {
                "question_id": int(question_id) if str(question_id).isdigit() else question_id,
                "answer": str(pred.get("pred_answer", "")).strip(),
            }
        )

    out_path = Path(args.out)
    if out_path.suffix != ".json":
        raise SystemExit("Standard VQA/EvalAI output must end with .json")
    write_json(rows, out_path, indent=2)

    if args.zip:
        zip_path = Path(args.zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        print(f"Exported {len(rows)} rows to {out_path} and {zip_path}")
    else:
        print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
