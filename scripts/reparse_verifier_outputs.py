from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.verifier import combine_base_and_verifier, parse_verifier_output
from src.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse saved verifier raw outputs and recompute combined confidence."
    )
    parser.add_argument("--pred", required=True, help="Verifier prediction JSONL with saved raw outputs.")
    parser.add_argument("--base-pred", required=True, help="Base prediction JSONL used before verifier.")
    parser.add_argument("--out", required=True, help="Output verifier prediction JSONL.")
    return parser.parse_args()


def _verifier_raw(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_outputs = dict(record.get("raw_outputs", {}) or {})
    verifier = dict(raw_outputs.get("verifier", {}) or {})
    raw = str(verifier.get("model_output", ""))
    if not raw:
        raise ValueError(f"Missing raw verifier model_output for question_id={record.get('question_id')}")
    return raw, verifier


def main() -> None:
    args = parse_args()
    base_by_qid = {str(row["question_id"]): row for row in read_jsonl(args.base_pred)}
    reparsed = []
    changed_parse_count = 0
    changed_confidence_count = 0

    for record in read_jsonl(args.pred):
        qid = str(record["question_id"])
        base = base_by_qid.get(qid)
        if base is None:
            raise ValueError(f"question_id={qid} not found in --base-pred")
        raw, previous_verifier = _verifier_raw(record)
        parsed = parse_verifier_output(raw)
        base_confidence = float(base.get("confidence", 0.0))
        combined = combine_base_and_verifier(base_confidence, parsed)

        raw_outputs = dict(record.get("raw_outputs", {}) or {})
        raw_outputs["verifier"] = {
            "model_output": raw,
            "prompt": previous_verifier.get("prompt", ""),
            "supported": parsed.supported,
            "confidence": parsed.confidence,
            "explanation": parsed.explanation,
            "parse_ok": parsed.parse_ok,
            "reparsed_from": args.pred,
        }
        out_record = {
            "question_id": record["question_id"],
            "image_path": record.get("image_path", base.get("image_path", "")),
            "question": record.get("question", base.get("question", "")),
            "pred_answer": record.get("pred_answer", base.get("pred_answer", "")),
            "confidence": combined,
            "method": "verifier",
            "raw_outputs": raw_outputs,
        }
        if bool(previous_verifier.get("parse_ok", False)) != parsed.parse_ok:
            changed_parse_count += 1
        if abs(float(record.get("confidence", 0.0)) - combined) > 1e-9:
            changed_confidence_count += 1
        reparsed.append(out_record)

    write_jsonl(reparsed, args.out)
    print(f"Wrote {len(reparsed)} reparsed predictions to {args.out}")
    print(f"Changed parse_ok for {changed_parse_count} rows")
    print(f"Changed final confidence for {changed_confidence_count} rows")


if __name__ == "__main__":
    main()
