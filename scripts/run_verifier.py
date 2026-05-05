from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.verifier import VERIFIER_PROMPT, combine_base_and_verifier, parse_verifier_output
from src.data.vqav2 import load_vqav2_subset
from src.models.qwen_vl import QwenVL, QwenVLConfig
from src.utils.io import append_jsonl, read_jsonl
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verifier confidence over existing predictions.")
    parser.add_argument("--input", required=True, help="Input VQAv2 subset JSONL.")
    parser.add_argument("--base-pred", required=True, help="Base predictions JSONL.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", default="data/models/huggingface")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)
    examples = {str(ex.question_id): ex for ex in load_vqav2_subset(args.input)}
    base_preds = {str(row["question_id"]): row for row in read_jsonl(args.base_pred)}

    out_path = Path(args.out)
    completed = set()
    if out_path.exists():
        completed = {str(row["question_id"]) for row in read_jsonl(out_path)}

    vlm = QwenVL(
        QwenVLConfig(
            model_name=args.model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            cache_dir=args.cache_dir,
        )
    )

    for qid, base in tqdm(base_preds.items(), desc="verifier"):
        if qid in completed or qid not in examples:
            continue
        example = examples[qid]
        prompt = VERIFIER_PROMPT.format(question=example.question, answer=base["pred_answer"])
        raw = vlm.generate(
            example.image_path,
            prompt,
            {"max_new_tokens": args.max_new_tokens, "do_sample": False},
        )
        verifier = parse_verifier_output(raw)
        combined = combine_base_and_verifier(float(base.get("confidence", 0.0)), verifier)
        raw_outputs = dict(base.get("raw_outputs", {}) or {})
        raw_outputs["verifier"] = {
            "model_output": raw,
            "prompt": prompt,
            "supported": verifier.supported,
            "confidence": verifier.confidence,
            "explanation": verifier.explanation,
            "parse_ok": verifier.parse_ok,
        }
        append_jsonl(
            {
                "question_id": example.question_id,
                "image_path": example.image_path,
                "question": example.question,
                "pred_answer": base["pred_answer"],
                "confidence": combined,
                "method": "verifier",
                "raw_outputs": raw_outputs,
            },
            out_path,
        )
    logger.info("Saved verifier predictions to %s", out_path)


if __name__ == "__main__":
    main()

