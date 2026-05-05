from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.self_consistency import DIRECT_PROMPT, majority_vote
from src.data.vqav2 import load_vqav2_subset
from src.models.qwen_vl import QwenVL, QwenVLConfig
from src.utils.io import append_jsonl, read_jsonl
from src.utils.logging import setup_logging
from src.utils.text import clean_short_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-consistency VQA inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", default="data/models/huggingface")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)
    examples = load_vqav2_subset(args.input)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    out_path = Path(args.out)
    completed = set()
    if out_path.exists():
        completed = {str(row["question_id"]) for row in read_jsonl(out_path)}
        logger.info("Resuming from %s completed predictions", len(completed))

    vlm = QwenVL(
        QwenVLConfig(
            model_name=args.model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            cache_dir=args.cache_dir,
        )
    )
    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    for example in tqdm(examples, desc="self-consistency"):
        if str(example.question_id) in completed:
            continue
        prompt = DIRECT_PROMPT.format(question=example.question)
        sample_rows = []
        answers = []
        for sample_idx in range(args.num_samples):
            raw = vlm.generate(example.image_path, prompt, gen_cfg)
            answer = clean_short_answer(raw)
            answers.append(answer)
            sample_rows.append({"sample_idx": sample_idx, "raw_output": raw, "answer": answer})
        voted = majority_vote(answers)
        append_jsonl(
            {
                "question_id": example.question_id,
                "image_path": example.image_path,
                "question": example.question,
                "pred_answer": voted.answer,
                "confidence": voted.confidence,
                "method": "self_consistency",
                "raw_outputs": {
                    "prompt": prompt,
                    "samples": sample_rows,
                    "normalized_counts": voted.normalized_counts,
                    "entropy": voted.entropy,
                    "self_consistency_score": voted.confidence,
                },
            },
            out_path,
        )

    logger.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()

