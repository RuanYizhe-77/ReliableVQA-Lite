from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.self_consistency import DIRECT_PROMPT
from src.confidence.verbalized import CONFIDENCE_PROMPT, parse_verbalized_output
from src.data.vqav2 import load_vqav2_subset
from src.models.qwen_vl import QwenVL, QwenVLConfig
from src.utils.io import append_jsonl, read_jsonl
from src.utils.logging import setup_logging
from src.utils.text import clean_short_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct or verbalized-confidence VQA inference.")
    parser.add_argument("--input", required=True, help="Input VQAv2 subset JSONL.")
    parser.add_argument("--out", required=True, help="Output predictions JSONL.")
    parser.add_argument("--method", choices=["direct", "verbalized"], default="direct")
    parser.add_argument("--backend", choices=["qwen", "api"], default="qwen")
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--model", default=None, help="Qwen model name.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_backend(args: argparse.Namespace, cfg: dict[str, Any]):
    if args.backend == "api":
        from src.models.api_vlm import APIVLM

        return APIVLM(api_url=args.api_url or cfg.get("api_url"))
    model_cfg = cfg.get("model", cfg)
    return QwenVL(
        QwenVLConfig(
            model_name=args.model or model_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
            device_map=args.device_map or model_cfg.get("device_map", "auto"),
            torch_dtype=args.torch_dtype or model_cfg.get("torch_dtype", "auto"),
            load_in_4bit=bool(args.load_in_4bit or model_cfg.get("load_in_4bit", False)),
            cache_dir=args.cache_dir or model_cfg.get("cache_dir", "data/models/huggingface"),
        )
    )


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        cfg.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        cfg.update({"do_sample": False})
    return cfg


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)
    cfg = load_yaml(args.config)
    examples = load_vqav2_subset(args.input)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    out_path = Path(args.out)
    completed = set()
    if out_path.exists():
        completed = {str(row["question_id"]) for row in read_jsonl(out_path)}
        logger.info("Resuming from %s completed predictions", len(completed))

    logger.info("Loading backend=%s", args.backend)
    vlm = build_backend(args, cfg)
    gen_cfg = generation_config(args)

    for example in tqdm(examples, desc=f"{args.method} answering"):
        if str(example.question_id) in completed:
            continue
        if args.method == "direct":
            prompt = DIRECT_PROMPT.format(question=example.question)
            raw = vlm.generate(example.image_path, prompt, gen_cfg)
            pred_answer = clean_short_answer(raw)
            confidence = 1.0
            raw_outputs = {"model_output": raw, "prompt": prompt}
        else:
            prompt = CONFIDENCE_PROMPT.format(question=example.question)
            raw = vlm.generate(example.image_path, prompt, gen_cfg)
            parsed = parse_verbalized_output(raw)
            pred_answer = parsed.answer
            confidence = parsed.confidence
            raw_outputs = {
                "model_output": raw,
                "prompt": prompt,
                "verbalized": {
                    "answer": parsed.answer,
                    "confidence": parsed.confidence,
                    "parse_ok": parsed.parse_ok,
                },
            }

        append_jsonl(
            {
                "question_id": example.question_id,
                "image_path": example.image_path,
                "question": example.question,
                "pred_answer": pred_answer,
                "confidence": confidence,
                "method": args.method,
                "raw_outputs": raw_outputs,
            },
            out_path,
        )

    logger.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
