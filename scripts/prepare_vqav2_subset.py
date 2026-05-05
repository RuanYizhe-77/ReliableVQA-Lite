from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.vqav2 import build_subset, save_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a real VQAv2 validation JSONL subset.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of usable examples to save.")
    parser.add_argument(
        "--questions",
        default="data/vqav2/raw/v2_OpenEnded_mscoco_val2014_questions.json",
        help="Path to VQAv2 validation questions JSON.",
    )
    parser.add_argument(
        "--annotations",
        default="data/vqav2/raw/v2_mscoco_val2014_annotations.json",
        help="Path to VQAv2 validation annotations JSON.",
    )
    parser.add_argument(
        "--image-dir",
        default="data/vqav2/images",
        help="Directory containing val2014 COCO images or its parent.",
    )
    parser.add_argument("--out", default=None, help="Output JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or f"data/vqav2/subsets/vqav2_val_{args.num_samples}.jsonl"
    examples = build_subset(
        questions_path=args.questions,
        annotations_path=args.annotations,
        image_dir=args.image_dir,
        num_samples=args.num_samples,
    )
    save_subset(examples, out_path)
    print(f"Saved {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()

