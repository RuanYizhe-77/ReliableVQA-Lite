from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


def disk_line(path: Path) -> str:
    total, used, free = shutil.disk_usage(path)
    gb = 1024**3
    return f"{path}: used={used / gb:.1f}GB free={free / gb:.1f}GB total={total / gb:.1f}GB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only the COCO val2014 images needed for a small VQAv2 smoke subset."
    )
    parser.add_argument(
        "--questions",
        default="data/vqav2/raw/v2_OpenEnded_mscoco_val2014_questions.json",
    )
    parser.add_argument("--image-dir", default="data/vqav2/images")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--scan-limit", type=int, default=None)
    return parser.parse_args()


def image_name(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    with questions_path.open("r", encoding="utf-8") as f:
        questions = json.load(f).get("questions", [])

    scan_limit = args.scan_limit or max(args.num_samples, args.num_samples * 3)
    image_ids: list[int] = []
    for item in questions[:scan_limit]:
        image_id = int(item["image_id"])
        if image_id not in image_ids:
            image_ids.append(image_id)
        if len(image_ids) >= args.num_samples:
            break

    out_dir = Path(args.image_dir) / "val2014"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Disk before subset image download:")
    print(disk_line(Path.cwd()))
    downloaded = 0
    skipped = 0
    for image_id in image_ids:
        name = image_name(image_id)
        out_path = out_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        url = f"http://images.cocodataset.org/val2014/{name}"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
        except urllib.error.URLError as exc:
            raise SystemExit(f"Failed to download {url}: {exc}") from exc
        tmp_path.replace(out_path)
        downloaded += 1

    print(f"Downloaded {downloaded}, skipped {skipped}, target dir: {out_dir}")
    print("Disk after subset image download:")
    print(disk_line(Path.cwd()))


if __name__ == "__main__":
    main()

