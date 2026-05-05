from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def disk_line(path: Path) -> str:
    total, used, free = shutil.disk_usage(path)
    gb = 1024**3
    return f"{path}: used={used / gb:.1f}GB free={free / gb:.1f}GB total={total / gb:.1f}GB"


def du(path: Path) -> str:
    if not path.exists():
        return f"{path}: <missing>"
    result = subprocess.run(["du", "-sh", str(path)], check=False, capture_output=True, text=True)
    return result.stdout.strip() or f"{path}: <du unavailable>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-VL weights with Hugging Face resume support.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", default="data/models/huggingface")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Disk before download:")
    print(disk_line(Path.cwd()))
    print(du(cache_dir))

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install transformers or huggingface_hub in cat-sam."
        ) from exc

    local_path = snapshot_download(
        repo_id=args.model,
        cache_dir=str(cache_dir),
        resume_download=True,
    )
    print(f"Downloaded/prepared model: {args.model}")
    print(f"Local snapshot: {local_path}")
    print("Disk after download:")
    print(disk_line(Path.cwd()))
    print(du(cache_dir))


if __name__ == "__main__":
    main()

