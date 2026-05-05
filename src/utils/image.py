from __future__ import annotations

from pathlib import Path

from PIL import Image


def verify_image(path: str | Path) -> None:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        image.verify()

