from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.vqav2 import load_vqav2_subset
from src.evaluation.vqa_accuracy import vqa_soft_accuracy
from src.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a presentation image with qualitative VQA samples.")
    parser.add_argument("--input", required=True, help="VQAv2 subset JSONL.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL to visualize.")
    parser.add_argument("--out", default="assets/sample_predictions.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=6)
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _score_records(pred_path: str, refs: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    records = []
    for pred in read_jsonl(pred_path):
        qid = str(pred["question_id"])
        ref = refs.get(qid)
        if ref is None:
            continue
        score = vqa_soft_accuracy(str(pred.get("pred_answer", "")), ref.answers)
        confidence = float(pred.get("confidence", 0.0))
        records.append(
            {
                **pred,
                "image_path": ref.image_path,
                "question": ref.question,
                "vqa_score": score,
                "answered": confidence >= threshold,
            }
        )
    return records


def _choose_examples(records: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    records = [
        record
        for record in records
        if not str(record.get("pred_answer", "")).strip().startswith(("{", "[", "```"))
        and "conservative employer" not in str(record.get("question", "")).lower()
    ]
    answered_correct = [r for r in records if r["answered"] and r["vqa_score"] >= 0.6]
    abstained_low = [r for r in records if not r["answered"] and r["vqa_score"] < 0.6]
    abstained_correct = [r for r in records if not r["answered"] and r["vqa_score"] >= 0.6]
    answered_wrong = [r for r in records if r["answered"] and r["vqa_score"] < 0.6]

    answered_correct.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
    abstained_low.sort(key=lambda r: float(r.get("confidence", 0.0)))
    abstained_correct.sort(key=lambda r: float(r.get("confidence", 0.0)))
    answered_wrong.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)

    selected = []
    for bucket in (answered_correct[:2], abstained_low[:2], abstained_correct[:1], answered_wrong[:1]):
        selected.extend(bucket)
    seen = {str(r["question_id"]) for r in selected}
    for record in records:
        if len(selected) >= max_samples:
            break
        if str(record["question_id"]) not in seen:
            selected.append(record)
            seen.add(str(record["question_id"]))
    return selected[:max_samples]


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width_chars: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def _thumbnail(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (244, 246, 248))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    refs = {str(ex.question_id): ex for ex in load_vqav2_subset(args.input)}
    records = _score_records(args.pred, refs, args.threshold)
    selected = _choose_examples(records, args.max_samples)
    if not selected:
        raise ValueError("No visualizable examples found.")

    card_w, card_h = 720, 310
    cols = 2
    rows = (len(selected) + cols - 1) // cols
    margin = 28
    header_h = 76
    width = cols * card_w + (cols + 1) * margin
    height = header_h + rows * card_h + (rows + 1) * margin

    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28, bold=True)
    body_font = _font(17)
    small_font = _font(15)
    label_font = _font(16, bold=True)

    draw.text((margin, 24), "ReliableVQA-Lite: VQAv2 Samples", fill=(23, 31, 43), font=title_font)
    draw.text(
        (margin, 55),
        f"Prediction file: {Path(args.pred).name} | gamma={args.threshold}",
        fill=(86, 97, 113),
        font=small_font,
    )

    for idx, record in enumerate(selected):
        col = idx % cols
        row = idx // cols
        x = margin + col * (card_w + margin)
        y = header_h + margin + row * (card_h + margin)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=8, fill=(255, 255, 255), outline=(218, 224, 232))

        thumb = _thumbnail(str(record["image_path"]), (240, 180))
        canvas.paste(thumb, (x + 18, y + 18))

        confidence = float(record.get("confidence", 0.0))
        score = float(record["vqa_score"])
        answered = bool(record["answered"])
        status = "ANSWERED" if answered else "ABSTAIN"
        status_color = (30, 122, 75) if answered and score >= 0.6 else (188, 112, 18) if not answered else (185, 54, 54)

        tx = x + 278
        ty = y + 18
        ty = _draw_wrapped(draw, (tx, ty), f"Q: {record['question']}", body_font, (23, 31, 43), 43)
        ty += 8
        ty = _draw_wrapped(
            draw,
            (tx, ty),
            f"Prediction: {record.get('pred_answer', '')}",
            label_font,
            (23, 31, 43),
            43,
        )
        ty += 8
        draw.text((tx, ty), f"confidence={confidence:.2f} | VQA score={score:.2f}", fill=(86, 97, 113), font=small_font)
        ty += 28
        draw.rounded_rectangle((tx, ty, tx + 118, ty + 28), radius=6, fill=status_color)
        draw.text((tx + 12, ty + 5), status, fill=(255, 255, 255), font=small_font)

        answers = record.get("answers") or refs[str(record["question_id"])].answers
        answer_text = ", ".join(str(answer) for answer in answers[:5])
        _draw_wrapped(draw, (x + 18, y + 220), f"Human answers: {answer_text}", small_font, (86, 97, 113), 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
