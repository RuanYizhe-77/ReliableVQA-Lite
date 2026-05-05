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
from src.utils.text import normalize_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create baseline-vs-proposed qualitative comparison image.")
    parser.add_argument("--input", required=True, help="VQAv2 subset JSONL.")
    parser.add_argument("--baseline-pred", required=True, help="Baseline prediction JSONL.")
    parser.add_argument("--proposed-pred", required=True, help="Proposed/selective prediction JSONL.")
    parser.add_argument("--out", default="assets/baseline_vs_proposed_500.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--baseline-label", default="Direct baseline")
    parser.add_argument("--proposed-label", default="Proposed")
    parser.add_argument(
        "--subtitle",
        default="Baseline answers directly. Proposed uses confidence-aware VQA output and abstains below gamma.",
    )
    return parser.parse_args()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_by_qid(path: str) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in read_jsonl(path)}


def _is_clean(record: dict[str, Any]) -> bool:
    question = str(record.get("question", "")).lower()
    answer = str(record.get("pred_answer", "")).strip()
    if answer.startswith(("{", "[", "```")):
        return False
    return "conservative employer" not in question


def _build_records(
    refs: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    proposed: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for qid, prop in proposed.items():
        base = baseline.get(qid)
        ref = refs.get(qid)
        if base is None or ref is None or not _is_clean(prop):
            continue
        base_score = vqa_soft_accuracy(str(base.get("pred_answer", "")), ref.answers)
        prop_score = vqa_soft_accuracy(str(prop.get("pred_answer", "")), ref.answers)
        confidence = float(prop.get("confidence", 0.0))
        rows.append(
            {
                "question_id": qid,
                "image_path": ref.image_path,
                "question": ref.question,
                "answers": ref.answers,
                "baseline_answer": str(base.get("pred_answer", "")),
                "baseline_score": base_score,
                "proposed_answer": str(prop.get("pred_answer", "")),
                "proposed_score": prop_score,
                "proposed_confidence": confidence,
                "proposed_answered": confidence >= threshold,
                "answer_diff": normalize_answer(str(base.get("pred_answer", "")))
                != normalize_answer(str(prop.get("pred_answer", ""))),
            }
        )
    return rows


def _select(records: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    fix_answered = [
        r
        for r in records
        if r["answer_diff"]
        and r["proposed_answered"]
        and r["baseline_score"] < 0.6
        and r["proposed_score"] >= 0.6
    ]
    both_correct_different = [
        r
        for r in records
        if r["answer_diff"]
        and r["proposed_answered"]
        and r["baseline_score"] >= 0.6
        and r["proposed_score"] >= 0.6
    ]
    good_abstain = [
        r
        for r in records
        if r["answer_diff"]
        and not r["proposed_answered"]
        and r["proposed_score"] < 0.6
        and r["baseline_score"] < 0.6
    ]
    confident_correct = [
        r for r in records if r["proposed_answered"] and r["proposed_score"] >= 0.6
    ]
    abstain_cost = [
        r
        for r in records
        if r["answer_diff"] and not r["proposed_answered"] and r["proposed_score"] >= 0.6
    ]
    failure = [
        r
        for r in records
        if r["answer_diff"] and r["proposed_answered"] and r["proposed_score"] < 0.6
    ]

    fix_answered.sort(key=lambda r: (r["proposed_score"] - r["baseline_score"], r["proposed_confidence"]), reverse=True)
    both_correct_different.sort(key=lambda r: r["proposed_confidence"], reverse=True)
    good_abstain.sort(key=lambda r: r["proposed_confidence"])
    confident_correct.sort(key=lambda r: r["proposed_confidence"], reverse=True)
    abstain_cost.sort(key=lambda r: r["proposed_score"], reverse=True)
    failure.sort(key=lambda r: r["proposed_confidence"], reverse=True)

    selected = []
    for bucket in (
        fix_answered[:5],
        both_correct_different[:1],
        good_abstain[:2],
        abstain_cost[:1],
        failure[:2],
    ):
        selected.extend(bucket)
    seen = {row["question_id"] for row in selected}
    for row in [r for r in records if r["answer_diff"]] + records:
        if len(selected) >= max_samples:
            break
        if row["question_id"] not in seen:
            selected.append(row)
            seen.add(row["question_id"])
    return selected[:max_samples]


def _thumbnail(path: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (244, 246, 248))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    gap: int = 4,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + gap
    return y


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 0.6:
        return (30, 122, 75)
    if score >= 0.3:
        return (188, 112, 18)
    return (185, 54, 54)


def main() -> None:
    args = parse_args()
    refs = {str(ex.question_id): ex for ex in load_vqav2_subset(args.input)}
    rows = _build_records(
        refs,
        _read_by_qid(args.baseline_pred),
        _read_by_qid(args.proposed_pred),
        args.threshold,
    )
    selected = _select(rows, args.max_samples)
    if not selected:
        raise ValueError("No comparison samples could be selected.")

    margin = 28
    row_h = 250
    header_h = 92
    width = 1420
    height = header_h + margin + len(selected) * row_h
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(28, bold=True)
    head_font = _font(18, bold=True)
    body_font = _font(16)
    small_font = _font(14)
    strong_font = _font(16, bold=True)

    draw.text((margin, 24), "Baseline vs Proposed Reliable VQA", fill=(23, 31, 43), font=title_font)
    draw.text(
        (margin, 58),
        args.subtitle,
        fill=(86, 97, 113),
        font=small_font,
    )

    for idx, row in enumerate(selected):
        y = header_h + idx * row_h
        x = margin
        draw.rounded_rectangle((x, y, width - margin, y + row_h - 18), radius=8, fill=(255, 255, 255), outline=(218, 224, 232))

        thumb = _thumbnail(row["image_path"], (190, 150))
        canvas.paste(thumb, (x + 18, y + 22))

        qx = x + 230
        qy = y + 18
        qy = _wrapped(draw, (qx, qy), f"Q: {row['question']}", head_font, (23, 31, 43), 44)
        answers = ", ".join(str(answer) for answer in row["answers"][:5])
        _wrapped(draw, (qx, y + 180), f"Human: {answers}", small_font, (86, 97, 113), 60)

        bx = x + 690
        py = y + 26
        draw.text((bx, py), args.baseline_label, font=head_font, fill=(23, 31, 43))
        py += 32
        py = _wrapped(draw, (bx, py), f"answer: {row['baseline_answer']}", strong_font, (23, 31, 43), 36)
        py += 8
        score_color = _score_color(row["baseline_score"])
        draw.rounded_rectangle((bx, py, bx + 140, py + 30), radius=6, fill=score_color)
        draw.text((bx + 12, py + 6), f"VQA={row['baseline_score']:.2f}", font=small_font, fill=(255, 255, 255))
        py += 42
        draw.text((bx, py), "always answers", font=small_font, fill=(86, 97, 113))

        px = x + 1020
        py = y + 26
        draw.text((px, py), args.proposed_label, font=head_font, fill=(23, 31, 43))
        py += 32
        py = _wrapped(draw, (px, py), f"answer: {row['proposed_answer']}", strong_font, (23, 31, 43), 34)
        py += 8
        status = "ANSWER" if row["proposed_answered"] else "ABSTAIN"
        status_color = (30, 122, 75) if row["proposed_answered"] and row["proposed_score"] >= 0.6 else (188, 112, 18) if not row["proposed_answered"] else (185, 54, 54)
        draw.rounded_rectangle((px, py, px + 126, py + 30), radius=6, fill=status_color)
        draw.text((px + 12, py + 6), status, font=small_font, fill=(255, 255, 255))
        py += 42
        draw.text(
            (px, py),
            f"conf={row['proposed_confidence']:.2f} | VQA={row['proposed_score']:.2f}",
            font=small_font,
            fill=(86, 97, 113),
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
