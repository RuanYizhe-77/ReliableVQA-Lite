from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import iter_jsonl, write_jsonl


@dataclass(frozen=True)
class VQAExample:
    question_id: int | str
    image_id: int
    image_path: str
    question: str
    answers: list[str]


def coco_val2014_image_path(image_dir: str | Path, image_id: int) -> Path:
    image_root = Path(image_dir)
    if image_root.name != "val2014":
        image_root = image_root / "val2014"
    return image_root / f"COCO_val2014_{image_id:012d}.jpg"


def load_vqav2_subset(path: str | Path) -> list[VQAExample]:
    examples: list[VQAExample] = []
    for row in iter_jsonl(path):
        examples.append(
            VQAExample(
                question_id=row["question_id"],
                image_id=int(row["image_id"]),
                image_path=row["image_path"],
                question=row["question"],
                answers=list(row.get("answers", [])),
            )
        )
    return examples


def _answer_texts(annotation: dict[str, Any]) -> list[str]:
    answers = annotation.get("answers", [])
    texts: list[str] = []
    for answer in answers:
        if isinstance(answer, dict) and "answer" in answer:
            texts.append(str(answer["answer"]))
        elif isinstance(answer, str):
            texts.append(answer)
    return texts


def build_subset(
    questions_path: str | Path,
    annotations_path: str | Path,
    image_dir: str | Path,
    num_samples: int,
) -> list[VQAExample]:
    questions_file = Path(questions_path)
    annotations_file = Path(annotations_path)
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    with questions_file.open("r", encoding="utf-8") as f:
        questions_json = json.load(f)
    with annotations_file.open("r", encoding="utf-8") as f:
        annotations_json = json.load(f)

    annotations = {
        str(item["question_id"]): item for item in annotations_json.get("annotations", [])
    }
    examples: list[VQAExample] = []

    for item in questions_json.get("questions", []):
        if len(examples) >= num_samples:
            break
        qid = item["question_id"]
        image_id = int(item["image_id"])
        image_path = coco_val2014_image_path(image_dir, image_id)
        if not image_path.exists():
            warnings.warn(f"Missing image for question_id={qid}: {image_path}")
            continue

        annotation = annotations.get(str(qid))
        if annotation is None:
            warnings.warn(f"Missing annotation for question_id={qid}")
            continue

        examples.append(
            VQAExample(
                question_id=qid,
                image_id=image_id,
                image_path=str(image_path),
                question=str(item["question"]),
                answers=_answer_texts(annotation),
            )
        )

    if len(examples) < num_samples:
        warnings.warn(f"Requested {num_samples} examples, found {len(examples)} usable examples")
    return examples


def save_subset(examples: Iterable[VQAExample], out_path: str | Path) -> None:
    write_jsonl([asdict(example) for example in examples], out_path)

