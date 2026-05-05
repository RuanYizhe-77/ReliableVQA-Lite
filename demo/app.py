from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confidence.self_consistency import DIRECT_PROMPT
from src.confidence.verbalized import CONFIDENCE_PROMPT, parse_verbalized_output
from src.models.qwen_vl import QwenVL, QwenVLConfig
from src.utils.text import clean_short_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReliableVQA-Lite Gradio demo.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", default="data/models/huggingface")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vlm: QwenVL | None = None

    def get_vlm() -> QwenVL:
        nonlocal vlm
        if vlm is None:
            vlm = QwenVL(QwenVLConfig(model_name=args.model, cache_dir=args.cache_dir))
        return vlm

    def answer(image_path: str, question: str, method: str) -> dict:
        if not image_path or not question.strip():
            return {"answer": "", "confidence": 0.0, "raw": ""}
        model = get_vlm()
        if method == "verbalized":
            prompt = CONFIDENCE_PROMPT.format(question=question)
            raw = model.generate(image_path, prompt, {"max_new_tokens": 96, "do_sample": False})
            parsed = parse_verbalized_output(raw)
            return {"answer": parsed.answer, "confidence": parsed.confidence, "raw": raw}
        prompt = DIRECT_PROMPT.format(question=question)
        raw = model.generate(image_path, prompt, {"max_new_tokens": 32, "do_sample": False})
        return {"answer": clean_short_answer(raw), "confidence": 1.0, "raw": raw}

    demo = gr.Interface(
        fn=answer,
        inputs=[
            gr.Image(type="filepath", label="Image"),
            gr.Textbox(label="Question"),
            gr.Radio(["direct", "verbalized"], value="direct", label="Method"),
        ],
        outputs=gr.JSON(label="Prediction"),
        title="ReliableVQA-Lite",
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()

