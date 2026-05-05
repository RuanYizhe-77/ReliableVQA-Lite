# Leaderboard-Style Exports

These files are generated from the current VQAv2 validation-subset demo runs. They are useful for format checks and method comparison, but they are not official leaderboard submissions because the official VQA leaderboard expects predictions on the challenge test/test-dev questions.

## Standard VQA / EvalAI Format

Standard VQA EvalAI submissions are JSON arrays with only:

```json
[
  {"question_id": 123, "answer": "yes"}
]
```

Generated files:

- `vqa_evalai/qwen3b_direct_500_val_style.json`
- `vqa_evalai/qwen3b_verbalized_strict_500_val_style.json`
- `vqa_evalai/selector_verbalized_500_heldout_val_style.json`

Each also has a `.zip` copy for portals that expect an archive.

## Reliable VQA-Style Format

These keep confidence and abstention fields:

- `reliable_vqa/qwen3b_direct_500_gamma0.5.json`
- `reliable_vqa/qwen3b_verbalized_strict_500_gamma0.5.json`
- `reliable_vqa/selector_verbalized_500_heldout_gamma0.5.json`

The exact live Reliable VQA challenge format should be checked before upload.

## Current Comparison

| Method | Split | Mean VQA | Coverage @ 0.5 | Selective Accuracy | Risk | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B direct | val-500 | 0.8560 | 1.0000 | 0.8560 | 0.1440 | 0.1440 |
| Qwen2.5-VL-3B verbalized | val-500 | 0.8160 | 0.9960 | 0.8193 | 0.1807 | 0.1229 |
| selector over verbalized | held-out 150 | 0.8244 | 0.4467 | 0.9303 | 0.0697 | 0.2612 |

## Official Upload Path

For the standard VQA leaderboard, generate predictions on the official VQAv2 test-dev/test split and export them with:

```bash
python scripts/export_vqa_evalai_format.py \
  --pred results/<test_predictions>.jsonl \
  --out leaderboard_exports/vqa_evalai/<method>_testdev.json
```

Then upload the JSON file on the EvalAI VQA challenge page for the correct phase.
