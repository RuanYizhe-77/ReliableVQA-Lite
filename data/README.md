# Data

This directory is intentionally mostly empty in Git.

Expected local layout after running `bash scripts/download_vqav2_val.sh`:

```text
data/vqav2/raw/v2_OpenEnded_mscoco_val2014_questions.json
data/vqav2/raw/v2_mscoco_val2014_annotations.json
data/vqav2/images/val2014/COCO_val2014_000000xxxxxx.jpg
data/vqav2/subsets/vqav2_val_10.jsonl
```

Qwen weights are cached by default under `data/models/huggingface/`.

