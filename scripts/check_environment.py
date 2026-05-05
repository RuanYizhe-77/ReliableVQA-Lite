from __future__ import annotations

import importlib.util
import os
import platform
import sys


def available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Python executable: {sys.executable}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    print(f"Conda env: {conda_env or '<not set>'}")
    print(f"Conda prefix: {conda_prefix or '<not set>'}")
    if conda_env != "cat-sam":
        print("WARNING: expected conda environment 'cat-sam' for GPU/model work.")
    if conda_prefix and not sys.executable.startswith(conda_prefix):
        print(
            "WARNING: python executable is not inside CONDA_PREFIX. "
            "Use $CONDA_PREFIX/bin/python if pyenv shims are ahead of conda on PATH."
        )

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    except ImportError:
        print("torch: not installed")

    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
    except ImportError:
        print("transformers: not installed")

    checks = {
        "accelerate": "accelerate",
        "qwen-vl-utils": "qwen_vl_utils",
        "bitsandbytes": "bitsandbytes",
        "PIL/pillow": "PIL",
        "sklearn": "sklearn",
        "matplotlib": "matplotlib",
    }
    for label, module_name in checks.items():
        print(f"{label}: {'available' if available(module_name) else 'missing'}")


if __name__ == "__main__":
    main()
