from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("logs") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy-coverage and risk-coverage curves.")
    parser.add_argument("--eval", required=True, help="Evaluation JSON.")
    parser.add_argument("--out", required=True, help="Output file prefix.")
    return parser.parse_args()


def save_curve(rows: list[dict], y_key: str, ylabel: str, out_path: Path) -> None:
    points = sorted((row["coverage"], row[y_key]) for row in rows)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1.5)
    plt.xlabel("Coverage")
    plt.ylabel(ylabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data = read_json(args.eval)
    rows = data.get("threshold_sweep", [])
    if not rows:
        raise SystemExit("No threshold_sweep rows found in eval JSON.")
    prefix = Path(args.out)
    acc_path = prefix.with_name(prefix.name + "_accuracy_coverage.png")
    risk_path = prefix.with_name(prefix.name + "_risk_coverage.png")
    save_curve(rows, "selective_accuracy", "Selective accuracy", acc_path)
    save_curve(rows, "risk", "Risk", risk_path)
    print(f"Saved {acc_path}")
    print(f"Saved {risk_path}")


if __name__ == "__main__":
    main()
