import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run dirs, optionally name=dir")
    p.add_argument("--metric", default="mpjpe")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for item in a.runs:
        name, _, d = item.rpartition("=")
        name = name or Path(d).name
        train, val = load(Path(d) / "metrics.jsonl")
        axes[0].plot([r["step"] for r in train], [r["total"] for r in train], label=name, linewidth=0.8)
        key = a.metric if a.metric in val[0] else "local_mpjpe_cm"
        axes[1].plot([r["epoch"] for r in val], [r[key] for r in val], marker="o", label=f"{name} ({key})")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("train total loss"); axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation (cm)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
