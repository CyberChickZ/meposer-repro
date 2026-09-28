import argparse
from pathlib import Path

import numpy as np

from meposer.config import load_config
from meposer.data.dataset import load_sequences


def main():
    p = argparse.ArgumentParser(description="per-subject error of the cached stage-1 joint predictions against the annotations")
    p.add_argument("--features", default="runs/meposer/stage1_image/features")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="reports/results/feature_quality.md")
    a = p.parse_args()
    cfg = load_config(a.config)
    train, val = list(cfg.data.train_subjects), list(cfg.data.val_subjects)
    lines = ["| subject | role | cached local-joint MPJPE cm |", "|---|---|---|"]
    for s in load_sequences(cfg, train + val, load_images=False, feature_dir=a.features):
        err = np.linalg.norm(s.image_features["joints_local"] - s.joints_local, axis=-1).mean() * 100
        lines.append(f"| {s.subject} | {'train' if s.subject in train else 'val'} | {err:.2f} |")
    Path(a.out).write_text(f"Source: `{a.features}` (stage-1 checkpoint listed in its meta.json), config `{a.config}`.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
