import json
from pathlib import Path

import numpy as np

from ..config import load_config, save_config
from ..metrics.evaluate import REPORT_ORDER
from .common import load_checkpoint
from .pipeline import run as run_pipeline


def make_folds(subjects, folds):
    subjects = sorted(subjects)
    return [subjects[k::folds] for k in range(folds)]


def run(config_path, out_dir, overrides=(), folds=5, only_fold=None):
    cfg = load_config(config_path, overrides)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(set(cfg.data.train_subjects) | set(cfg.data.val_subjects))
    splits = make_folds(subjects, folds)
    save_config(cfg, out_dir / "config.yaml")
    with open(out_dir / "folds.json", "w") as f:
        json.dump({"subjects": subjects, "folds": splits}, f, indent=2)
    for k, held_out in enumerate(splits):
        if only_fold is not None and k != only_fold:
            continue
        fold_dir = out_dir / f"fold{k}"
        if (fold_dir / "last.pt").exists():
            print(f"== fold {k}: reusing {fold_dir}", flush=True)
            continue
        train = [s for s in subjects if s not in held_out]
        print(f"== fold {k}/{folds}: held out {held_out}, train {train}", flush=True)
        fold_overrides = list(overrides) + [f"data.train_subjects={json.dumps(train)}", f"data.val_subjects={json.dumps(held_out)}"]
        run_pipeline(config_path, fold_dir, fold_overrides)
    return aggregate(out_dir)


def aggregate(out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / "folds.json") as f:
        folds = json.load(f)["folds"]
    per_subject, per_subject_best = {}, {}
    for k in range(len(folds)):
        fold_dir = out_dir / f"fold{k}"
        if not (fold_dir / "last.pt").exists():
            continue
        last = load_checkpoint(fold_dir / "last.pt")
        per_subject.update(last["val_per_sequence"])
        if (fold_dir / "best.pt").exists():
            per_subject_best.update(load_checkpoint(fold_dir / "best.pt")["val_per_sequence"])
    if not per_subject:
        raise RuntimeError(f"no finished folds in {out_dir}")
    keys = [m for m in REPORT_ORDER if m in next(iter(per_subject.values()))]
    frames = np.array([per_subject[s]["num_frames"] for s in per_subject])
    summary = {}
    for m in keys:
        vals = np.array([per_subject[s][m] for s in per_subject])
        summary[m] = {"mean_over_subjects": float(vals.mean()), "std_over_subjects": float(vals.std()),
                      "frame_weighted": float((vals * frames).sum() / frames.sum()), "min": float(vals.min()), "max": float(vals.max())}
    lines = ["| held-out subject | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for s in sorted(per_subject):
        lines.append(f"| {s} | " + " | ".join(f"{per_subject[s][m]:.2f}" for m in keys) + " |")
    lines.append("| **mean +- std over subjects** | " + " | ".join(f"**{summary[m]['mean_over_subjects']:.2f} +- {summary[m]['std_over_subjects']:.2f}**" for m in keys) + " |")
    lines.append("| frame-weighted | " + " | ".join(f"{summary[m]['frame_weighted']:.2f}" for m in keys) + " |")
    if per_subject_best:
        lines.append("| (best-epoch selection, optimistic) mean | " + " | ".join(f"{np.mean([per_subject_best[s][m] for s in per_subject_best]):.2f}" for m in keys) + " |")
    table = "\n".join(lines)
    result = {"folds": folds, "finished_subjects": sorted(per_subject), "per_subject_last_epoch": per_subject, "per_subject_best_epoch": per_subject_best, "summary_last_epoch": summary,
              "protocol": "K-fold by subject; every fold retrains all stages on the remaining subjects; stage-2 numbers are from the fixed-schedule last epoch; the stage-1 image branch feeding it is its best epoch by held-out local MPJPE (a mild selection on the held-out fold)"}
    with open(out_dir / "crossval.json", "w") as f:
        json.dump(result, f, indent=2)
    (out_dir / "crossval.md").write_text(table + "\n")
    print(table)
    return result
