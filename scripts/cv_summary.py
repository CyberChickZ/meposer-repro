import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["mpjpe", "pa_mpjpe", "mpjre", "lowerpe", "handpe", "pred_jitter"]


def main():
    p = argparse.ArgumentParser(description="side-by-side per-subject table for several K-fold runs (from crossval.json, last epoch)")
    p.add_argument("runs", nargs="+", help="name=dir")
    p.add_argument("--metric", default="mpjpe")
    p.add_argument("--out", default="reports/results/crossval_summary.md")
    a = p.parse_args()
    data = {}
    for item in a.runs:
        name, _, d = item.partition("=")
        f = Path(d) / "crossval.json"
        if f.exists():
            data[name] = json.load(open(f))["per_subject_last_epoch"]
    subjects = sorted(set().union(*[set(v) for v in data.values()]))
    lines = []
    for m in [a.metric] + [x for x in METRICS if x != a.metric]:
        lines += [f"**{m}** (held-out subject, fixed 20-epoch schedule, last epoch)", "",
                  "| model | " + " | ".join(subjects) + " | mean +- std | median | frame-weighted |", "|" + "---|" * (len(subjects) + 4)]
        for name, per in data.items():
            vals = np.array([per[s][m] if s in per else np.nan for s in subjects])
            frames = np.array([per[s]["num_frames"] if s in per else 0 for s in subjects])
            ok = ~np.isnan(vals)
            fw = (vals[ok] * frames[ok]).sum() / frames[ok].sum()
            lines.append(f"| {name} | " + " | ".join("-" if np.isnan(v) else f"{v:.2f}" for v in vals) + f" | {np.nanmean(vals):.2f} +- {np.nanstd(vals):.2f} | {np.nanmedian(vals):.2f} | {fw:.2f} |")
        lines.append("")
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines[: 4 + len(data)]))


if __name__ == "__main__":
    main()
