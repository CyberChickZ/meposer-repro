"""Full metrics table (paper's Table 2 columns + HandPE + foot skating) from the per-subject json written by
final_table.py, with the EMHI paper's own numbers (Table 2, full dataset) for reference."""
import argparse
import json

import numpy as np

COLS = [("MPJRE", "mpjre"), ("MPJPE", "mpjpe"), ("PA-MPJPE", "pa_mpjpe"), ("UpperPE", "upperpe"), ("LowerPE", "lowerpe"),
        ("RootPE", "rootpe"), ("Jitter", "pred_jitter"), ("HandPE", "handpe"), ("FS", "foot_skate")]
# EMHI paper, Table 2 (full dataset); MPJRE deg, PE cm, Jitter unit not stated; HandPE and FS not reported
PAPER = {
    "Protocol 1": {"UnrealEgo": [None, 5.5, 3.9, 4.0, 7.7, 4.2, 592.5], "HMD-Poser": [4.6, 5.8, 2.8, 4.8, 7.1, 5.8, 114.9],
                   "MEPoser-CV": [5.4, 4.5, 2.9, 3.3, 6.3, 3.8, 511.0], "MEPoser-IMU": [5.0, 6.2, 3.6, 5.0, 8.0, 5.2, 121.7],
                   "MEPoser-Full": [4.1, 3.7, 2.5, 2.7, 5.1, 3.2, 161.8]},
    "Protocol 2": {"UnrealEgo": [None, 6.4, 4.3, 4.6, 8.9, 5.0, 610.5], "HMD-Poser": [4.9, 7.0, 3.4, 5.2, 9.7, 7.2, 165.7],
                   "MEPoser-CV": [6.0, 5.4, 3.5, 3.8, 7.8, 4.4, 566.5], "MEPoser-IMU": [5.7, 7.1, 4.2, 5.4, 9.9, 6.6, 161.7],
                   "MEPoser-Full": [4.7, 4.8, 2.9, 3.2, 7.0, 3.8, 204.9]},
}
NAMES = {"imu_only": "MEPoser-IMU (reproduced)", "cv_only": "MEPoser-CV (reproduced)", "two_stage": "MEPoser-Full (reproduced, two-stage)",
         "joint": "MEPoser-Full, joint training (first config)", "joint_b32": "MEPoser-Full, joint training (effective batch 32)",
         "best": "ours: + augmentation", "best_contact": "ours: + augmentation + contact head", "best_long": "ours: + augmentation + contact head, 40 epochs",
         "kitchen_sink": "ours: all training additions"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+")
    p.add_argument("--out", default="reports/results/metrics_full.md")
    a = p.parse_args()
    res = {}
    for f in a.json:
        res.update(json.load(open(f)))
    head = "| model | post | " + " | ".join(c for c, _ in COLS) + " |"
    lines = ["All metrics, 3-fold cross-validation by folder (6 held-out folders: 0000+0005, 0001+0006, 0002+0007), mean over "
             "folders, last epoch. MPJRE in degrees; PE in cm; Jitter as in HMD-Poser's code (the paper gives no unit); "
             "FS = mean foot-joint speed in cm/s on frames where the ground-truth foot is planted (our metric, not in the paper; "
             "ground truth itself: {gt_fs:.1f}). post: raw network output, IK = controller wrist IK, 2D = stereo reprojection fit, "
             "legROI2D = 2D fit on adaptive leg-crop peaks, LP4 = 4 Hz zero-phase low-pass, contact = contact fit. Rows of the ours "
             "models with 2D fitting were recomputed after the half-cell fix in peak decoding; the 2D-fit rows of the other models "
             "predate it and are about 0.05 cm pessimistic.", "", head, "|---|---|" + "---|" * len(COLS)]
    gt_fs = []
    for key, per in res.items():
        m, post = key.split("|")
        subs = sorted(per)
        if len(subs) < 6:
            continue
        vals = [np.mean([per[s][k] for s in subs]) for _, k in COLS]
        gt_fs.append(np.mean([per[s]["gt_foot_skate"] for s in subs]))
        lines.append(f"| {NAMES.get(m, m)} | {post} | " + " | ".join(f"{v:.2f}" if c != "Jitter" else f"{v:.0f}" for (c, _), v in zip(COLS, vals)) + " |")
    lines[0] = lines[0].format(gt_fs=float(np.mean(gt_fs)))
    lines += ["", "EMHI paper, Table 2 (full dataset, different subjects and actions; not directly comparable):", "",
              "| protocol | method | " + " | ".join(c for c, _ in COLS[:7]) + " | HandPE | FS |", "|---|---|" + "---|" * (len(COLS))]
    for proto, rows in PAPER.items():
        for meth, v in rows.items():
            lines.append(f"| {proto} | {meth} | " + " | ".join("—" if x is None else (f"{x:.1f}") for x in v) + " | n/r | n/r |")
    lines += ["", "Protocol 1: unseen subjects, seen actions. Protocol 2: unseen subjects and unseen actions. n/r: not reported."]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
