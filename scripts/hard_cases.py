"""Per-frame error by difficulty category (fast motion, legs occluded/visible, arms out of view) for a sequence of
progressively improved variants on one fold; also returns the hardest clip windows for visualisation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import predict_sequence
from meposer.models.meposer import anchor_joints

LEG = [4, 5, 7, 8, 10, 11]
ARM = [18, 19, 20, 21]


def variants(h, feat, roi):
    ik, fit = ["refine_wrists=true"], ["refine_2d=true"]
    roi_fit = [f"refine_2d_features={roi}", "refine_2d_weight=3", "refine_prior=0.03", "refine_iters=150"]
    base = [f"model.image_features={feat}"]
    return [
        ("MEPoser (reproduced)", f"{h}/two_stage/fold1/last.pt", base),
        ("+ augmentation + contact head", f"{h}/best_contact/fold1/last.pt", base),
        ("+ wrist IK", f"{h}/best_contact/fold1/last.pt", base + ik),
        ("+ stereo 2D fit", f"{h}/best_contact/fold1/last.pt", base + ik + fit),
        ("+ adaptive leg ROI", f"{h}/best_contact/fold1/last.pt", base + ik + fit + roi_fit),
        ("+ low-pass (final)", f"{h}/best_contact/fold1/last.pt", base + ik + fit + roi_fit + ["post_lowpass_hz=4"]),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/final", help="root with <model>/fold1/last.pt (scripts/reproduce.sh layout)")
    p.add_argument("--features", default="runs/cv/fold1/stage1_image/features")
    p.add_argument("--roi", default="runs/leg_roi_fold1/features")
    p.add_argument("--subjects", nargs="*", default=["0001", "0006"])
    p.add_argument("--out", default="reports/results/hard_cases")
    a = p.parse_args()
    torch.set_num_threads(6)
    cats, errs = None, {}
    for name, ck, extra in variants(a.runs, a.features, a.roi):
        per = []
        for subj in a.subjects:
            model, cfg = full_model_from_checkpoint(load_checkpoint(ck), ["device=cpu"] + extra)
            model.eval()
            cfg.data.val_subjects = [subj]
            seq = sequences_for(cfg, "val")[0]
            (s0, s1), out = max(predict_sequence(model, seq, torch.device("cpu"), cfg), key=lambda r: r[0][1] - r[0][0])
            gt = seq.joints_world[s0:s1]
            pred = anchor_joints(out["joints_fk"], torch.as_tensor(gt[:, 15])).numpy()
            per.append(np.linalg.norm(pred - gt, axis=-1) * 100)
            if cats is None or len(cats["subject"]) < sum(len(x) for x in per):
                p2d = seq.p2d[s0:s1]
                vis = ((p2d[..., 0] >= 0) & (p2d[..., 0] < 640) & (p2d[..., 1] >= 0) & (p2d[..., 1] < 480)).any(1)
                speed = np.zeros(len(gt)); speed[1:] = np.linalg.norm(np.diff(gt, axis=0), axis=-1).mean(1) * 30 * 100
                c = {"subject": np.full(len(gt), subj), "frame": seq.frames[s0:s1], "speed": speed,
                     "leg_vis": vis[:, LEG].mean(1), "arm_vis": vis[:, ARM].mean(1)}
                cats = c if cats is None else {k: np.concatenate([cats[k], c[k]]) for k in c}
        errs[name] = np.concatenate(per)
        print(name, "done", flush=True)
    fast = cats["speed"] >= np.percentile(cats["speed"], 75)
    groups = {
        "all frames": np.ones_like(fast),
        "fast motion (top 25% joint speed)": fast,
        "legs occluded (<50% leg joints visible)": cats["leg_vis"] < 0.5,
        "legs visible (all leg joints visible)": cats["leg_vis"] == 1.0,
        "arms out of view (no arm joint visible)": cats["arm_vis"] == 0.0,
        "fast + legs occluded": fast & (cats["leg_vis"] < 0.5),
    }
    parts = {"MPJPE": slice(None), "legs": LEG, "arms": ARM}
    lines = ["Fold holding out 0001 + 0006 (the two most active subjects); per-frame joint error in cm, head-aligned; each row adds one change to the previous row.", ""]
    for gname, m in groups.items():
        lines += [f"**{gname}** ({int(m.sum())} frames)", "", "| variant | MPJPE | legs | arms |", "|---|---|---|---|"]
        for name, e in errs.items():
            lines.append(f"| {name} | " + " | ".join(f"{e[m][:, j].mean():.2f}" for j in parts.values()) + " |")
        lines.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".md").write_text("\n".join(lines))
    # hardest 6 s windows per subject: speed x (1 - leg visibility) x (1 - arm visibility), on frames of the first variant
    base = errs[list(errs)[0]].mean(1)
    win = {}
    for subj in a.subjects:
        idx = np.flatnonzero(cats["subject"] == subj)
        score = cats["speed"][idx] / cats["speed"].max() + (1 - cats["leg_vis"][idx]) + 0.5 * (1 - cats["arm_vis"][idx])
        L = 180
        cs = np.convolve(score, np.ones(L), "valid")
        k = int(np.argmax(cs))
        win[subj] = {"start_frame": int(cats["frame"][idx][k]), "mean_speed_cm_s": float(cats["speed"][idx][k:k + L].mean()),
                     "leg_visible": float(cats["leg_vis"][idx][k:k + L].mean()), "arm_visible": float(cats["arm_vis"][idx][k:k + L].mean()),
                     "baseline_mpjpe": float(base[idx][k:k + L].mean()), "final_mpjpe": float(errs[list(errs)[-1]].mean(1)[idx][k:k + L].mean())}
    json.dump(win, open(a.out + "_windows.json", "w"), indent=1)
    print("\n".join(lines)); print(json.dumps(win, indent=1))


if __name__ == "__main__":
    main()
