import json
import sys
from pathlib import Path

import numpy as np
import torch

from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import evaluate_sequences

MODELS = ["imu_only", "cv_only", "two_stage", "joint", "joint_b32", "best", "best_contact", "kitchen_sink"]
ROI_MODELS = {"best", "best_contact", "kitchen_sink", "two_stage", "cv_only"}
POST = {"raw": [], "IK+LP4": ["refine_wrists=true", "post_lowpass_hz=4"], "IK+2D+LP4": ["refine_wrists=true", "refine_2d=true", "post_lowpass_hz=4"],
        "IK+legROI2D+LP4": ["refine_wrists=true", "refine_2d=true", "post_lowpass_hz=4", "refine_2d_features=runs/leg_roi_{fold}/features",
                            "refine_2d_weight=3", "refine_prior=0.03", "refine_iters=150"]}


def main(root):
    root = Path(root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for m in MODELS:
        for post, extra in POST.items():
            if post == "IK+legROI2D+LP4" and m not in ROI_MODELS:
                continue
            per = {}
            folds = sorted((root / m).glob("fold*/last.pt"))
            if not folds or not all("] done" in (ck.parent / "log.txt").read_text() for ck in folds):
                print(m, "skipped (not finished)", flush=True)
                continue
            for ck in folds:
                ckpt = load_checkpoint(ck)
                fold = ck.parent.name
                ex = [e.replace("{fold}", fold) for e in extra]
                if any("refine_2d_features" in e for e in ex) and not Path(ex[[i for i, e in enumerate(ex) if "refine_2d_features" in e][0]].split("=", 1)[1]).exists():
                    continue
                model, cfg = full_model_from_checkpoint(ckpt, [f"device={dev}"] + ex)
                model = model.to(dev).eval()
                names, rows, _ = evaluate_sequences(model, sequences_for(cfg, "val"), torch.device(dev), cfg)
                per.update(dict(zip(names, rows)))
            if per:
                res[(m, post)] = per
            print(m, post, "done", flush=True)
    subs = sorted(next(iter(res.values())))
    lines = ["| model | post | " + " | ".join(subs) + " | MPJPE mean | PA | MPJRE | HandPE | LowerPE | Jitter (GT) |", "|" + "---|" * (len(subs) + 8)]
    for (m, post), per in res.items():
        if not per:
            continue
        g = lambda key: np.mean([per[s][key] for s in subs])
        lines.append(f"| {m} | {post} | " + " | ".join(f"{per[s]['mpjpe']:.2f}" for s in subs) +
                     f" | **{g('mpjpe'):.2f}** | {g('pa_mpjpe'):.2f} | {g('mpjre'):.2f} | {g('handpe'):.2f} | {g('lowerpe'):.2f} | {g('pred_jitter'):.0f} ({g('gt_jitter'):.0f}) |")
    table = "\n".join(lines)
    (root / "final_table.md").write_text("3-fold cross-validation by subject (folds 0-2: held out 0000+0005, 0001+0006, 0002+0007); every model on the same GPU-trained per-fold stage-1 features; last epoch; MPJPE cm per held-out subject. post: raw network output, or test-time wrist IK from the controllers + 4 Hz zero-phase low-pass (offline).\n\n" + table + "\n")
    json.dump({f"{m}|{p}": v for (m, p), v in res.items()}, open(root / "final_table.json", "w"), indent=1)
    print(table)


if __name__ == "__main__":
    main(sys.argv[1])
