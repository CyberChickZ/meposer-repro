"""Per-subject data audit of the EMHI subset: inventory, completeness, consistency/accuracy and distributions.
Writes reports/results/DATA_AUDIT.md. Reads the raw folders (file inventory) and the prepared arrays (values)."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from meposer.config import load_config
from meposer.data.calibrate import load_calib
from meposer.data.raw import CAMERAS, IMU_KEYS
from meposer.engine.build import sequences_for
from meposer.geometry.fisheye import project_kb4
from meposer.data.features import quat_wxyz_to_matrix_np as quaternion_wxyz_to_matrix
from meposer.geometry.se3 import transform_points

LEG, ARM, FEET = [4, 5, 7, 8, 10, 11], [18, 19, 20, 21], [7, 8, 10, 11]
BONES = [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21)]
WINDOW, STRIDE = 32, 4


def segments(frames):
    return np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != 1) + 1)


def longest_run(x):
    """longest run of consecutive identical rows"""
    same = np.all(np.isclose(x[1:], x[:-1], atol=0, rtol=0), axis=1)
    best = run = 0
    for s in same:
        run = run + 1 if s else 0
        best = max(best, run)
    return best + 1 if best else 1, float(same.mean()) if len(same) else 0.0


def inventory(raw, s):
    ann = {int(p.stem) for p in (raw / s / "annots").glob("*.npy")}
    img = [{int(p.stem) for p in (raw / s / "camera" / c).glob("*.png")} for c in CAMERAS]
    return ann, img


def audit_subject(raw, proc, seq, calib):
    s = seq.subject
    a = dict(np.load(proc / s / "annots.npz"))
    ann, img = inventory(raw, s)
    f = a["frame"]
    segs = segments(f)
    lens = np.array([len(x) for x in segs])
    gaps = np.diff(f)[np.diff(f) != 1] - 1
    r = {"subject": s}
    # 1. inventory
    r["annot_files"], r["cam0_png"], r["cam1_png"] = len(ann), len(img[0]), len(img[1])
    r["img_without_annot"] = len((img[0] | img[1]) - ann)
    r["annot_without_img"] = len(ann - (img[0] & img[1]))
    r["cam0_cam1_mismatch"] = len(img[0] ^ img[1])
    r["id_range"] = f"{f.min()}-{f.max()}"
    r["missing_ids"] = int(f.max() - f.min() + 1 - len(f))
    r["segments"], r["longest_seg"], r["median_seg"] = len(segs), int(lens.max()), int(np.median(lens))
    r["gap_sizes"] = dict(Counter(gaps.tolist()).most_common(4))
    r["train_windows"] = int(sum(max(0, (l - WINDOW) // STRIDE + 1) for l in lens))
    r["eval_frames"] = int(lens[lens >= 4].sum())
    # 2. completeness: NaN/Inf, all-zero rows, frozen runs, duplicated fields
    fields = {k: a[k].reshape(len(f), -1) for k in ["root_orient", "pose", "shape", "trans", "p2d", "extrinsics"] + IMU_KEYS}
    r["nonfinite"] = {k: int((~np.isfinite(v)).any(1).sum()) for k, v in fields.items() if (~np.isfinite(v)).any()}
    r["zero_rows"] = {k: int((np.abs(v) < 1e-12).all(1).sum()) for k, v in fields.items() if (np.abs(v) < 1e-12).all(1).any()}
    frozen = {}
    for k in IMU_KEYS + ["pose", "trans"]:
        run, frac = longest_run(fields[k])
        if run >= 3 or frac > 0.01:
            frozen[k] = f"longest {run}, {frac * 100:.1f}% repeats"
    r["frozen"] = frozen
    keys = list(fields)
    r["duplicate_fields"] = [f"{x}=={y}" for i, x in enumerate(keys) for y in keys[i + 1:]
                             if fields[x].shape == fields[y].shape and np.allclose(fields[x], fields[y])]
    r["duplicate_consecutive_poses"] = int(np.all(np.diff(np.concatenate([fields["pose"], fields["root_orient"], fields["trans"]], 1), axis=0) == 0, 1).sum())
    # 3. consistency / accuracy
    qn = [np.abs(np.linalg.norm(a[k].reshape(len(f), 4), axis=1) - 1).max() for k in IMU_KEYS if k.endswith("_rot")]
    r["quat_norm_err_max"] = float(max(qn))
    cont = np.concatenate([np.diff(x) for x in [seg for seg in segs if len(seg) > 1]]) if len(segs) else []
    within = np.concatenate([x[1:] for x in segs if len(x) > 1])  # indices whose predecessor is the previous frame
    jumps = {}
    for k in ["head_pos", "left_hand_pos", "right_hand_pos", "trans"]:
        d = np.linalg.norm(a[k].reshape(len(f), 3)[within] - a[k].reshape(len(f), 3)[within - 1], axis=1)
        jumps[k] = f"p99 {np.percentile(d, 99) * 100:.1f} / max {d.max() * 100:.1f} cm, >10 cm: {int((d > 0.1).sum())}"
    r["per_frame_jumps"] = jumps
    accs = {k: float(np.percentile(np.linalg.norm(a[k].reshape(len(f), 3), axis=1), 99.9)) for k in IMU_KEYS if k.endswith("_acc")}
    r["acc_p99.9"] = {k: round(v, 1) for k, v in accs.items()}
    r["betas_std_max"] = float(a["shape"].std(0).max())
    j = seq.joints_world
    bl = np.stack([np.linalg.norm(j[:, x] - j[:, y], axis=1) for x, y in BONES], 1)
    r["bone_len_std_mm_max"] = float(bl.std(0).max() * 1000)
    # headset vs GT head joint: offset in the headset frame (should be rigid) and time lag
    rh = quaternion_wxyz_to_matrix(a["head_rot"].reshape(len(f), 4))
    off = np.einsum("nji,nj->ni", rh, j[:, 15] - a["head_pos"].reshape(len(f), 3))
    r["head_offset_std_cm"] = float(np.linalg.norm(off.std(0)) * 100)
    r["head_offset_outliers>3cm"] = int((np.linalg.norm(off - np.median(off, 0), axis=1) > 0.03).sum())
    hv = np.diff(a["head_pos"].reshape(len(f), 3), axis=0)[within[:-1] - 0] if len(within) > 2 else None
    lag_best = 0
    if hv is not None:
        gv = np.diff(j[:, 15], axis=0)
        hv, gv = np.diff(a["head_pos"].reshape(len(f), 3), axis=0), gv
        ok = np.isin(np.arange(1, len(f)), within)
        corr = {L: float(np.sum(hv[max(0, L):len(hv) + min(0, L)][ok[max(0, L):len(hv) + min(0, L)]] *
                               gv[max(0, -L):len(gv) + min(0, -L)][ok[max(0, L):len(hv) + min(0, L)]])) for L in range(-5, 6)}
        lag_best = max(corr, key=corr.get)
    r["headset_vs_gt_lag_frames"] = int(lag_best)
    # controllers vs GT wrists
    for side, wj in (("left", 20), ("right", 21)):
        d = np.linalg.norm(a[f"{side}_hand_pos"].reshape(len(f), 3) - j[:, wj], axis=1) * 100
        r[f"{side}_ctrl_wrist_cm"] = f"median {np.median(d):.1f}, p99 {np.percentile(d, 99):.1f}, max {d.max():.1f}"
    # leg trackers vs GT shank orientation: relative rotation should be constant
    from meposer.geometry.rotations import axis_angle_to_matrix
    import torch
    aa = np.concatenate([a["root_orient"], a["pose"]], 1).reshape(len(f), -1, 3)
    rel = axis_angle_to_matrix(torch.as_tensor(aa[:, :22], dtype=torch.float32)).numpy()
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    glob_r = np.zeros_like(rel)
    for i, p in enumerate(parents):
        glob_r[:, i] = rel[:, i] if p < 0 else glob_r[:, p] @ rel[:, i]
    for side, knee in (("left", 4), ("right", 5)):
        rl = quaternion_wxyz_to_matrix(a[f"{side}_leg_rot"].reshape(len(f), 4))
        m = np.einsum("nji,njk->nik", rl, glob_r[:, knee])
        ref = m[len(m) // 2]
        ang = np.degrees(np.arccos(np.clip((np.einsum("ij,nij->n", ref, m) - 1) / 2, -1, 1)))
        r[f"{side}_leg_tracker_drift_deg"] = f"median {np.median(ang):.1f}, p99 {np.percentile(ang, 99):.1f}"
    # GT 3D joints projected with the provided extrinsics vs the provided 2D joints
    errs, extr_rot, extr_tr = [], [], []
    for c, cam in enumerate(CAMERAS):
        e = a["extrinsics"][:, c].astype(np.float64)
        e[:, 3] = [0, 0, 0, 1]
        uv = project_kb4(transform_points(e, j.astype(np.float64)), calib["cameras"][cam]["params"])
        p = a["p2d"][:, c]
        v = (p[..., 0] >= 0) & (p[..., 0] < 640) & (p[..., 1] >= 0) & (p[..., 1] < 480)
        errs.append(np.linalg.norm(uv - p, axis=-1)[v])
        dt = e[:, :3, 3] - seq.T_cam_world[:, c, :3, 3]
        extr_tr.append(np.linalg.norm(np.einsum("nij,nj->ni", np.transpose(e[:, :3, :3], (0, 2, 1)), dt), axis=1) * 100)
        extr_rot.append(np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", e[:, :3, :3], seq.T_cam_world[:, c, :3, :3]) - 1) / 2, -1, 1))))
    e2 = np.concatenate(errs)
    r["gt3d_vs_p2d_px"] = f"median {np.median(e2):.1f}, p99 {np.percentile(e2, 99):.1f}, >10 px {np.mean(e2 > 10) * 100:.2f}%"
    et, er = np.concatenate(extr_tr), np.concatenate(extr_rot)
    r["extrinsics_vs_headset_pose"] = f"{np.median(er):.2f} deg / {np.median(et):.1f} cm median, p99 {np.percentile(er, 99):.2f} deg / {np.percentile(et, 99):.1f} cm"
    # images
    im = seq.images if seq.images is not None else np.load(proc / s / seq.meta["images"], mmap_mode="r")
    means = im.reshape(len(im), 2, -1).mean(-1)
    r["dark_frames(<10)"] = int((means < 10).any(1).sum())
    r["saturated_frames(>245)"] = int((means > 245).any(1).sum())
    sub = np.asarray(im[:, :, ::8, ::8], np.float32)
    dup = np.abs(sub[1:] - sub[:-1]).mean((2, 3)) < 0.5
    r["frozen_image_pairs"] = int(dup[np.isin(np.arange(1, len(f)), within)].any(1).sum())
    # 4. distributions
    p = a["p2d"]
    vis = ((p[..., 0] >= 0) & (p[..., 0] < 640) & (p[..., 1] >= 0) & (p[..., 1] < 480)).any(1)
    sp = np.zeros(len(f))
    sp[within] = np.linalg.norm(j[within] - j[within - 1], axis=-1).mean(1) * 30 * 100
    fs = np.zeros(len(f))
    fs[within] = np.linalg.norm(j[within][:, FEET] - j[within - 1][:, FEET], axis=-1).mean(1) * 30 * 100
    r["joint_speed_cm_s"] = f"median {np.median(sp):.1f}, p90 {np.percentile(sp, 90):.1f}"
    r["foot_speed_cm_s"] = f"median {np.median(fs):.1f}, p90 {np.percentile(fs, 90):.1f}"
    r["frames_foot_moving(>20cm/s)"] = f"{np.mean(fs > 20) * 100:.1f}%"
    r["legs_all_visible"] = f"{vis[:, LEG].all(1).mean() * 100:.1f}%"
    r["arms_none_visible"] = f"{(~vis[:, ARM]).all(1).mean() * 100:.1f}%"
    r["contact_rate_L/R"] = "/".join(f"{x * 100:.0f}%" for x in seq.contact.mean(0))
    r["head_height_m"] = f"{np.median(j[:, 15, 1] - seq.floor_y):.2f}" if np.ndim(seq.floor_y) == 0 else "n/a"
    r["betas0"] = f"{a['shape'][:, 0].mean():.2f}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="reports/results/DATA_AUDIT.md")
    arg = ap.parse_args()
    cfg = load_config("configs/contact_vertex.yaml", ["device=cpu", "model.modalities=[imu]"])
    cfg.data.processed = arg.processed
    subs = sorted(p.name for p in Path(arg.raw).iterdir() if (p / "annots").is_dir())
    cfg.data.train_subjects = subs
    calib = load_calib(cfg.data.calib)
    rows = []
    for seq in sequences_for(cfg, "train"):
        rows.append(audit_subject(Path(arg.raw), Path(arg.processed), seq, calib))
        print(json.dumps(rows[-1]), flush=True)
    groups = {
        "1. Inventory": ["annot_files", "cam0_png", "cam1_png", "img_without_annot", "annot_without_img", "cam0_cam1_mismatch", "id_range",
                         "missing_ids", "segments", "longest_seg", "median_seg", "gap_sizes", "train_windows", "eval_frames"],
        "2. Completeness": ["nonfinite", "zero_rows", "frozen", "duplicate_fields", "duplicate_consecutive_poses", "frozen_image_pairs",
                            "dark_frames(<10)", "saturated_frames(>245)"],
        "3. Consistency and accuracy": ["quat_norm_err_max", "per_frame_jumps", "acc_p99.9", "betas_std_max", "bone_len_std_mm_max",
                                        "head_offset_std_cm", "head_offset_outliers>3cm", "headset_vs_gt_lag_frames", "left_ctrl_wrist_cm",
                                        "right_ctrl_wrist_cm", "left_leg_tracker_drift_deg", "right_leg_tracker_drift_deg", "gt3d_vs_p2d_px",
                                        "extrinsics_vs_headset_pose"],
        "4. Distributions": ["joint_speed_cm_s", "foot_speed_cm_s", "frames_foot_moving(>20cm/s)", "legs_all_visible", "arms_none_visible",
                             "contact_rate_L/R", "head_height_m", "betas0"],
    }
    lines = ["# Data audit of the EMHI subset", "", "Generated by `python scripts/data_audit.py`. One column per subject.", ""]
    for g, keys in groups.items():
        lines += [f"## {g}", "", "| check | " + " | ".join(r["subject"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
        for k in keys:
            cells = []
            for r in rows:
                v = r[k]
                if isinstance(v, dict):
                    v = "; ".join(f"{a}: {b}" for a, b in v.items()) or "none"
                elif isinstance(v, list):
                    v = ", ".join(v) or "none"
                elif isinstance(v, float):
                    v = f"{v:.3g}"
                cells.append(str(v).replace("|", "/"))
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    Path(arg.out).write_text("\n".join(lines))
    json.dump(rows, open(Path(arg.out).with_suffix(".json"), "w"), indent=1)
    print(f"wrote {arg.out}")


if __name__ == "__main__":
    main()
