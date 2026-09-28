import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..geometry.fisheye import PARAM_NAMES, fit_kb4
from ..geometry.se3 import make_transform, transform_points
from ..smpl.constants import HEAD, NUM_BODY_JOINTS
from ..smpl.model import SMPLJoints
from .features import gt_world_joints, head_pose
from .raw import CAMERAS, IMAGE_SIZE


def load_calib(path):
    with open(path) as f:
        c = json.load(f)
    for cam in c["cameras"].values():
        cam["T_cam_head"] = np.asarray(cam["T_cam_head"], np.float32)
        cam["params"] = np.asarray([cam["intrinsics"][k] for k in PARAM_NAMES], np.float64)
    c["head_joint_offset"] = np.asarray(c["head_joint_offset"], np.float32)
    return c


def _mean_transform(ts):
    rot = R.from_matrix(ts[:, :3, :3]).mean().as_matrix()
    spread = np.rad2deg(R.from_matrix(np.einsum("ij,njk->nik", rot.T, ts[:, :3, :3])).magnitude())
    return make_transform(rot, ts[:, :3, 3].mean(0)), spread


def calibrate(processed_dir, out_path, frames_per_subject=400, smpl_path="assets/smpl/SMPL_NEUTRAL.npz", seed=0):
    processed_dir = Path(processed_dir)
    smpl = SMPLJoints(smpl_path)
    rng = np.random.default_rng(seed)
    per_cam = {c: {"T": [], "pts": [], "pix": []} for c in CAMERAS}
    offsets = []
    subjects = sorted(p.name for p in processed_dir.iterdir() if (p / "annots.npz").exists())
    for s in subjects:
        a = dict(np.load(processed_dir / s / "annots.npz"))
        joints, _ = gt_world_joints(a, smpl)
        r_head, p_head = head_pose(a)
        t_world_head = make_transform(r_head, p_head)
        offsets.append(np.einsum("nji,nj->ni", r_head, joints[:, HEAD] - p_head))
        idx = rng.choice(len(joints), min(frames_per_subject, len(joints)), replace=False)
        for ci, c in enumerate(CAMERAS):
            extr = a["extrinsics"][:, ci].astype(np.float64)
            extr[:, 3] = [0, 0, 0, 1]
            per_cam[c]["T"].append(extr @ t_world_head)
            pts = transform_points(extr[idx], joints[idx].astype(np.float64))
            pix = a["p2d"][idx, ci].astype(np.float64)
            vis = (pix[..., 0] >= 0) & (pix[..., 0] < IMAGE_SIZE[0]) & (pix[..., 1] >= 0) & (pix[..., 1] < IMAGE_SIZE[1]) & (pts[..., 2] > 0.05)
            per_cam[c]["pts"].append(pts[vis])
            per_cam[c]["pix"].append(pix[vis])
    offsets = np.concatenate(offsets)
    result = {"image_size": list(IMAGE_SIZE), "num_joints": NUM_BODY_JOINTS, "quaternion_order": "wxyz", "cameras": {}}
    for c in CAMERAS:
        ts = np.concatenate(per_cam[c]["T"])
        t_cam_head, spread = _mean_transform(ts)
        pts, pix = np.concatenate(per_cam[c]["pts"]), np.concatenate(per_cam[c]["pix"])
        params, err = fit_kb4(pts, pix, IMAGE_SIZE)
        result["cameras"][c] = {
            "model": "kannala_brandt_4",
            "intrinsics": dict(zip(PARAM_NAMES, map(float, params))),
            "T_cam_head": t_cam_head.tolist(),
            "stats": {
                "reproj_px_mean": float(err.mean()), "reproj_px_median": float(np.median(err)), "num_points": int(len(err)),
                "extrinsic_rot_spread_deg_mean": float(spread.mean()), "extrinsic_rot_spread_deg_p99": float(np.percentile(spread, 99)),
                "extrinsic_trans_std_cm": (ts[:, :3, 3].std(0) * 100).tolist(),
            },
        }
    result["head_joint_offset"] = offsets.mean(0).tolist()
    result["head_joint_offset_resid_cm"] = float(np.linalg.norm(offsets - offsets.mean(0), axis=1).mean() * 100)
    result["subjects"] = subjects
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
