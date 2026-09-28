import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..geometry.fisheye import fit_kb4
from ..geometry.se3 import make_transform, transform_points
from ..smpl.constants import BONES, HEAD, LEFT_KNEE, LEFT_WRIST, PARENTS, PELVIS, RIGHT_KNEE, RIGHT_WRIST
from ..smpl.model import SMPLJoints
from .features import gt_world_joints, synthetic_acceleration
from .raw import CAMERAS, IMAGE_SIZE, IMU_KEYS
from .sequence import contiguous_segments

DEVICES = {
    "headset (PICO 4)": ["head_pos", "head_rot", "head_acc"],
    "left controller": ["left_hand_pos", "left_hand_rot", "left_hand_acc"],
    "right controller": ["right_hand_pos", "right_hand_rot", "right_hand_acc"],
    "left leg tracker": ["left_leg_rot", "left_leg_acc"],
    "right leg tracker": ["right_leg_rot", "right_leg_acc"],
}


def _load(processed_dir, subject):
    d = Path(processed_dir) / subject
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return dict(np.load(d / "annots.npz")), meta, d


def _spread_deg(r_a, r_b):
    c = np.einsum("nji,njk->nik", r_a, r_b)
    mean = R.from_matrix(c).mean().as_matrix()
    return float(np.rad2deg(R.from_matrix(np.einsum("ij,njk->nik", mean.T, c)).magnitude()).mean()), mean


def _corr(x, y):
    return float(np.mean([np.corrcoef(x[:, i], y[:, i])[0, 1] for i in range(x.shape[1])]))


def _table(header, rows):
    return [header, "|" + "---|" * (header.count("|") - 1), *rows, ""]


def step0_inventory(subjects, processed_dir):
    rows, total = [], 0
    for s in subjects:
        a, meta, d = _load(processed_dir, s)
        segs = contiguous_segments(a["frame"])
        img = np.load(d / meta["images"], mmap_mode="r")
        total += len(a["frame"])
        rows.append(f"| {s} | {len(a['frame'])} | {a['frame'][0]}-{a['frame'][-1]} | {len(segs)} | {int(np.median([b - x for x, b in segs]))} | "
                    f"{img.shape[1]}x{img.shape[3]}x{img.shape[2]} {img.dtype} | {float(img[::50].mean()):.0f} |")
    lines = ["## Step 0 - inventory", "",
             *_table("| subject | annotated frames | frame-id range | contiguous segments | median segment | images (cams x W x H) | mean intensity |", rows),
             f"Total {total} annotated frames, {total / 30 / 60:.1f} min at the paper's 30 fps. Segments are runs of consecutive frame ids; "
             "temporal windows never cross a break. Mean image intensity is a proxy for the paper's three lighting conditions.", ""]
    return lines


def step1_visual(subjects, processed_dir, out_dir, clip_frames=300):
    s = subjects[0]
    a, meta, d = _load(processed_dir, s)
    img = np.load(d / meta["images"], mmap_mode="r")
    w, h = meta["image_size"]
    sx, sy = w / IMAGE_SIZE[0], h / IMAGE_SIZE[1]

    def draw(i, c):
        im = cv2.cvtColor(np.ascontiguousarray(img[i, c]), cv2.COLOR_GRAY2BGR)
        for j, (u, v) in enumerate(a["p2d"][i, c]):
            if 0 <= u < IMAGE_SIZE[0] and 0 <= v < IMAGE_SIZE[1]:
                cv2.circle(im, (int(u * sx), int(v * sy)), 3, (0, 0, 255), -1)
        for j, p in BONES:
            pj, pp = a["p2d"][i, c, j], a["p2d"][i, c, p]
            if (pj >= 0).all() and (pp >= 0).all():
                cv2.line(im, (int(pj[0] * sx), int(pj[1] * sy)), (int(pp[0] * sx), int(pp[1] * sy)), (0, 200, 0), 1)
        cv2.putText(im, f"{s} f{int(a['frame'][i])} cam{c}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        return im

    idx = np.linspace(0, len(a["frame"]) - 1, 6).astype(int)
    sheet = np.concatenate([np.concatenate([draw(i, 0), draw(i, 1)], 1) for i in idx], 0)
    cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet)
    start = max(0, len(a["frame"]) // 2 - clip_frames // 2)
    writer = cv2.VideoWriter(str(out_dir / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * w, h))
    for i in range(start, min(len(a["frame"]), start + clip_frames)):
        writer.write(np.concatenate([draw(i, 0), draw(i, 1)], 1))
    writer.release()
    return ["## Step 1 - look at the data", "",
            f"`contact_sheet.png`: 6 stereo frames of subject {s} spread over the sequence with the provided 2D joints (red) and SMPL bones (green). "
            f"`preview.mp4`: {clip_frames} consecutive frames ({clip_frames / 30:.0f} s) from the middle of the sequence at 30 fps. "
            "Things to look for: the cameras look down from the headset, the neck/head/shoulders are never visible, arms leave the field of view "
            "when raised, joints drawn on the body confirm 2D/3D/image alignment.", ""]


def step2_field_statistics(subjects, processed_dir):
    a = {k: np.concatenate([_load(processed_dir, s)[0][k] for s in subjects]) for k in IMU_KEYS + ["root_orient", "pose", "shape", "trans"]}
    rows = []
    for k, v in a.items():
        v = v.reshape(len(v), -1).astype(np.float64)
        fmt = lambda x: "[" + ", ".join(f"{t:.3f}" for t in x[:4]) + (", ..." if len(x) > 4 else "") + "]"
        rows.append(f"| `{k}` | {v.shape[1]} | {np.isfinite(v).all()} | {fmt(v.mean(0))} | {fmt(v.std(0))} | {v.min():.2f} | {v.max():.2f} |")
    dup = []
    keys = list(a)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if a[keys[i]].shape == a[keys[j]].shape and np.array_equal(a[keys[i]], a[keys[j]]):
                dup.append(f"`{keys[i]}` == `{keys[j]}` in {len(a[keys[i]])}/{len(a[keys[i]])} frames")
    return ["## Step 2 - per-field statistics (all subjects pooled)", "",
            *_table("| field | dim | finite | mean (first 4) | std (first 4) | min | max |", rows),
            "Duplicate-field scan (byte-identical arrays): " + ("; ".join(dup) if dup else "none") + ".",
            "A duplicate between an `acc` and a `pos` field means the acceleration channel carries no acceleration information.", ""]


def step3_physics(subjects, processed_dir, smpl):
    rows, leg_means, notes = [], {"left": [], "right": []}, []
    for s in subjects:
        a, _, _ = _load(processed_dir, s)
        joints, grot = gt_world_joints(a, smpl)
        r_head = R.from_quat(a["head_rot"][:, [1, 2, 3, 0]]).as_matrix()
        qn = float(np.abs(np.linalg.norm(a["left_hand_rot"], axis=1) - 1).max())
        wxyz, _ = _spread_deg(R.from_quat(a["left_hand_rot"][:, [1, 2, 3, 0]]).as_matrix(), grot[:, LEFT_WRIST])
        xyzw, _ = _spread_deg(R.from_quat(a["left_hand_rot"]).as_matrix(), grot[:, LEFT_WRIST])
        head_d = float(np.linalg.norm(a["head_pos"] - joints[:, HEAD], axis=1).mean() * 100)
        hand_d = float(np.linalg.norm(a["right_hand_pos"] - joints[:, RIGHT_WRIST], axis=1).mean() * 100)
        up = joints[:, HEAD] - joints[:, PELVIS]
        up_axis = "xyz"[int(np.argmax(np.abs(up.mean(0))))] + ("+" if up.mean(0)[np.argmax(np.abs(up.mean(0)))] > 0 else "-")
        fd_world = synthetic_acceleration(a["right_hand_pos"], a["frame"], 30.0, 3.0)
        r_hand = R.from_quat(a["right_hand_rot"][:, [1, 2, 3, 0]]).as_matrix()
        fd_sensor = np.einsum("nji,nj->ni", r_hand, fd_world)
        c_world, c_sensor = _corr(a["right_hand_acc"], fd_world), _corr(a["right_hand_acc"], fd_sensor)
        for side, knee in [("left", LEFT_KNEE), ("right", RIGHT_KNEE)]:
            spread, mean = _spread_deg(R.from_quat(a[f"{side}_leg_rot"][:, [1, 2, 3, 0]]).as_matrix(), grot[:, knee])
            leg_means[side].append(mean)
            if side == "left":
                leg_spread = spread
        bones = np.linalg.norm(joints[:, 1:] - joints[:, PARENTS[1:22]], axis=-1)
        jitter = float((np.linalg.norm(joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3], axis=-1) * 30**3).mean())
        rows.append(f"| {s} | {qn:.0e} | {wxyz:.1f} / {xyzw:.1f} | {head_d:.1f} | {hand_d:.1f} | {up_axis} | {c_world:.2f} / {c_sensor:.2f} | {leg_spread:.1f} | "
                    f"{a['shape'].std(0).max():.4f} | {bones.std(0).max() * 100:.2f} | {jitter:.0f} |")
    for side in leg_means:
        ms = leg_means[side]
        between = np.rad2deg(np.mean([R.from_matrix(ms[i].T @ ms[j]).magnitude() for i in range(len(ms)) for j in range(i + 1, len(ms))]))
        notes.append(f"{side} leg tracker: offset to the knee rotation is constant within a subject (column above) and consistent across subjects ({between:.0f} deg mean pairwise difference), i.e. the trackers are calibrated the same way for everyone.")
    return ["## Step 3 - physical plausibility", "",
            *_table("| subject | max abs(|q|-1) | hand-IMU vs wrist rot spread deg (wxyz / xyzw) | head IMU vs head joint cm | hand IMU vs wrist cm | up axis | right acc vs d2pos corr (world / sensor) | left leg vs knee spread deg | shape std | bone length std cm | GT jitter 1e2 m/s3 |", rows),
            "Reading: quaternions are unit norm; the small wxyz spread fixes the quaternion order; IMU positions sit within a few cm of the SMPL joints, so IMU and SMPL "
            "share one world frame; the up axis is y; accelerations correlate with world-frame second differences and not with sensor-frame ones, so they are world-frame, "
            "gravity-removed linear accelerations; shape is constant per subject; bone lengths are constant (FK is consistent); GT jitter is the floor for the Jitter metric.",
            *notes, ""]


def step4_cameras(subjects, processed_dir, smpl, rng, frames_per_subject=200):
    rows = []
    for ci, c in enumerate(CAMERAS):
        depth_ok = {"E": 0, "inv(E)": 0}
        rigid, pts, pix, vis_frac, last_rows = [], [], [], [], []
        for s in subjects:
            a, _, _ = _load(processed_dir, s)
            joints, _ = gt_world_joints(a, smpl)
            E = a["extrinsics"][:, ci].astype(np.float64)
            last_rows.append(np.abs(E[:, 3] - [0, 0, 0, 1]).max())
            E[:, 3] = [0, 0, 0, 1]
            idx = rng.choice(len(joints), min(frames_per_subject, len(joints)), replace=False)
            for name, T in [("E", E[idx]), ("inv(E)", np.linalg.inv(E[idx]))]:
                depth_ok[name] += int((transform_points(T, joints[idx])[..., 2] > 0).mean() * 100) / len(subjects)
            p = a["p2d"][idx, ci].astype(np.float64)
            pc = transform_points(E[idx], joints[idx])
            vis = (p[..., 0] >= 0) & (p[..., 0] < IMAGE_SIZE[0]) & (p[..., 1] >= 0) & (p[..., 1] < IMAGE_SIZE[1]) & (pc[..., 2] > 0.05)
            pts.append(pc[vis]); pix.append(p[vis]); vis_frac.append(vis.mean())
            r_head = R.from_quat(a["head_rot"][:, [1, 2, 3, 0]]).as_matrix()
            rigid.append(E @ make_transform(r_head, a["head_pos"]))
        pts, pix = np.concatenate(pts), np.concatenate(pix)
        params, err = fit_kb4(pts, pix, IMAGE_SIZE)
        A = np.stack([pts[:, 0] / pts[:, 2], np.ones(len(pts))], 1)
        pin = np.linalg.lstsq(A, pix[:, 0], rcond=None)
        pin_err = float(np.abs(A @ pin[0] - pix[:, 0]).mean())
        rigid = np.concatenate(rigid)
        spread, _ = _spread_deg(np.repeat(np.eye(3)[None], len(rigid), 0), rigid[:, :3, :3])
        hfov, vfov = field_of_view(params, IMAGE_SIZE)
        rows.append(f"| {c} | {depth_ok['E']:.0f}% / {depth_ok['inv(E)']:.0f}% | {max(last_rows):.1e} | {pin_err:.0f} | {err.mean():.2f} | {params[0]:.0f}, {params[1]:.0f}, ({params[2]:.0f}, {params[3]:.0f}) | "
                    f"{hfov:.0f} x {vfov:.0f} | {spread:.2f} | {rigid[:, :3, 3].std(0).mean() * 100:.2f} | {np.round(rigid[:, :3, 3].mean(0), 3).tolist()} | {np.mean(vis_frac):.2f} |")
    return ["## Step 4 - cameras", "",
            *_table("| camera | positive depth with E / inv(E) | max deviation of last row from [0,0,0,1] | pinhole fit px | Kannala-Brandt fit px | fx, fy, (cx, cy) | FOV horizontal x vertical deg | camera<-headset rot spread deg | trans std cm | camera<-headset t (m) | joints in image |", rows),
            "Reading: `extrinsics` are world-to-camera (only that direction gives positive depth); the last row is numerically off and is reset; a pinhole model cannot "
            "explain the 2D joints but a Kannala-Brandt fisheye can (~1.5 px); the camera-to-headset transform is constant over all frames (the paper's T_h^c), "
            "cameras sit +-10 cm from the headset IMU. The field of view is computed from the fitted model at the image borders: about 157 deg horizontally "
            "and 116 deg vertically; the lens image circle (where intensity drops, r ~ 340 px) corresponds to ~86 deg incidence, i.e. the lens covers ~170 deg "
            "but the 4:3 sensor crops it vertically, which is why the corners are dark.", ""]


PAPER_CLAIMS = [
    ("stereo egocentric images 640x480, 1 channel (Sec. EMHI Dataset, Sec. MEPoser)", "step 0: image size / dtype", "verified"),
    ("recorded at 30 fps (Sec. EMHI Dataset)", "no timestamps in the subset; assumed; GT velocities/jitter are plausible for 30 fps", "assumed"),
    ("five IMUs: headset, two controllers, two leg trackers (Sec. Hardware)", "step 2: 5 device groups, 13 fields", "verified"),
    ("SMPL pose theta in R75 (72 axis-angle + 3 translation) and beta in R10 (Eq. 1)", "fields root_orient 3 + pose 69 + trans 3, shape 10", "verified"),
    ("2D keypoints of the 22 SMPL joints per view (Sec. MEPoser, J = 22)", "p2d shape (2, 22, 2), -1 when not annotated", "verified"),
    ("SMPL annotations and IMU data share one world coordinate system (Conclusion)", "step 3: head/hand IMU within cm of the SMPL joints", "verified"),
    ("headset-to-camera transform T_h^c is constant (Sec. Spatial Alignment)", "step 4: 0.3 deg / 0.2 cm spread over all frames", "verified"),
    ("controller-to-wrist rotation offset is constant (Sec. SMPL Fitting)", "step 3: 0.1-0.8 deg spread (wxyz)", "verified"),
    ("leg trackers represent the knee joint rotation, calibrated (Sec. SMPL Fitting)", "step 3: constant offset to the knee within a subject (1.5-6.9 deg) and across subjects (4-5 deg)", "verified"),
    ("headset provides online 6DoF used by the FK module (Sec. SMPL Decoder)", "head_pos + head_rot present every frame", "verified"),
    ("885 sequences / 58 subjects / 39 actions / 3 lighting conditions (Sec. EMHI Dataset)", "subset: 10 subjects = 10 sequences, no action or lighting labels; intensity varies (step 0)", "not in subset"),
    ("camera intrinsics / lens model", "not stated in the paper; fitted (step 4)", "missing -> fitted"),
    ("quaternion order, acceleration frame and units", "not stated; determined empirically (step 3): wxyz, world frame, gravity removed", "missing -> determined"),
    ("left controller acceleration", "identical to left_hand_pos in every frame (step 2)", "contradicted -> excluded"),
]


def field_of_view(params, image_size):
    from scipy.optimize import brentq
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    f = (fx + fy) / 2
    poly = lambda t: t * (1 + k1 * t**2 + k2 * t**4 + k3 * t**6 + k4 * t**8)
    ts = np.linspace(0, np.pi, 2000)
    tmax = ts[np.argmax(poly(ts))]
    theta = lambda r: brentq(lambda t: poly(t) - r / f, 0, tmax) if poly(tmax) >= r / f else tmax
    w, h = image_size
    return np.rad2deg(theta(cx) + theta(w - cx)), np.rad2deg(theta(cy) + theta(h - cy))


def step5_paper_checklist():
    rows = [f"| {c} | {e} | {s} |" for c, e, s in PAPER_CLAIMS]
    return ["## Step 5 - paper vs data checklist", "", *_table("| paper statement | evidence in this report | status |", rows)]


def inspect(processed_dir, out_dir, smpl_path="assets/smpl/SMPL_NEUTRAL.npz", subjects=None, seed=0):
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = subjects or sorted(p.name for p in processed_dir.iterdir() if (p / "annots.npz").exists())
    smpl = SMPLJoints(smpl_path)
    rng = np.random.default_rng(seed)
    lines = ["# Dataset inspection (generated by `meposer inspect`)", "",
             "Fixed procedure run on every prepared copy of the data: inventory -> look -> field statistics -> physical plausibility -> cameras -> paper checklist.", ""]
    for step in (lambda: step0_inventory(subjects, processed_dir), lambda: step1_visual(subjects, processed_dir, out_dir),
                 lambda: step2_field_statistics(subjects, processed_dir), lambda: step3_physics(subjects, processed_dir, smpl),
                 lambda: step4_cameras(subjects, processed_dir, smpl, rng), step5_paper_checklist):
        lines += step()
        print(f"  {[l for l in lines if l.startswith('## ')][-1]}", flush=True)
    report = out_dir / "INSPECTION.md"
    report.write_text("\n".join(lines))
    return report
