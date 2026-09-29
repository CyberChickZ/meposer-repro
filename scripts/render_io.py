"""Input -> output video of one clip, without ground truth or comparisons.
input:  both raw headset fisheye images + the five tracked devices (headset and controller positions as 3D trails,
        shank-tracker accelerations as curves)
output: our predicted SMPL mesh (3D view) and the same mesh projected into both fisheye views."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from meposer.data.calibrate import load_calib
from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import predict_sequence
from meposer.geometry.fisheye import project_kb4
from meposer.geometry.rotations import rotation_6d_to_axis_angle
from meposer.geometry.se3 import transform_points
from meposer.smpl.vertices import load_smpl_full
from render_mesh import POSTS, mesh_world, pinhole, raster

BG, FG, ACCENT = (24, 24, 24), (235, 235, 235), (60, 140, 255)
DEV_COL = {"headset": (255, 255, 255), "left controller": (255, 170, 60), "right controller": (80, 200, 255),
           "left leg tracker": (120, 220, 120), "right leg tracker": (200, 120, 220)}


def label(img, text, org, s, col=FG):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, max(1, round(s * 2)), cv2.LINE_AA)


def view(centre, yaw_deg=35):
    yaw = np.deg2rad(yaw_deg)
    r = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, -1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
    return r, -r @ centre + np.array([0, 0, 3.0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--post", default="ik_2d_lp", choices=list(POSTS))
    p.add_argument("--images-dir", default="data/processed_640")
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()

    smpl = load_smpl_full("assets/smpl/SMPL_NEUTRAL.npz")
    faces = np.load("assets/smpl/SMPL_NEUTRAL.npz")["f"].astype(np.int64)
    calib = load_calib("assets/calib/calib.json")
    model, cfg = full_model_from_checkpoint(load_checkpoint(a.ckpt), ["device=cpu"] + POSTS[a.post] + a.set)
    model.eval()
    cfg.data.val_subjects = [a.subject]
    seq = sequences_for(cfg, "val")[0]
    d = Path(a.images_dir) / a.subject
    images = np.load(d / json.load(open(d / "meta.json"))["images"], mmap_mode="r")
    ann = dict(np.load(Path(cfg.data.processed) / a.subject / "annots.npz"))
    (s0, s1), out = max(predict_sequence(model, seq, torch.device("cpu"), cfg), key=lambda r: r[0][1] - r[0][0])
    i0 = max(s0, a.start)
    frames = list(range(i0, min(s1, i0 + int(a.seconds * 30))))
    rel = [f - s0 for f in frames]
    aa = rotation_6d_to_axis_angle(out["rot6d"][rel].reshape(-1, 6)).reshape(len(rel), 66).numpy()
    verts = mesh_world(smpl, np.concatenate([aa, np.zeros((len(rel), 6))], 1), out["betas"][rel].numpy(), seq.head_anchor[frames])

    h, w = images.shape[-2:]
    H = 2 * h                       # panel height
    fs = H / 960
    dev_w, mesh_w = int(0.55 * H), int(0.8 * H)
    head_h = int(60 * fs)
    pos = {"headset": ann["head_pos"], "left controller": ann["left_hand_pos"], "right controller": ann["right_hand_pos"]}
    legacc = {"left leg tracker": np.linalg.norm(ann["left_leg_acc"], axis=1), "right leg tracker": np.linalg.norm(ann["right_leg_acc"], axis=1)}
    amax = max(float(np.percentile(v[frames], 99)) for v in legacc.values()) + 1e-6
    anchor = seq.head_anchor[frames]
    kern = np.ones(31) / 31
    follow = np.stack([np.convolve(np.pad(anchor[:, i], 15, mode="edge"), kern, "valid") for i in range(3)], 1) - np.array([0, 0.75, 0])
    writer = None
    tmp = Path(a.out).with_suffix(".raw.mp4")
    for k, f in enumerate(frames):
        R, t = view(follow[k])
        # input: raw fisheye views
        raw = np.concatenate([cv2.cvtColor(np.ascontiguousarray(images[f, c]), cv2.COLOR_GRAY2BGR) for c in range(2)], 0)
        # input: devices
        dev = np.full((H, dev_w, 3), BG, np.uint8)
        top = int(0.62 * H)
        for name, pts in pos.items():
            trail = pts[max(frames[0], f - 45):f + 1]
            uv, _, _ = pinhole(trail, R, t, H * 0.85, dev_w / 2, top * 0.66)
            for j in range(1, len(uv)):
                cv2.line(dev, tuple(np.round(uv[j - 1]).astype(int)), tuple(np.round(uv[j]).astype(int)), DEV_COL[name], max(1, round(2 * fs)), cv2.LINE_AA)
            cv2.circle(dev, tuple(np.round(uv[-1]).astype(int)), int(7 * fs), DEV_COL[name], -1, cv2.LINE_AA)
        gy0, gy1 = top + int(40 * fs), H - int(20 * fs)
        cv2.line(dev, (int(10 * fs), gy1), (dev_w - int(10 * fs), gy1), (90, 90, 90), 1)
        lo = frames[0]
        for name, acc in legacc.items():
            seg = acc[lo:f + 1]
            xs = np.linspace(int(10 * fs), dev_w - int(10 * fs), len(frames))[:len(seg)]
            ys = gy1 - np.clip(seg / amax, 0, 1) * (gy1 - gy0)
            for j in range(1, len(seg)):
                cv2.line(dev, (int(xs[j - 1]), int(ys[j - 1])), (int(xs[j]), int(ys[j])), DEV_COL[name], max(1, round(2 * fs)), cv2.LINE_AA)
        y = int(24 * fs)
        for name, col in DEV_COL.items():
            cv2.circle(dev, (int(16 * fs), y - int(6 * fs)), int(6 * fs), col, -1, cv2.LINE_AA)
            label(dev, name, (int(28 * fs), y), 0.5 * fs)
            y += int(26 * fs)
        label(dev, "headset / controller positions", (int(10 * fs), top - int(10 * fs)), 0.5 * fs, (170, 170, 170))
        label(dev, "leg tracker acceleration", (int(10 * fs), gy0 - int(10 * fs)), 0.5 * fs, (170, 170, 170))
        # output: 3D mesh
        mesh = np.full((H, mesh_w, 3), BG, np.uint8)
        uv, dep, pc = pinhole(verts[k], R, t, H * 1.05, mesh_w / 2, H * 0.5)
        mesh = raster(mesh, uv, dep, faces, pc, ACCENT, 1.0)
        # output: projection into the fisheye views
        proj = []
        for c, cam in enumerate(("cam0", "cam1")):
            img = cv2.cvtColor(np.ascontiguousarray(images[f, c]), cv2.COLOR_GRAY2BGR)
            pcam = transform_points(seq.T_cam_world[f, c].astype(np.float64), verts[k].astype(np.float64))
            u = project_kb4(pcam, calib["cameras"][cam]["params"]) * np.array([w / 640, h / 480])
            proj.append(raster(img, u, pcam[:, 2], faces, pcam, ACCENT, 0.55, max_edge=0.08 * w))
        proj = np.concatenate(proj, 0)
        body = np.concatenate([raw, dev, np.full((H, int(12 * fs), 3), 60, np.uint8), mesh, proj], 1)
        head = np.full((head_h, body.shape[1], 3), BG, np.uint8)
        label(head, "input", (int(12 * fs), int(40 * fs)), 0.9 * fs)
        label(head, "headset fisheye cameras + 5 tracked devices", (int(120 * fs), int(40 * fs)), 0.55 * fs, (170, 170, 170))
        ox = w + dev_w + int(12 * fs)
        label(head, "output", (ox + int(12 * fs), int(40 * fs)), 0.9 * fs, ACCENT)
        label(head, "SMPL body mesh, and projected into the cameras", (ox + int(140 * fs), int(40 * fs)), 0.55 * fs, (170, 170, 170))
        frame = np.concatenate([head, body], 0)
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(Path(a.out).with_suffix(".mp4"))], check=True)
    tmp.unlink()
    print(Path(a.out).with_suffix(".mp4"), f"{len(frames)} frames, {frame.shape[1]}x{frame.shape[0]}")


if __name__ == "__main__":
    main()
