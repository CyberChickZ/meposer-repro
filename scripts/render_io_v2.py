"""Input -> output video for the project page, no comparison to other methods.
input:  both raw headset fisheye images with the 2D joints the network detects in them (heatmap peaks; legs from the
        adaptive leg crop), confident joints only
output: one 3D view with the ground-truth joints, the predicted joints, the predicted SMPL mesh (semi-transparent, its
        joints are the predicted joints) and the headset / controller positions; below it the jitter of the predicted
        and the ground-truth joints over time."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import predict_sequence
from meposer.geometry.rotations import rotation_6d_to_axis_angle
from meposer.smpl.constants import BONES
from meposer.smpl.vertices import load_smpl_full
from render_mesh import POSTS, mesh_world, pinhole, raster

BG, FG, DIM = (24, 24, 24), (235, 235, 235), (160, 160, 160)
GT_C, PRED_C, MESH_C, DEV_C, KP_C = (90, 210, 90), (40, 150, 255), (235, 205, 170), (255, 255, 255), (60, 230, 255)


def text(img, s, org, scale, col=FG):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, max(1, round(scale * 2)), cv2.LINE_AA)


def view(centre, yaw_deg=35):
    yaw = np.deg2rad(yaw_deg)
    r = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, -1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
    return r, -r @ centre + np.array([0, 0, 3.0])


def jitter(j, fps=30):
    """jitter per frame as in the HMD-Poser metric code: mean over joints of |third derivative of position|, m/s^3"""
    out = np.zeros(len(j))
    jerk = np.linalg.norm(j[3:] - 3 * j[2:-1] + 3 * j[1:-2] - j[:-3], axis=-1).mean(-1) * fps ** 3
    out[3:] = jerk
    out[:3] = jerk[0]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--post", default="ik_2d_lp", choices=list(POSTS))
    p.add_argument("--peaks", required=True, help="2D joint peaks npz directory (leg-crop output, full-image pixels)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--images-dir", default="data/processed_640")
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()

    smpl = load_smpl_full("assets/smpl/SMPL_NEUTRAL.npz")
    faces = np.load("assets/smpl/SMPL_NEUTRAL.npz")["f"].astype(np.int64)
    model, cfg = full_model_from_checkpoint(load_checkpoint(a.ckpt), ["device=cpu"] + POSTS[a.post] + a.set)
    model.eval()
    cfg.data.val_subjects = [a.subject]
    seq = sequences_for(cfg, "val")[0]
    d = Path(a.images_dir) / a.subject
    images = np.load(d / json.load(open(d / "meta.json"))["images"], mmap_mode="r")
    pk = np.load(Path(a.peaks) / f"{a.subject}.npz")
    peaks, conf = pk["peaks"], pk["conf"]
    ann = dict(np.load(Path(cfg.data.processed) / a.subject / "annots.npz"))
    (s0, s1), out = max(predict_sequence(model, seq, torch.device("cpu"), cfg), key=lambda r: r[0][1] - r[0][0])
    i0 = max(s0, a.start)
    frames = list(range(i0, min(s1, i0 + int(a.seconds * 30))))
    rel = [f - s0 for f in frames]
    aa = rotation_6d_to_axis_angle(out["rot6d"][rel].reshape(-1, 6)).reshape(len(rel), 66).numpy()
    verts = mesh_world(smpl, np.concatenate([aa, np.zeros((len(rel), 6))], 1), out["betas"][rel].numpy(), seq.head_anchor[frames])
    fk_c = out["joints_fk"][rel].numpy()
    pred_j = fk_c - fk_c[:, 15:16] + seq.head_anchor[frames][:, None]   # the model's output joints, anchored like the mesh
    gt_j = seq.joints_world[frames]
    # jitter over the whole segment (so the clip edges are not affected), then the clip
    all_rel = np.arange(s1 - s0)
    fk = out["joints_fk"].numpy()
    gt_seg = seq.joints_world[s0:s1]
    jit_pred = jitter(fk - fk[:, 15:16] + gt_seg[:, 15:16])[rel]      # evaluation protocol: prediction placed at the GT head
    jit_gt = jitter(gt_seg)[rel]

    h, w = images.shape[-2:]
    H = 2 * h
    fs = H / 960
    view_w, view_h = int(0.95 * H), int(0.74 * H)
    plot_h = H - view_h
    head_h = int(60 * fs)
    dev = {"headset": ann["head_pos"], "left controller": ann["left_hand_pos"], "right controller": ann["right_hand_pos"]}
    kern = np.ones(31) / 31
    follow = np.stack([np.convolve(np.pad(seq.head_anchor[frames][:, i], 15, mode="edge"), kern, "valid") for i in range(3)], 1) - np.array([0, 0.75, 0])
    ymax = float(np.percentile(np.concatenate([jit_pred, jit_gt]), 95)) * 1.6 + 1e-6    # single-frame spikes are clipped
    th = max(1, round(2 * fs))
    tmp = Path(a.out).with_suffix(".raw.mp4")
    writer = None
    for k, f in enumerate(frames):
        # input: fisheye + detected 2D joints
        views = []
        for c in range(2):
            img = cv2.cvtColor(np.ascontiguousarray(images[f, c]), cv2.COLOR_GRAY2BGR)
            uv = peaks[f, c] * np.array([w / 640, h / 480])
            ok = conf[f, c] > a.conf
            for jj, pp in BONES:
                if ok[jj] and ok[pp] and np.linalg.norm(uv[jj] - uv[pp]) < 0.35 * w:
                    cv2.line(img, tuple(np.round(uv[jj]).astype(int)), tuple(np.round(uv[pp]).astype(int)), KP_C, th, cv2.LINE_AA)
            for j in np.flatnonzero(ok):
                cv2.circle(img, tuple(np.round(uv[j]).astype(int)), int(5 * fs), KP_C, -1, cv2.LINE_AA)
            views.append(img)
        left = np.concatenate(views, 0)
        text(left, "2D joints detected in the images", (int(10 * fs), int(26 * fs)), 0.55 * fs, KP_C)
        # output: 3D view
        R, t = view(follow[k])
        pan = np.full((view_h, view_w, 3), BG, np.uint8)
        cy = view_h * 0.52
        uvm, dm, pcm = pinhole(verts[k], R, t, view_h * 1.3, view_w / 2, cy)
        pan = raster(pan, uvm, dm, faces, pcm, MESH_C, 0.35)
        for pts, col in ((gt_j[k], GT_C), (pred_j[k], PRED_C)):
            uv, _, _ = pinhole(pts, R, t, view_h * 1.3, view_w / 2, cy)
            for jj, pp in BONES:
                cv2.line(pan, tuple(np.round(uv[jj]).astype(int)), tuple(np.round(uv[pp]).astype(int)), col, th + 1, cv2.LINE_AA)
            for u in uv:
                cv2.circle(pan, tuple(np.round(u).astype(int)), int(4 * fs), col, -1, cv2.LINE_AA)
        for name, pos in dev.items():
            uv, _, _ = pinhole(pos[f][None], R, t, view_h * 1.3, view_w / 2, cy)
            cv2.circle(pan, tuple(np.round(uv[0]).astype(int)), int(9 * fs), DEV_C, 2 * th, cv2.LINE_AA)
        y = int(28 * fs)
        for label, col, kind in (("ground-truth joints", GT_C, "line"), ("predicted joints", PRED_C, "line"),
                                 ("predicted body mesh", MESH_C, "box"), ("headset and controllers", DEV_C, "ring")):
            x0 = int(14 * fs)
            if kind == "line":
                cv2.line(pan, (x0, y - int(6 * fs)), (x0 + int(26 * fs), y - int(6 * fs)), col, th + 1, cv2.LINE_AA)
            elif kind == "box":
                ov = pan.copy()
                cv2.rectangle(ov, (x0, y - int(14 * fs)), (x0 + int(26 * fs), y + int(2 * fs)), col, -1)
                pan = cv2.addWeighted(ov, 0.35, pan, 0.65, 0)
            else:
                cv2.circle(pan, (x0 + int(13 * fs), y - int(6 * fs)), int(8 * fs), col, th, cv2.LINE_AA)
            text(pan, label, (x0 + int(38 * fs), y), 0.52 * fs)
            y += int(30 * fs)
        # output: jitter plot
        plot = np.full((plot_h, view_w, 3), (32, 32, 32), np.uint8)
        x0, x1 = int(70 * fs), view_w - int(20 * fs)
        y0, y1 = int(36 * fs), plot_h - int(18 * fs)
        cv2.line(plot, (x0, y1), (x1, y1), (90, 90, 90), 1)
        cv2.line(plot, (x0, y0), (x0, y1), (90, 90, 90), 1)
        text(plot, f"{ymax:.0f}", (int(10 * fs), y0 + int(8 * fs)), 0.42 * fs, DIM)
        text(plot, "0", (int(40 * fs), y1), 0.42 * fs, DIM)
        text(plot, "jitter (m/s^3)", (x0 + int(8 * fs), int(24 * fs)), 0.5 * fs, DIM)
        xs = np.linspace(x0, x1, len(frames))
        for series, col, name, dx in ((jit_gt, GT_C, "ground truth", 0), (jit_pred, PRED_C, "prediction", 1)):
            ys = y1 - np.clip(series / ymax, 0, 1) * (y1 - y0)
            pts = np.stack([xs[:k + 1], ys[:k + 1]], 1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(plot, [pts], False, col, th, cv2.LINE_AA)
            lx = x1 - int((330 - 160 * dx) * fs)
            cv2.line(plot, (lx, int(18 * fs)), (lx + int(22 * fs), int(18 * fs)), col, th, cv2.LINE_AA)
            text(plot, name, (lx + int(28 * fs), int(24 * fs)), 0.45 * fs, DIM)
        cv2.line(plot, (int(xs[k]), y0), (int(xs[k]), y1), (110, 110, 110), 1)
        right = np.concatenate([pan, plot], 0)
        body = np.concatenate([left, np.full((H, int(12 * fs), 3), 60, np.uint8), right], 1)
        head = np.full((head_h, body.shape[1], 3), BG, np.uint8)
        text(head, "input", (int(12 * fs), int(42 * fs)), 0.95 * fs)
        text(head, "output", (w + int(24 * fs), int(42 * fs)), 0.95 * fs, PRED_C)
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
