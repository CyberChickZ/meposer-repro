import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import predict_sequence
from meposer.engine.video import _image_panel, _skeleton_panel
from meposer.models.meposer import anchor_joints
from meposer.smpl.constants import BONES, HEAD
from meposer.data.calibrate import load_calib
from meposer.data.raw import CAMERAS, IMAGE_SIZE
from meposer.geometry.fisheye import project_kb4
from meposer.geometry.se3 import transform_points

COLORS = [(80, 80, 255), (255, 170, 0), (255, 0, 255), (0, 230, 230)]


def predict(ckpt, subject, post, extra=()):
    model, cfg = full_model_from_checkpoint(load_checkpoint(ckpt), ["device=cpu"] + post + list(extra))
    model.eval()
    cfg.data.val_subjects = [subject]
    seq = sequences_for(cfg, "val")[0]
    (a, b), out = max(predict_sequence(model, seq, torch.device("cpu"), cfg), key=lambda r: r[0][1] - r[0][0])
    joints = anchor_joints(out["joints_fk"], torch.as_tensor(seq.joints_world[a:b, HEAD])).numpy()
    return seq, a, b, joints


def fisheye_panel(seq, fi, c, joints_list, calib, label):
    im = cv2.cvtColor(np.ascontiguousarray(seq.images[fi, c]), cv2.COLOR_GRAY2BGR)
    sx, sy = im.shape[1] / IMAGE_SIZE[0], im.shape[0] / IMAGE_SIZE[1]
    params = calib["cameras"][CAMERAS[c]]["params"]
    for joints, color in joints_list:
        pc = transform_points(seq.T_cam_world[fi, c].astype(np.float64), joints.astype(np.float64))
        uv = project_kb4(pc, params) * np.array([sx, sy])
        ok = pc[:, 2] > 0.02
        for j, p in BONES:
            if ok[j] and ok[p]:
                cv2.line(im, tuple(uv[j].astype(int)), tuple(uv[p].astype(int)), color, max(2, im.shape[0] // 120), cv2.LINE_AA)
    f = im.shape[0] / 240
    cv2.putText(im, label, (int(5 * f), int(14 * f)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f, (255, 255, 0), max(1, round(f)), cv2.LINE_AA)
    return im


def main():
    p = argparse.ArgumentParser(description="side-by-side video: several models vs GT on one held-out subject")
    p.add_argument("--subject", required=True)
    p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "CKPT", "POST"), required=True,
                   help="POST: raw | ik_lp (test-time wrist IK + 4 Hz low-pass)")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--no-images", action="store_true", help="3D panels only, no dataset fisheye frames (for redistribution)")
    p.add_argument("--gif", help="also write a downscaled gif (every 2nd frame)")
    p.add_argument("--gif-scale", type=float, default=0.6)
    p.add_argument("--images-dir", default="data/processed_640", help="prepared images for the fisheye panels (falls back to data/processed)")
    a = p.parse_args()
    preds = []
    for name, ck, post in a.model:
        extra = {"ik_lp": ["refine_wrists=true", "post_lowpass_hz=4"], "ik_2d_lp": ["refine_wrists=true", "refine_2d=true", "post_lowpass_hz=4"],
                 "full": ["refine_wrists=true", "refine_2d=true", "post_lowpass_hz=4", "refine_contact=true"]}.get(post, [])
        seq, s0, s1, j = predict(ck, a.subject, extra, a.set)
        preds.append((name, j))
    gt = seq.joints_world[s0:s1]
    calib = load_calib("assets/calib/calib.json")
    if not a.no_images:
        import json
        d = Path(a.images_dir) / a.subject
        d = d if (d / "meta.json").exists() else Path("data/processed") / a.subject
        seq.images = np.load(d / json.load(open(d / "meta.json"))["images"], mmap_mode="r")
    errs = [np.linalg.norm(j - gt, axis=-1).mean(-1) * 100 for _, j in preds]
    start = 0 if a.start is None else max(0, a.start - s0)
    end = min(s1 - s0, start + int(a.seconds * 30))
    h = 240 if a.no_images else seq.images.shape[-2]
    size = 2 * h
    size3d = int(0.9 * h)
    writer, gif = None, []
    for i in range(start, end):
        fi = s0 + i
        world = [(gt[i], (0, 200, 0)), (preds[-1][1][i], (0, 0, 255))]
        panels = [_skeleton_panel(gt[i], j[i], size, 35, e[i], name, size3d) for (name, j), e in zip(preds, errs)]
        if not a.no_images:
            panels.insert(0, np.concatenate([fisheye_panel(seq, fi, c, world, calib, f"{a.subject} f{int(seq.frames[fi])} cam{c}") for c in range(2)], 0))
        frame = np.concatenate(panels, 1)
        k2 = h / 240
        t2 = max(1, round(k2))
        strip = np.full((int(90 * k2), frame.shape[1], 3), 20, np.uint8)
        lo = max(0, i - 300)
        ymax = max(10.0, max(float(np.percentile(e, 99)) for e in errs))
        for (name, _), e, col in zip(preds, errs, COLORS):
            seg = e[lo : i + 1]
            xs = np.linspace(0, frame.shape[1] - 1, len(seg)).astype(int)
            ys = ((86 - np.clip(seg, 0, ymax) / ymax * 50) * k2).astype(int)
            for k in range(1, len(seg)):
                cv2.line(strip, (xs[k - 1], ys[k - 1]), (xs[k], ys[k]), col, t2, cv2.LINE_AA)
        x = int(6 * k2)
        cv2.putText(strip, "per-frame MPJPE (cm), clip mean in brackets. GT green, prediction red." + ("" if a.no_images else " Fisheye: last model."),
                    (x, int(14 * k2)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * k2, (255, 255, 255), t2, cv2.LINE_AA)
        for (name, _), e, col in zip(preds, errs, COLORS):
            t = f"{name}: {e[i]:.1f} ({e[start:end].mean():.2f})"
            cv2.putText(strip, t, (x, int(32 * k2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k2, col, t2, cv2.LINE_AA)
            x += int(12 * k2) + cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k2, t2)[0][0]
        frame = np.concatenate([frame, strip], 0)
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if a.gif and (i - start) % 2 == 0:
            gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(frame, None, fx=a.gif_scale, fy=a.gif_scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    writer.release()
    if gif:
        gif[0].save(a.gif, save_all=True, append_images=gif[1:], duration=67, loop=0, optimize=True)
    print(a.out, {n: round(float(e[start:end].mean()), 2) for (n, _), e in zip(preds, errs)})


if __name__ == "__main__":
    main()
