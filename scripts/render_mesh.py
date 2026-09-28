"""SMPL mesh visualisation: predicted mesh overlaid on both headset fisheye views, plus ground-truth vs predicted
meshes from a virtual third-person camera. Writes an mp4 and a gif."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from meposer.data.calibrate import load_calib
from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import predict_sequence
from meposer.geometry.fisheye import project_kb4
from meposer.geometry.rotations import rotation_6d_to_axis_angle
from meposer.geometry.se3 import transform_points
from meposer.smpl.vertices import load_smpl_full, posed_vertices

POSTS = {"raw": [], "ik_2d_lp": ["refine_wrists=true", "refine_2d=true", "post_lowpass_hz=4"]}


def mesh_world(smpl, pose72, beta, anchor_joints_head):
    verts = posed_vertices(smpl, pose72, beta, np.zeros((len(pose72), 3)), np.arange(6890))
    joints = np.einsum("jv,nvc->njc", smpl["J_regressor"], verts)
    return verts - joints[:, 15:16] + anchor_joints_head[:, None]


def shade(tri_pts3d, light=np.array([0.3, 0.8, 0.5])):
    n = np.cross(tri_pts3d[:, 1] - tri_pts3d[:, 0], tri_pts3d[:, 2] - tri_pts3d[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return 0.35 + 0.65 * np.abs(n @ (light / np.linalg.norm(light)))


def raster(canvas, uv, depth, faces, pts_cam, color, alpha=0.6, valid=None, max_edge=None):
    tri_uv = uv[faces]
    tri_d = depth[faces].mean(1)
    ok = np.isfinite(tri_uv).all((1, 2)) & (depth[faces] > 0.05).all(1)
    if max_edge is not None:
        edge = np.linalg.norm(tri_uv - np.roll(tri_uv, 1, axis=1), axis=-1).max(1)
        ok &= edge < max_edge
    if valid is not None:
        ok &= valid[faces].all(1)
    order = np.argsort(-tri_d[ok])
    idx = np.flatnonzero(ok)[order]
    s = shade(pts_cam[faces[idx]])
    layer = canvas.copy()
    for f, k in zip(idx, s):
        cv2.fillConvexPoly(layer, np.round(tri_uv[f]).astype(np.int32), tuple(float(c * k) for c in color), lineType=cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)


def pinhole(pts, R, t, f, cx, cy):
    pc = pts @ R.T + t
    return np.stack([f * pc[:, 0] / pc[:, 2] + cx, f * pc[:, 1] / pc[:, 2] + cy], 1), pc[:, 2], pc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--post", default="ik_2d_lp", choices=list(POSTS))
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--step", type=int, default=2, help="render every n-th frame")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", help="second checkpoint rendered as a third mesh (e.g. the reproduced MEPoser)")
    p.add_argument("--baseline-post", default="raw", choices=list(POSTS))
    p.add_argument("--no-images", action="store_true", help="omit the dataset fisheye frames (for redistribution)")
    p.add_argument("--png", help="also write a static grid of 4 frames")
    a = p.parse_args()
    smpl = load_smpl_full("assets/smpl/SMPL_NEUTRAL.npz")
    faces = np.load("assets/smpl/SMPL_NEUTRAL.npz")["f"].astype(np.int64)
    calib = load_calib("assets/calib/calib.json")
    model, cfg = full_model_from_checkpoint(load_checkpoint(a.ckpt), ["device=cpu"] + POSTS[a.post] + a.set)
    model.eval()
    cfg.data.val_subjects = [a.subject]
    seq = sequences_for(cfg, "val")[0]
    if seq.images is None:
        seq.images = np.load(Path("data/processed") / a.subject / seq.meta["images"], mmap_mode="r")
    (s0, s1), out = max(predict_sequence(model, seq, torch.device("cpu"), cfg), key=lambda r: r[0][1] - r[0][0])
    ann = dict(np.load(Path(cfg.data.processed) / a.subject / "annots.npz"))
    i0 = max(s0, a.start); i1 = min(s1, i0 + int(a.seconds * 30))
    frames = list(range(i0, i1, a.step))
    rel = [f - s0 for f in frames]
    aa = rotation_6d_to_axis_angle(out["rot6d"][rel].reshape(-1, 6)).reshape(len(rel), 22 * 3).numpy()
    pose_pred = np.concatenate([aa, np.zeros((len(rel), 6))], 1)
    v_pred = mesh_world(smpl, pose_pred, out["betas"][rel].numpy(), seq.head_anchor[frames])
    pose_gt = np.concatenate([ann["root_orient"][frames], ann["pose"][frames]], 1)
    v_gt = mesh_world(smpl, pose_gt, ann["shape"][frames], seq.joints_world[frames, 15])
    v_base, err_base = None, None
    if a.baseline:
        bm, bcfg = full_model_from_checkpoint(load_checkpoint(a.baseline), ["device=cpu"] + POSTS[a.baseline_post] + a.set)
        bm.eval(); bcfg.data.val_subjects = [a.subject]
        bseq = sequences_for(bcfg, "val")[0]
        (b0, b1), bout = max(predict_sequence(bm, bseq, torch.device("cpu"), bcfg), key=lambda r: r[0][1] - r[0][0])
        brel = [f - b0 for f in frames]
        baa = rotation_6d_to_axis_angle(bout["rot6d"][brel].reshape(-1, 6)).reshape(len(brel), 66).numpy()
        v_base = mesh_world(smpl, np.concatenate([baa, np.zeros((len(brel), 6))], 1), bout["betas"][brel].numpy(), seq.head_anchor[frames])
        bj = bout["joints_fk"][brel].numpy()
        err_base = np.linalg.norm(bj - bj[:, 15:16] + seq.joints_world[frames, 15][:, None] - seq.joints_world[frames], axis=-1).mean(1) * 100
    err = np.linalg.norm((out["joints_fk"][rel].numpy() - out["joints_fk"][rel].numpy()[:, 15:16] + seq.joints_world[frames, 15][:, None]) - seq.joints_world[frames], axis=-1).mean(1) * 100
    h, w = seq.images.shape[-2:]
    sx, sy = w / 640, h / 480
    size = 2 * h
    rows = []
    for k, f in enumerate(frames):
        views = []
        for c, cam in enumerate(("cam0", "cam1")):
            img = cv2.cvtColor(np.ascontiguousarray(seq.images[f, c]), cv2.COLOR_GRAY2BGR)
            pc = transform_points(seq.T_cam_world[f, c].astype(np.float64), v_pred[k].astype(np.float64))
            uv = project_kb4(pc, calib["cameras"][cam]["params"]) * np.array([sx, sy])
            img = raster(img, uv, pc[:, 2], faces, pc, (60, 140, 255), 0.45, max_edge=0.08 * w)
            cv2.putText(img, f"{a.subject} f{int(seq.frames[f])} {cam}: predicted mesh", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
            views.append(img)
        left = np.concatenate(views, 0)
        gt_j = seq.joints_world[f]
        meshes = [("ground truth", v_gt[k], (150, 150, 150), None, None)]
        if v_base is not None:
            meshes.append(("MEPoser (reproduced)", v_base[k], (230, 150, 60), err_base[k], v_base[k]))
        meshes.append(("ours", v_pred[k], (60, 140, 255), err[k], v_pred[k]))
        n = len(meshes)
        panel = np.full((size, int(size * 0.55 * n), 3), 245, np.uint8)
        centre = seq.joints_world[f, 0]
        yaw = np.deg2rad(35)
        R = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, -1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
        t = -R @ centre + np.array([0, 0, 3.0])
        J = smpl["J_regressor"]
        for m, (label, verts, color, e, _) in enumerate(meshes):
            cx = panel.shape[1] * (m + 0.5) / n
            uv, d, pc = pinhole(verts, R, t, size * 1.05, cx, size * 0.5)
            panel = raster(panel, uv, d, faces, pc, color, 1.0)
            joints = (J @ verts)[:22]
            juv, _, _ = pinhole(joints, R, t, size * 1.05, cx, size * 0.5)
            guv, _, _ = pinhole(gt_j, R, t, size * 1.05, cx, size * 0.5)
            from meposer.smpl.constants import BONES
            for jj, pp in BONES:
                if m > 0:
                    cv2.line(panel, tuple(np.round(guv[jj]).astype(int)), tuple(np.round(guv[pp]).astype(int)), (0, 170, 0), 2, cv2.LINE_AA)
                cv2.line(panel, tuple(np.round(juv[jj]).astype(int)), tuple(np.round(juv[pp]).astype(int)), (0, 0, 200) if m > 0 else (60, 60, 60), 2, cv2.LINE_AA)
            cv2.putText(panel, label, (int(cx - 70), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
            if e is not None:
                cv2.putText(panel, f"MPJPE {e:.1f} cm", (int(cx - 55), size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{a.subject} f{int(seq.frames[f])}  skeleton: GT green / prediction red", (8, size - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        rows.append(panel if a.no_images else np.concatenate([left, panel], 1))
    out_path = Path(a.out)
    wr = cv2.VideoWriter(str(out_path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30 / a.step, (rows[0].shape[1], rows[0].shape[0]))
    for r in rows:
        wr.write(r)
    wr.release()
    ims = [Image.fromarray(cv2.cvtColor(cv2.resize(r, (r.shape[1] * 3 // 4, r.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)).quantize(colors=160) for r in rows]
    ims[0].save(out_path.with_suffix(".gif"), save_all=True, append_images=ims[1:], duration=int(1000 * a.step / 30), loop=0, optimize=True)
    if a.png:
        pick = np.linspace(0, len(rows) - 1, 4).astype(int)
        cv2.imwrite(a.png, np.concatenate([rows[i] for i in pick], 0))
    print(out_path.with_suffix(".mp4"), out_path.with_suffix(".gif"), f"mean MPJPE {err.mean():.2f} cm over {len(frames)} frames")


if __name__ == "__main__":
    main()
