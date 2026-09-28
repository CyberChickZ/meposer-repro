from pathlib import Path

import cv2
import numpy as np

from ..data.raw import IMAGE_SIZE
from ..smpl.constants import BONES, HEAD

GREEN, RED, WHITE, GREY = (0, 200, 0), (0, 0, 255), (255, 255, 255), (90, 90, 90)


def _project(points, azimuth_deg, size, scale):
    a = np.deg2rad(azimuth_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = points @ rot.T
    u = size / 2 + p[:, 0] * scale
    v = size * 0.22 - p[:, 1] * scale
    return np.stack([u, v], 1).astype(int)


def _skeleton_panel(gt, pred, size, azimuth, err_cm, text):
    panel = np.full((size, size, 3), 30, np.uint8)
    for x in range(0, size, size // 8):
        cv2.line(panel, (x, 0), (x, size), GREY, 1)
        cv2.line(panel, (0, x), (size, x), GREY, 1)
    scale = size / 2.6
    for pts, color, thick in [(gt, GREEN, 2), (pred, RED, 2)]:
        uv = _project(pts - gt[HEAD], azimuth, size, scale)
        for j, p in BONES:
            cv2.line(panel, tuple(uv[j]), tuple(uv[p]), color, thick, cv2.LINE_AA)
        for u, v in uv:
            cv2.circle(panel, (u, v), 3, color, -1, cv2.LINE_AA)
    cv2.putText(panel, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"MPJPE {err_cm:.1f} cm   GT green / pred red", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return panel


def overlay_points(image_shape, gt_p2d, peaks, conf, hs, conf_thresh=0.3):
    h, w = image_shape[:2]
    vis = (gt_p2d[:, 0] >= 0) & (gt_p2d[:, 0] < IMAGE_SIZE[0]) & (gt_p2d[:, 1] >= 0) & (gt_p2d[:, 1] < IMAGE_SIZE[1])
    gt = gt_p2d[vis] * np.array([w / IMAGE_SIZE[0], h / IMAGE_SIZE[1]])
    pred = None
    if peaks is not None:
        keep = vis if conf is None else conf > conf_thresh
        pred = peaks[keep] * np.array([w / hs[0], h / hs[1]])
    return gt, pred


def _image_panel(image, gt_p2d, peaks, conf, hs, label, conf_thresh=0.3):
    im = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_GRAY2BGR)
    gt, pred = overlay_points(im.shape, gt_p2d, peaks, conf, hs, conf_thresh)
    for u, v in gt:
        cv2.circle(im, (int(u), int(v)), 3, GREEN, -1, cv2.LINE_AA)
    if pred is not None:
        for u, v in pred:
            cv2.drawMarker(im, (int(u), int(v)), RED, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    cv2.putText(im, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return im


def _curve_strip(err, i, width, height=70, window=300):
    strip = np.full((height, width, 3), 20, np.uint8)
    lo = max(0, i - window)
    seg = err[lo : i + 1]
    ymax = max(10.0, float(np.percentile(err, 99)))
    xs = np.linspace(0, width - 1, len(seg)).astype(int)
    ys = (height - 8 - (np.clip(seg, 0, ymax) / ymax) * (height - 16)).astype(int)
    for k in range(1, len(seg)):
        cv2.line(strip, (xs[k - 1], ys[k - 1]), (xs[k], ys[k]), (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(strip, f"per-frame MPJPE, last {window} frames (0-{ymax:.0f} cm); mean so far {err[: i + 1].mean():.2f} cm", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    return strip


def render_prediction_video(seq, pred, a, b, hs, out_path, start=None, seconds=20, fps=30, azimuth=35):
    gt = seq.joints_world[a:b]
    pj = pred["joints_world_gt_anchored"].numpy()
    err = np.linalg.norm(pj - gt, axis=-1).mean(-1) * 100
    n = b - a
    start = 0 if start is None else max(0, min(n - 1, start - a))
    end = min(n, start + int(seconds * fps))
    h, w = seq.images.shape[-2:]
    size = 2 * h
    writer = None
    for i in range(start, end):
        fi = a + i
        peaks = conf = None
        if seq.image_features is not None:
            peaks, conf = seq.image_features["peaks"][fi], seq.image_features["conf"][fi]
        left = np.concatenate([_image_panel(seq.images[fi, c], seq.p2d[fi, c], None if peaks is None else peaks[c], None if conf is None else conf[c], hs, f"{seq.subject} f{int(seq.frames[fi])} cam{c}") for c in range(2)], 0)
        panel = _skeleton_panel(gt[i], pj[i], size, azimuth, err[i], f"3D joints, head-anchored, view azimuth {azimuth} deg")
        frame = np.concatenate([left, panel], 1)
        frame = np.concatenate([frame, _curve_strip(err, i, frame.shape[1])], 0)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    return out_path, err[start:end].mean()
