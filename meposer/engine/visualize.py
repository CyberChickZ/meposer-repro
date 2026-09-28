from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..data.heatmap import heatmap_argmax
from ..smpl.constants import BONES, HEAD
from .build import full_model_from_checkpoint, sequences_for
from .common import load_checkpoint, resolve_device
from .infer import run_sequence
from .video import overlay_points


def _skeleton(ax, joints, color, label):
    x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
    for j, p in BONES:
        ax.plot([x[j], x[p]], [z[j], z[p]], [y[j], y[p]], color=color, linewidth=2)
    ax.scatter(x, z, y, color=color, s=8, label=label)


def _image_panel(ax, image, gt_p2d, pred_peaks, heatmap_size, title, pred_conf=None, conf_thresh=0.3):
    ax.imshow(image, cmap="gray")
    gt, pred = overlay_points(image.shape, gt_p2d, pred_peaks, pred_conf, heatmap_size, conf_thresh)
    ax.scatter(gt[:, 0], gt[:, 1], s=14, c="lime", marker="o", label="GT 2D")
    if pred is not None:
        ax.scatter(pred[:, 0], pred[:, 1], s=14, c="red", marker="x", label="pred peak")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def run(ckpt_path, subject, frames=None, out="reports/figures", overrides=(), video_seconds=0, video_start=None):
    ckpt = load_checkpoint(ckpt_path)
    model, cfg = full_model_from_checkpoint(ckpt, overrides)
    device = resolve_device(cfg.device)
    model = model.to(device).eval()
    cfg.data.val_subjects = [subject]
    seq = sequences_for(cfg, "val")[0]
    if seq.images is None:
        seq.images = np.load(Path(cfg.data.processed) / subject / seq.meta["images"], mmap_mode="r")
    (a, b), pred = run_sequence(model, seq, device, cfg)[0]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gt = torch.as_tensor(seq.joints_world[a:b])
    err = (pred["joints_world_gt_anchored"] - gt).norm(dim=-1).mean(-1).numpy() * 100
    frames = frames or list(np.linspace(a, b - 1, 6).astype(int))
    hs = tuple(ckpt["heatmap_size"])
    fig, axes = plt.subplots(len(frames), 3, figsize=(12, 3.4 * len(frames)), subplot_kw=None)
    for r, fi in enumerate(frames):
        peaks, conf = None, None
        if seq.image_features is not None:
            peaks, conf = seq.image_features["peaks"][fi], seq.image_features["conf"][fi]
        elif "heatmap_logits" in pred:
            hm = torch.sigmoid(pred["heatmap_logits"][fi - a]).numpy().reshape(2, -1, hs[1], hs[0])
            pk = [heatmap_argmax(hm[c]) for c in range(2)]
            peaks, conf = np.stack([p for p, _ in pk]), np.stack([c for _, c in pk])
        for c in range(2):
            _image_panel(axes[r, c], seq.images[fi, c], seq.p2d[fi, c], None if peaks is None else peaks[c], hs,
                         f"{subject} frame {int(seq.frames[fi])} cam{c}", None if conf is None else conf[c])
        axes[r, 2].remove()
        ax = fig.add_subplot(len(frames), 3, r * 3 + 3, projection="3d")
        g = seq.joints_world[fi] - seq.joints_world[fi, HEAD]
        p = pred["joints_world_gt_anchored"][fi - a].numpy() - seq.joints_world[fi, HEAD]
        _skeleton(ax, g, "green", "GT")
        _skeleton(ax, p, "red", "pred")
        ax.set_title(f"MPJPE {err[fi - a]:.1f} cm", fontsize=8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1.8, 0.3)
        ax.view_init(elev=15, azim=-60)
        if r == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"qualitative_{subject}.png", dpi=110)
    plt.close(fig)
    if video_seconds:
        from .video import render_prediction_video
        path, mean_err = render_prediction_video(seq, pred, a, b, hs, out / f"prediction_{subject}.mp4", video_start, video_seconds)
        print(f"saved {path} ({video_seconds} s, mean MPJPE in clip {mean_err:.2f} cm)")
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(seq.frames[a:b], err, linewidth=0.8)
    ax.set_xlabel("frame"); ax.set_ylabel("MPJPE (cm)"); ax.set_title(f"{subject}: per-frame MPJPE, mean {err.mean():.2f} cm")
    fig.tight_layout()
    fig.savefig(out / f"mpjpe_curve_{subject}.png", dpi=110)
    plt.close(fig)
    print(f"saved {out / f'qualitative_{subject}.png'} and {out / f'mpjpe_curve_{subject}.png'}")
