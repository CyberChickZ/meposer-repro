import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..data.heatmap import heatmap_argmax, heatmap_subpixel
from .build import apply_overrides, image_branch_from_checkpoint, sequences_for
from .common import load_checkpoint, resolve_device


@torch.no_grad()
def extract_sequence(model, seq, device, batch_size=64):
    feats, joints, peaks, confs = [], [], [], []
    for s in range(0, len(seq), batch_size):
        images = torch.from_numpy(np.array(seq.images[s : s + batch_size])).to(device)
        out = model(images)
        feats.append(out["image_feat"].cpu().numpy())
        joints.append(out["joints_local"].cpu().numpy())
        hm = torch.sigmoid(out["heatmap_logits"]).cpu().numpy()
        p, c = heatmap_subpixel(hm)
        peaks.append(p.reshape(len(hm), 2, -1, 2))
        confs.append(c.reshape(len(hm), 2, -1))
    out = {"image_feat": np.concatenate(feats), "joints_local": np.concatenate(joints), "peaks": np.concatenate(peaks), "conf": np.concatenate(confs)}
    bad = [k for k in ("image_feat", "joints_local") if not np.isfinite(out[k]).all()]
    assert not bad, f"{seq.subject}: non-finite image-branch outputs in {bad} (re-run with --set device=cpu)"
    return out


def run(ckpt_path, out_dir, overrides=(), cfg=None, subjects=None):
    ckpt = load_checkpoint(ckpt_path)
    model, ckpt_cfg = image_branch_from_checkpoint(ckpt)
    cfg = apply_overrides((cfg or ckpt_cfg).to_dict(), overrides)
    cfg.model.image_features = None
    if subjects is not None:
        cfg.data.train_subjects, cfg.data.val_subjects = list(subjects), []
    device = resolve_device(cfg.device)
    torch.set_num_threads(cfg.get("threads", 10))
    model = model.to(device).eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sequences_for(cfg, "train") + sequences_for(cfg, "val")
    for seq in tqdm(seqs, desc="subjects"):
        np.savez(out_dir / f"{seq.subject}.npz", **extract_sequence(model, seq, device))
    meta_path = out_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta.update({"checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "val": ckpt.get("val"), "heatmap_size": ckpt["heatmap_size"]})
    meta["subjects"] = sorted(set(meta.get("subjects", [])) | {s.subject for s in seqs})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir
