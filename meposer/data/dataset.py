import numpy as np
import torch
from torch.utils.data import Dataset

from ..smpl.model import SMPLJoints
from .calibrate import load_calib
from .heatmap import render_heatmaps
from .raw import IMAGE_SIZE
from .sequence import SubjectSequence


def load_sequences(cfg, subjects, load_images=True, feature_dir=None):
    calib = load_calib(cfg.data.calib)
    smpl = SMPLJoints(cfg.data.smpl)
    seqs = []
    for s in subjects:
        seq = SubjectSequence(cfg.data.processed, s, calib, smpl, list(cfg.data.imu_features), load_images=load_images,
                              fps=cfg.data.fps, synth_acc_cutoff_hz=cfg.data.get("synth_acc_cutoff_hz", 3.0), contact=dict(cfg.data.get("contact") or {}, smpl=cfg.data.smpl))
        if feature_dir is not None:
            seq.attach_features(dict(np.load(f"{feature_dir}/{s}.npz")))
        seqs.append(seq)
    return seqs


def _frame_targets(seq, i):
    return {
        "joints_local": torch.as_tensor(seq.joints_local[i]),
        "joints_world": torch.as_tensor(seq.joints_world[i]),
        "head_pos": torch.as_tensor(seq.head_pos[i]),
        "head_anchor": torch.as_tensor(seq.head_anchor[i]),
        "T_world_cam0": torch.as_tensor(seq.T_world_cam0[i]),
    }


class FrameDataset(Dataset):
    def __init__(self, sequences, frame_stride, heatmap_size, heatmap_sigma):
        self.sequences = sequences
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = heatmap_sigma
        self.items = [(si, fi) for si, s in enumerate(sequences) for fi in range(0, len(s), frame_stride)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        si, fi = self.items[idx]
        seq = self.sequences[si]
        hm, vis = zip(*(render_heatmaps(seq.p2d[fi, c], IMAGE_SIZE, self.heatmap_size, self.sigma) for c in range(2)))
        out = _frame_targets(seq, fi)
        out["images"] = torch.from_numpy(np.array(seq.images[fi]))
        out["heatmaps"] = torch.as_tensor(np.concatenate(hm))
        out["visible"] = torch.as_tensor(np.stack(vis))
        return out


class WindowDataset(Dataset):
    def __init__(self, sequences, window, stride, use_images=False, heatmap_size=None, heatmap_sigma=1.5, augment=None, feature_names=None):
        self.sequences = sequences
        self.augment = augment if augment and augment.get("enabled", False) else None
        self.feature_names = feature_names
        self.window = window
        self.use_images = use_images
        self.heatmap_size = tuple(heatmap_size) if heatmap_size else None
        self.sigma = heatmap_sigma
        self.items = [
            (si, start)
            for si, s in enumerate(sequences)
            for a, b in s.segments(min_length=window)
            for start in range(a, b - window + 1, stride)
        ]
        for si, seq in enumerate(sequences):
            n = sum(1 for i, _ in self.items if i == si)
            expected = max(1, (len(seq.frames) - window) // stride + 1)
            if n < 0.5 * expected:
                used = sum(b - a for a, b in seq.segments(min_length=window))
                print(f"WARNING: subject {seq.subject}: only {n} training windows of {window} frames (a gap-free sequence of this "
                      f"length would give {expected}); {len(seq.segments())} contiguous segments, {len(seq.frames) - used} of "
                      f"{len(seq.frames)} frames are in segments shorter than the window and are not used", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        si, start = self.items[idx]
        sample = window_sample(self.sequences[si], start, start + self.window, self.use_images, self.heatmap_size, self.sigma)
        if self.augment is not None:
            from .augment import augment_window
            sample = augment_window(sample, self.feature_names, self.augment)
        return sample


def window_sample(seq, a, b, use_images=False, heatmap_size=None, sigma=1.5):
    sl = slice(a, b)
    out = {
        "imu": torch.as_tensor(seq.imu[sl]),
        "rot6d_local": torch.as_tensor(seq.rot6d_local[sl]),
        "rot6d_global": torch.as_tensor(seq.rot6d_global[sl]),
        "betas": torch.as_tensor(seq.betas[sl]),
        "joints_world": torch.as_tensor(seq.joints_world[sl]),
        "joints_local": torch.as_tensor(seq.joints_local[sl]),
        "head_pos": torch.as_tensor(seq.head_pos[sl]),
        "head_rot": torch.as_tensor(seq.head_rot[sl]),
        "contact": torch.as_tensor(seq.contact[sl]),
        "head_anchor": torch.as_tensor(seq.head_anchor[sl]),
        "T_world_cam0": torch.as_tensor(seq.T_world_cam0[sl]),
        "frames": torch.as_tensor(seq.frames[sl]),
    }
    if seq.image_features is not None:
        out["image_feat"] = torch.as_tensor(seq.image_features["image_feat"][sl])
        out["joints_local_pred"] = torch.as_tensor(seq.image_features["joints_local"][sl])
    if use_images:
        out["images"] = torch.from_numpy(np.array(seq.images[sl]))
        hms, vis = [], []
        for i in range(a, b):
            hm, v = zip(*(render_heatmaps(seq.p2d[i, c], IMAGE_SIZE, heatmap_size, sigma) for c in range(2)))
            hms.append(np.concatenate(hm))
            vis.append(np.stack(v))
        out["heatmaps"] = torch.as_tensor(np.stack(hms))
        out["visible"] = torch.as_tensor(np.stack(vis))
    return out


def compute_feature_stats(sequences):
    x = np.concatenate([s.imu for s in sequences])
    return torch.as_tensor(x.mean(0)), torch.as_tensor(x.std(0) + 1e-6)
