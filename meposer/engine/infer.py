import torch

from ..data.dataset import window_sample
from ..metrics.evaluate import aggregate, sequence_metrics
from ..models.meposer import anchor_joints
from ..smpl.constants import HEAD
from .common import heatmap_size, to_device

COLLECT = ["rot6d", "betas", "joints_fk", "rot6d_global", "joints_local", "joints_global", "heatmap_logits", "contact_logits"]


@torch.no_grad()
def run_segment(model, seq, a, b, device, cfg, chunk):
    model.eval()
    use_images = model.use_image and not model.cached_image_features
    hs = heatmap_size(cfg)
    states = None
    parts = {k: [] for k in COLLECT}
    for s in range(a, b, chunk):
        e = min(b, s + chunk)
        batch = window_sample(seq, s, e, use_images=use_images, heatmap_size=hs, sigma=cfg.data.heatmap_sigma)
        batch = to_device({k: v[None] for k, v in batch.items()}, device)
        out = model(batch, states)
        states = out["states"]
        for k in COLLECT:
            if k in out:
                parts[k].append(out[k][0].cpu())
    out = {k: torch.cat(v) for k, v in parts.items() if v}
    gt_joints = torch.as_tensor(seq.joints_world[a:b])
    out["joints_world_gt_anchored"] = anchor_joints(out["joints_fk"], gt_joints[:, HEAD])
    out["joints_world"] = anchor_joints(out["joints_fk"], torch.as_tensor(seq.head_anchor[a:b]))
    return out


@torch.no_grad()
def run_sequence(model, seq, device, cfg, chunk=None, min_segment=4):
    chunk = chunk or (32 if (model.use_image and not model.cached_image_features) else 512)
    results = []
    for a, b in seq.segments(min_length=min_segment):
        results.append(((a, b), run_segment(model, seq, a, b, device, cfg, chunk)))
    return results


class PostProcessor:
    """Optional test-time post-processing: zero-phase low-pass of the rotations and controller-driven wrist IK."""

    def __init__(self, model, cfg):
        import copy
        self.cfg = cfg
        self.lowpass = cfg.get("post_lowpass_hz")
        self.refine = cfg.get("refine_wrists", False)
        self.fit2d = cfg.get("refine_2d", False)
        self.contact_fit = cfg.get("refine_contact", False)
        self.smpl = copy.deepcopy(model.smpl).cpu() if (self.lowpass or self.refine or self.fit2d or self.contact_fit) else None
        if self.fit2d:
            from ..data.calibrate import load_calib
            calib = load_calib(cfg.data.calib)
            self.cam_params = [calib["cameras"][c]["params"] for c in ("cam0", "cam1")]
            self.cell = 640.0 / (cfg.data.image_size[0] / cfg.data.heatmap_stride)
        if self.refine or self.fit2d:
            from .refine import controller_wrist_offsets
            self.offsets = controller_wrist_offsets(cfg.data.processed, list(cfg.data.train_subjects), self.smpl)

    def __call__(self, seq, a, b, out):
        from ..geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix
        if self.fit2d and seq.image_features is not None:
            from .refine import refine_body, wrist_targets
            tg = wrist_targets(self.cfg.data.processed, seq.subject, self.offsets, a, b) if self.refine else None
            src, cell = seq.image_features, self.cell
            if self.cfg.get("refine_2d_features"):
                import json
                import numpy as np
                from pathlib import Path
                d = Path(self.cfg.refine_2d_features)
                src = np.load(d / f"{seq.subject}.npz")
                cell = 640.0 / json.load(open(d / "meta.json"))["heatmap_size"][0]
            peaks = torch.as_tensor(src["peaks"][a:b] * cell, dtype=torch.float32)
            conf = torch.as_tensor(src["conf"][a:b], dtype=torch.float32)
            tcw = torch.as_tensor(seq.T_cam_world[a:b], dtype=torch.float32)
            out["rot6d"], out["joints_fk"] = refine_body(self.smpl, out["rot6d"], out["betas"], torch.as_tensor(seq.head_anchor[a:b]), tg,
                                                         peaks, conf, tcw, self.cam_params, w2d=self.cfg.get("refine_2d_weight", 1.0),
                                                         prior=self.cfg.get("refine_prior", 0.1), iters=self.cfg.get("refine_iters", 80))
        elif self.refine:
            from .refine import refine_arms, wrist_targets
            tg = wrist_targets(self.cfg.data.processed, seq.subject, self.offsets, a, b)
            out["rot6d"], out["joints_fk"] = refine_arms(self.smpl, out["rot6d"], out["betas"], torch.as_tensor(seq.head_anchor[a:b]), tg)
        if self.lowpass and b - a > 12:
            from scipy.signal import butter, filtfilt
            bb, aa = butter(2, self.lowpass / (self.cfg.data.fps / 2))
            r = torch.as_tensor(filtfilt(bb, aa, out["rot6d"].numpy(), axis=0).copy(), dtype=torch.float32)
            out["rot6d"] = matrix_to_rotation_6d(rotation_6d_to_matrix(r))
            with torch.no_grad():
                out["joints_fk"], _ = self.smpl(rotation_6d_to_matrix(out["rot6d"]), out["betas"])
        if self.contact_fit and out.get("contact_logits") is not None and b - a > 2:
            from .refine import refine_contact
            out["rot6d"], out["joints_fk"] = refine_contact(self.smpl, out["rot6d"], out["betas"], torch.as_tensor(seq.head_anchor[a:b]),
                                                            out["contact_logits"], threshold=self.cfg.get("refine_contact_p", 0.9))
        return out


def predict_sequence(model, seq, device, cfg, post=None):
    post = post or PostProcessor(model, cfg)
    return [((a, b), post(seq, a, b, out)) for (a, b), out in run_sequence(model, seq, device, cfg)]


def evaluate_sequences(model, seqs, device, cfg):
    post = PostProcessor(model, cfg)
    per_seq, names = [], []
    for seq in seqs:
        seg_metrics = []
        for (a, b), out in predict_sequence(model, seq, device, cfg, post):
            m = sequence_metrics(out["rot6d"], out["joints_fk"], torch.as_tensor(seq.rot6d_local[a:b]), torch.as_tensor(seq.joints_world[a:b]), cfg.data.fps,
                                 contact=torch.as_tensor(seq.contact[a:b]), contact_logits=out.get("contact_logits"))
            seg_metrics.append(m)
        agg = aggregate(seg_metrics)["frame_weighted"]
        agg["num_frames"] = sum(m["num_frames"] for m in seg_metrics)
        agg["num_segments"] = len(seg_metrics)
        per_seq.append(agg)
        names.append(seq.subject)
    return names, per_seq, aggregate(per_seq)
