import math

import numpy as np
import torch

from ..geometry.rotations import rotation_6d_to_axis_angle
from ..smpl.constants import FOOT_JOINTS, HEAD, LOWER_INDEX, NUM_BODY_JOINTS, UPPER_INDEX
from .hmdposer import get_metric_function
from .procrustes import pa_mpjpe

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0
PRED_METRICS = ["mpjre", "mpjpe", "mpjve", "handpe", "upperpe", "lowerpe", "rootpe", "pred_jitter"]
GT_METRICS = ["gt_jitter"]
COEFFS = {
    "mpjre": RADIANS_TO_DEGREES, "mpjpe": METERS_TO_CENTIMETERS, "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS, "upperpe": METERS_TO_CENTIMETERS, "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS, "pred_jitter": 1.0, "gt_jitter": 1.0, "pa_mpjpe": METERS_TO_CENTIMETERS,
}
REPORT_ORDER = ["mpjre", "mpjpe", "pa_mpjpe", "upperpe", "lowerpe", "rootpe", "handpe", "mpjve", "pred_jitter", "gt_jitter", "foot_skate", "gt_foot_skate", "contact_acc", "contact_f1"]


@torch.no_grad()
def foot_skate_cm_s(joints, contact, fps):
    feet = joints[:, FOOT_JOINTS]
    vel = (feet[1:] - feet[:-1]).norm(dim=-1).mean(-1) * fps
    planted = contact[1:]
    return float((vel * planted).sum() / planted.sum().clamp_min(1.0)) * METERS_TO_CENTIMETERS


def contact_scores(logits, contact):
    pred = (logits > 0).float()
    tp = (pred * contact).sum()
    precision = tp / pred.sum().clamp_min(1.0)
    recall = tp / contact.sum().clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return float((pred == contact).float().mean()), float(f1)


def sequence_metrics(pred_rot6d, pred_joints, gt_rot6d, gt_joints, fps, contact=None, contact_logits=None):
    n = pred_joints.shape[0]
    pred_aa = rotation_6d_to_axis_angle(pred_rot6d.reshape(-1, 6)).reshape(n, NUM_BODY_JOINTS * 3)
    gt_aa = rotation_6d_to_axis_angle(gt_rot6d.reshape(-1, 6)).reshape(n, NUM_BODY_JOINTS * 3)
    pred_pos = pred_joints - pred_joints[:, HEAD : HEAD + 1] + gt_joints[:, HEAD : HEAD + 1]
    args = (pred_pos, pred_aa[:, 3:66], pred_aa[:, :3], gt_joints, gt_aa[:, 3:66], gt_aa[:, :3], UPPER_INDEX, LOWER_INDEX, fps)
    out = {m: float(get_metric_function(m)(*args)) * COEFFS[m] for m in PRED_METRICS + GT_METRICS}
    out["pa_mpjpe"] = pa_mpjpe(pred_pos.cpu().numpy().astype(np.float64), gt_joints.cpu().numpy().astype(np.float64)) * COEFFS["pa_mpjpe"]
    if contact is not None:
        out["foot_skate"] = foot_skate_cm_s(pred_pos, contact, fps)
        out["gt_foot_skate"] = foot_skate_cm_s(gt_joints, contact, fps)
        if contact_logits is not None:
            out["contact_acc"], out["contact_f1"] = contact_scores(contact_logits, contact)
    out["num_frames"] = n
    return out


def aggregate(per_sequence):
    keys = [k for k in per_sequence[0] if k != "num_frames"]
    total = sum(s["num_frames"] for s in per_sequence)
    mean = {k: float(np.mean([s[k] for s in per_sequence])) for k in keys}
    weighted = {k: float(sum(s[k] * s["num_frames"] for s in per_sequence) / total) for k in keys}
    return {"mean_over_sequences": mean, "frame_weighted": weighted, "num_frames": total}


def format_table(rows, names):
    keys = [k for k in REPORT_ORDER if k in rows[0]]
    head = "| sequence | " + " | ".join(keys) + " |"
    sep = "|" + "---|" * (len(keys) + 1)
    lines = [head, sep]
    for name, r in zip(names, rows):
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.2f}" for k in keys) + " |")
    return "\n".join(lines)
