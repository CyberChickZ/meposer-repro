from pathlib import Path

import numpy as np
import torch

from ..geometry.rotations import rotation_6d_to_axis_angle
from ..metrics.evaluate import sequence_metrics
from ..smpl.constants import HEAD, NUM_BODY_JOINTS
from .build import full_model_from_checkpoint, sequences_for
from .common import load_checkpoint, resolve_device
from .infer import run_sequence


@torch.no_grad()
def predict_subject(model, seq, device, cfg):
    n = len(seq)
    pose = np.zeros((n, 72), np.float32)
    betas = np.zeros((n, 10), np.float32)
    trans = np.full((n, 3), np.nan, np.float32)
    joints = np.full((n, NUM_BODY_JOINTS, 3), np.nan, np.float32)
    valid = np.zeros(n, bool)
    metrics = []
    for (a, b), out in run_sequence(model, seq, device, cfg):
        aa = rotation_6d_to_axis_angle(out["rot6d"].reshape(-1, 6)).reshape(b - a, NUM_BODY_JOINTS * 3).numpy()
        pose[a:b, : NUM_BODY_JOINTS * 3] = aa
        betas[a:b] = out["betas"].numpy()
        joints[a:b] = out["joints_world"].numpy()
        trans[a:b] = seq.head_anchor[a:b] - out["joints_fk"][:, HEAD].numpy()
        valid[a:b] = True
        metrics.append(sequence_metrics(out["rot6d"], out["joints_fk"], torch.as_tensor(seq.rot6d_local[a:b]), torch.as_tensor(seq.joints_world[a:b]), cfg.data.fps))
    mpjpe = float(sum(m["mpjpe"] * m["num_frames"] for m in metrics) / max(1, sum(m["num_frames"] for m in metrics)))
    return {"frames": seq.frames, "valid": valid, "pose": pose, "betas": betas, "trans": trans, "joints_world": joints}, mpjpe


def run(ckpt_path, subject, out_path, overrides=()):
    ckpt = load_checkpoint(ckpt_path)
    model, cfg = full_model_from_checkpoint(ckpt, overrides)
    device = resolve_device(cfg.device)
    torch.set_num_threads(cfg.get("threads", 10))
    model = model.to(device).eval()
    cfg.data.val_subjects = [subject]
    seq = sequences_for(cfg, "val")[0]
    pred, mpjpe = predict_subject(model, seq, device, cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **pred, checkpoint=str(ckpt_path), subject=subject)
    print(f"{subject}: {int(pred['valid'].sum())}/{len(seq)} frames predicted, MPJPE vs annotations {mpjpe:.2f} cm -> {out_path}")
    print("keys: frames, valid, pose (72 axis-angle, SMPL order, hands zero), betas (10), trans (3, pelvis, from IMU head anchor), joints_world (22x3)")
