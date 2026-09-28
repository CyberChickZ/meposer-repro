from pathlib import Path

import torch

from meposer.geometry.rotations import axis_angle_to_rotation_6d
from meposer.metrics.evaluate import sequence_metrics


def test_perfect_prediction_gives_zero_errors():
    g = torch.Generator().manual_seed(0)
    rot6d = axis_angle_to_rotation_6d(torch.randn(10, 22, 3, generator=g) * 0.3)
    joints = torch.randn(10, 22, 3, generator=g)
    m = sequence_metrics(rot6d, joints + 0.5, rot6d, joints, fps=30)
    for k in ["mpjre", "mpjpe", "pa_mpjpe", "upperpe", "lowerpe", "rootpe", "mpjve"]:
        assert abs(m[k]) < 1e-3, (k, m[k])
    assert abs(m["pred_jitter"] - m["gt_jitter"]) < 1e-2


def test_translation_offsets_are_removed_by_head_alignment():
    g = torch.Generator().manual_seed(1)
    rot6d = axis_angle_to_rotation_6d(torch.zeros(5, 22, 3))
    joints = torch.randn(5, 22, 3, generator=g)
    m = sequence_metrics(rot6d, joints + torch.tensor([1.0, 2.0, 3.0]), rot6d, joints, fps=30)
    assert m["mpjpe"] < 1e-4


def test_foot_skate_and_contact_shapes():
    from meposer.metrics.evaluate import contact_scores, foot_skate_cm_s
    g = torch.Generator().manual_seed(2)
    joints = torch.randn(20, 22, 3, generator=g)
    contact = (torch.rand(20, 2, generator=g) > 0.5).float()
    assert foot_skate_cm_s(joints, contact, 30) > 0
    assert foot_skate_cm_s(joints.repeat(1, 1, 1) * 0 + 1, contact, 30) == 0
    acc, f1 = contact_scores(torch.where(contact > 0, 5.0, -5.0), contact)
    assert acc == 1.0 and abs(f1 - 1.0) < 1e-6
    rot6d = axis_angle_to_rotation_6d(torch.zeros(20, 22, 3))
    m = sequence_metrics(rot6d, joints, rot6d, joints, fps=30, contact=contact, contact_logits=torch.zeros(20, 2))
    assert "foot_skate" in m and "contact_acc" in m


def test_loss_is_zero_for_perfect_prediction_and_weights_apply():
    from meposer.config import load_config
    from meposer.losses import MEPoserLoss
    from meposer.smpl.constants import HEAD
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    g = torch.Generator().manual_seed(3)
    b, t = 2, 8
    rot6d = axis_angle_to_rotation_6d(torch.randn(b, t, 22, 3, generator=g) * 0.2)
    joints = torch.randn(b, t, 22, 3, generator=g)
    batch = {"rot6d_local": rot6d, "rot6d_global": rot6d, "betas": torch.zeros(b, t, 10), "joints_world": joints, "contact": torch.ones(b, t, 2)}
    out = {"rot6d": rot6d.clone(), "rot6d_global": rot6d.clone(), "betas": torch.zeros(b, t, 10), "joints_fk": joints - joints[..., HEAD : HEAD + 1, :]}
    total, parts = MEPoserLoss(cfg.loss)(out, batch)
    assert float(total) < 1e-6
    out["betas"] = torch.ones(b, t, 10)
    total2, parts2 = MEPoserLoss(cfg.loss)(out, batch)
    assert abs(float(total2) - cfg.loss.shape * 1.0) < 1e-6
