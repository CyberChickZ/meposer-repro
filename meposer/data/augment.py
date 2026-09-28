import math

import torch

from ..geometry.rotations import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

FEATURE_DIMS = {"rot6d": 6, "height": 1, "acc": 3, "pos_rel": 3, "in_head_pos": 3, "in_head_rot6d": 6, "acc_synth": 3}


def _kind(name):
    for suffix in ("in_head_rot6d", "in_head_pos", "acc_synth", "pos_rel", "rot6d", "height", "acc"):
        if name.endswith(suffix):
            return suffix
    raise KeyError(name)


def yaw_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot6d(r, x):
    return matrix_to_rotation_6d(r @ rotation_6d_to_matrix(x))


def augment_window(sample, feature_names, cfg, generator=None):
    """Rotate the whole world about the vertical (y) axis by a random angle and add sensor noise.

    Camera-frame quantities (images, cached image features, local joints, contact labels) are unchanged; every
    world-frame quantity (IMU rotations, positions and accelerations, root and global rotations, joints, headset pose,
    camera-to-world transform) is rotated consistently, so the supervision stays exact.
    """
    theta = (torch.rand(1, generator=generator).item() * 2 - 1) * math.radians(cfg.get("yaw_deg", 180.0))
    r = yaw_matrix(theta)
    imu = sample["imu"].clone()
    leg_bias = {}
    if cfg.get("leg_bias_deg", 0):
        for side in ("left", "right"):
            v = torch.randn(3, generator=generator)
            v = v / v.norm() * math.radians(cfg.leg_bias_deg) * torch.rand(1, generator=generator).item()
            leg_bias[side] = axis_angle_to_matrix(v)
    col = 0
    for name in feature_names:
        kind = _kind(name)
        d = FEATURE_DIMS[kind]
        sl = slice(col, col + d)
        if kind == "rot6d":
            x = imu[:, sl]
            side = name.split("_")[0]
            if name.endswith("leg_rot6d") and side in leg_bias:
                x = matrix_to_rotation_6d(rotation_6d_to_matrix(x) @ leg_bias[side])
            imu[:, sl] = _rot6d(r, x)
        elif kind in ("acc", "acc_synth", "pos_rel"):
            imu[:, sl] = imu[:, sl] @ r.T
        if kind in ("acc", "acc_synth") and cfg.get("acc_noise", 0):
            imu[:, sl] += torch.randn(imu[:, sl].shape, generator=generator) * cfg.acc_noise
        if kind in ("pos_rel", "in_head_pos") and cfg.get("pos_noise", 0):
            imu[:, sl] += torch.randn(imu[:, sl].shape, generator=generator) * cfg.pos_noise
        col += d
    assert col == imu.shape[-1], (col, imu.shape)
    out = dict(sample)
    out["imu"] = imu
    root = out["rot6d_local"].clone()
    root[:, 0] = _rot6d(r, root[:, 0])
    out["rot6d_local"] = root
    out["rot6d_global"] = _rot6d(r, out["rot6d_global"])
    for k in ("joints_world", "head_pos", "head_anchor"):
        out[k] = out[k] @ r.T
    out["head_rot"] = r @ out["head_rot"]
    t = out["T_world_cam0"].clone()
    t[:, :3, :3] = r @ t[:, :3, :3]
    t[:, :3, 3] = t[:, :3, 3] @ r.T
    out["T_world_cam0"] = t
    return out
