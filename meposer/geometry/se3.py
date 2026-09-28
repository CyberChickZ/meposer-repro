import numpy as np
import torch


def make_transform(rot, trans):
    lib = torch if torch.is_tensor(rot) else np
    shape = rot.shape[:-2]
    if lib is torch:
        t = torch.zeros(shape + (4, 4), dtype=rot.dtype, device=rot.device)
    else:
        t = np.zeros(shape + (4, 4), dtype=rot.dtype)
    t[..., :3, :3] = rot
    t[..., :3, 3] = trans
    t[..., 3, 3] = 1.0
    return t


def invert_transform(t):
    rot = t[..., :3, :3]
    rt = rot.transpose(-1, -2) if torch.is_tensor(t) else np.swapaxes(rot, -1, -2)
    trans = -(rt @ t[..., :3, 3:4])[..., 0]
    return make_transform(rt, trans)


def _swap(m):
    return m.transpose(-1, -2) if torch.is_tensor(m) else np.swapaxes(m, -1, -2)


def rotate_points(rot, points):
    return points @ _swap(rot)


def transform_points(t, points):
    return rotate_points(t[..., :3, :3], points) + t[..., None, :3, 3]
