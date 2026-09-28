import numpy as np
import torch
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from ..geometry.se3 import invert_transform, make_transform
from ..smpl.constants import NUM_BODY_JOINTS

DEFAULT_IMU_FEATURES = [
    "head_rot6d", "head_height", "head_acc",
    "left_hand_rot6d", "left_hand_pos_rel", "right_hand_rot6d", "right_hand_pos_rel", "right_hand_acc",
    "left_leg_rot6d", "left_leg_acc", "right_leg_rot6d", "right_leg_acc",
    "left_hand_in_head_pos", "left_hand_in_head_rot6d", "right_hand_in_head_pos", "right_hand_in_head_rot6d",
]


def quat_wxyz_to_matrix_np(q):
    return R.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)


def matrix_to_6d_np(m):
    return m[..., :2, :].reshape(m.shape[:-2] + (6,)).astype(np.float32)


def synthetic_acceleration(pos, frames, fps=30.0, cutoff_hz=3.0, order=2):
    out = np.zeros_like(pos, dtype=np.float32)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    b, a = butter(order, cutoff_hz / (fps / 2))
    for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, len(frames)]):
        if hi - lo < 3:
            continue
        acc = np.gradient(np.gradient(pos[lo:hi], axis=0), axis=0) * fps**2
        out[lo:hi] = filtfilt(b, a, acc, axis=0) if hi - lo > 3 * (order + 1) * 3 else acc
    return out


def head_pose(annots):
    return quat_wxyz_to_matrix_np(annots["head_rot"]), np.asarray(annots["head_pos"], np.float32)


def imu_feature_dict(annots, fps=30.0, synth_acc_cutoff_hz=3.0):
    r_head, p_head = head_pose(annots)
    feats = {"head_rot6d": matrix_to_6d_np(r_head), "head_height": p_head[:, 1:2], "head_acc": annots["head_acc"]}
    r_head_t = np.swapaxes(r_head, 1, 2)
    for side in ["left", "right"]:
        r = quat_wxyz_to_matrix_np(annots[f"{side}_hand_rot"])
        p = annots[f"{side}_hand_pos"]
        feats[f"{side}_hand_rot6d"] = matrix_to_6d_np(r)
        feats[f"{side}_hand_pos_rel"] = p - p_head
        feats[f"{side}_hand_acc"] = annots[f"{side}_hand_acc"]
        feats[f"{side}_hand_acc_synth"] = synthetic_acceleration(p, annots["frame"], fps, synth_acc_cutoff_hz)
        feats[f"{side}_hand_in_head_pos"] = np.einsum("nij,nj->ni", r_head_t, p - p_head).astype(np.float32)
        feats[f"{side}_hand_in_head_rot6d"] = matrix_to_6d_np(r_head_t @ r)
        feats[f"{side}_leg_rot6d"] = matrix_to_6d_np(quat_wxyz_to_matrix_np(annots[f"{side}_leg_rot"]))
        feats[f"{side}_leg_acc"] = annots[f"{side}_leg_acc"]
    return {k: np.asarray(v, np.float32) for k, v in feats.items()}


def stack_features(feats, names):
    return np.concatenate([feats[n] for n in names], axis=1).astype(np.float32)


def gt_rotmats(annots):
    aa = np.concatenate([annots["root_orient"], annots["pose"][:, : 3 * (NUM_BODY_JOINTS - 1)]], axis=1)
    n = len(aa)
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(n, NUM_BODY_JOINTS, 3, 3).astype(np.float32)


@torch.no_grad()
def gt_world_joints(annots, smpl, rotmats=None):
    rotmats = gt_rotmats(annots) if rotmats is None else rotmats
    joints, global_rot = smpl(torch.as_tensor(rotmats), torch.as_tensor(annots["shape"]))
    joints = joints.numpy() + annots["trans"][:, None]
    return joints.astype(np.float32), global_rot.numpy().astype(np.float32)


def world_from_head(r_head, p_head):
    return make_transform(r_head, p_head)


def camera_from_world(r_head, p_head, t_cam_head):
    return (t_cam_head[None] @ invert_transform(world_from_head(r_head, p_head))).astype(np.float32)


def head_joint_anchor(r_head, p_head, head_offset):
    return p_head + np.einsum("nij,j->ni", r_head, np.asarray(head_offset, np.float32))
