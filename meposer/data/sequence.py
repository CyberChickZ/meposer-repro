import json
from pathlib import Path

import numpy as np

from ..geometry.se3 import invert_transform, transform_points
from ..smpl.constants import FOOT_JOINTS, HEAD
from .features import (camera_from_world, gt_rotmats, gt_world_joints, head_joint_anchor, head_pose, imu_feature_dict,
                       matrix_to_6d_np, stack_features)
from .raw import CAMERAS


def foot_contact(joints_world, frames, floor_y=None, height_m=0.03, speed_m_per_frame=0.01):
    feet = joints_world[:, FOOT_JOINTS]
    if floor_y is None:
        floor_y = np.percentile(feet[..., 1].min(axis=(1, 2)), 5)
    speed = np.zeros(feet.shape[:3], np.float32)
    for a, b in contiguous_segments(frames):
        if b - a > 1:
            speed[a + 1 : b] = np.linalg.norm(np.diff(feet[a:b], axis=0), axis=-1)
            speed[a] = speed[a + 1]
    contact = ((feet[..., 1] - floor_y) < height_m) & (speed < speed_m_per_frame)
    return contact.any(-1).astype(np.float32), float(floor_y)


def vertex_foot_contact(processed_dir, subject, annots, smpl_path, frames, height_m=0.02, speed_m_per_frame=0.01):
    from pathlib import Path
    cache = Path(processed_dir) / subject / "sole_vertices.npz"
    if cache.exists():
        verts = np.load(cache)["verts"]
    else:
        from ..smpl.vertices import load_smpl_full, posed_vertices, sole_vertex_ids
        smpl = load_smpl_full(smpl_path)
        ids = sole_vertex_ids(smpl)
        vids = np.concatenate([ids["left"], ids["right"]])
        pose = np.concatenate([annots["root_orient"], annots["pose"]], 1)
        verts = np.concatenate([posed_vertices(smpl, pose[i:i + 512], annots["shape"][i:i + 512], annots["trans"][i:i + 512], vids)
                                for i in range(0, len(pose), 512)]).astype(np.float32)
        np.savez(cache, verts=verts, n_left=len(ids["left"]))
    n = verts.shape[1] // 2
    feet = np.stack([verts[:, :n], verts[:, n:]], 1)
    floor = np.percentile(feet[..., 1].min(axis=(1, 2)), 5)
    speed = np.zeros(feet.shape[:3], np.float32)
    for a, b in contiguous_segments(frames):
        if b - a > 1:
            speed[a + 1 : b] = np.linalg.norm(np.diff(feet[a:b], axis=0), axis=-1)
            speed[a] = speed[a + 1]
    planted = ((feet[..., 1] - floor) < height_m) & (speed < speed_m_per_frame)
    return planted.any(-1).astype(np.float32), float(floor)


def contiguous_segments(frames, min_length=1):
    frames = np.asarray(frames)
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    bounds = np.concatenate([[0], breaks, [len(frames)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_length]


class SubjectSequence:
    def __init__(self, processed_dir, subject, calib, smpl, imu_features, load_images=True, fps=30.0, synth_acc_cutoff_hz=3.0, contact=None):
        d = Path(processed_dir) / subject
        with open(d / "meta.json") as f:
            self.meta = json.load(f)
        self.subject = subject
        a = dict(np.load(d / "annots.npz"))
        self.frames = a["frame"]
        self.p2d = a["p2d"]
        self.images = np.load(d / self.meta["images"], mmap_mode="r") if load_images else None
        self.image_size = tuple(self.meta["image_size"])
        self.imu = stack_features(imu_feature_dict(a, fps, synth_acc_cutoff_hz), imu_features)
        rotmats = gt_rotmats(a)
        self.rot6d_local = matrix_to_6d_np(rotmats)
        self.betas = a["shape"]
        self.joints_world, global_rot = gt_world_joints(a, smpl, rotmats)
        self.rot6d_global = matrix_to_6d_np(global_rot)
        r_head, p_head = head_pose(a)
        self.head_pos = p_head
        self.head_rot = r_head
        self.head_anchor = head_joint_anchor(r_head, p_head, calib["head_joint_offset"])
        self.T_cam_world = np.stack([camera_from_world(r_head, p_head, calib["cameras"][c]["T_cam_head"]) for c in CAMERAS], 1)
        self.T_world_cam0 = invert_transform(self.T_cam_world[:, 0])
        self.joints_local = transform_points(self.T_cam_world[:, 0], self.joints_world)
        contact = contact or {}
        if contact.get("source", "joints") == "vertices":
            self.contact, self.floor_y = vertex_foot_contact(processed_dir, subject, a, contact["smpl"], self.frames,
                                                             contact.get("vertex_height_m", 0.02), contact.get("speed_m_per_frame", 0.01))
        else:
            self.contact, self.floor_y = foot_contact(self.joints_world, self.frames, contact.get("floor_y"), contact.get("height_m", 0.03), contact.get("speed_m_per_frame", 0.01))
        self.image_features = None

    def __len__(self):
        return len(self.frames)

    def segments(self, min_length=1):
        return contiguous_segments(self.frames, min_length)

    def attach_features(self, feats):
        assert len(feats["image_feat"]) == len(self), (self.subject, len(feats["image_feat"]), len(self))
        self.image_features = feats
