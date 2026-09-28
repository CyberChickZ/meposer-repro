import torch
import torch.nn.functional as F
from torch import nn

from ..geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix
from ..geometry.se3 import transform_points
from ..smpl.constants import HEAD, NUM_BODY_JOINTS
from ..smpl.model import SMPLJoints
from .image_branch import StereoImageBranch
from .imu_branch import IMUBranch
from .layers import mlp
from .temporal import TemporalEncoder

IDENTITY_6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def align_to_world(joints_local, t_world_cam0, head_pos):
    return transform_points(t_world_cam0, joints_local) - head_pos[..., None, :]


class MEPoser(nn.Module):
    def __init__(self, cfg, imu_dim, imu_mean, imu_std, smpl_path, heatmap_size, cached_image_features=False):
        super().__init__()
        self.modalities = list(cfg.modalities)
        self.use_image = "image" in self.modalities
        self.use_imu = "imu" in self.modalities
        self.cached_image_features = cached_image_features
        self.feature_noise = cfg.get("feature_noise") or {"joints_m": 0.0, "image_feat_dropout": 0.0}
        self.feature_noise = type(cfg)(self.feature_noise)
        self.register_buffer("imu_mean", imu_mean.clone())
        self.register_buffer("imu_std", imu_std.clone())
        fused = 0
        self.image_branch = None
        self.image_amp = bool(cfg.image.get("amp", False)) if self.use_image else False
        if self.use_image:
            self.image_branch = None if cached_image_features else StereoImageBranch(cfg.image, heatmap_size)
            self.joint_frame = cfg.image.get("joint_frame", "world")
            assert self.joint_frame in ("world", "head"), self.joint_frame
            joint_in = NUM_BODY_JOINTS * 3 + (6 if self.joint_frame == "head" else 0)
            self.joint_feat = mlp([joint_in, 256, cfg.image.joint_feat_dim], final_act=True)
            fused += cfg.image.heatmap_feat_dim + cfg.image.joint_feat_dim
        if self.use_imu:
            self.imu_branch = IMUBranch(imu_dim, cfg.imu)
            fused += self.imu_branch.out_dim
        d = cfg.temporal.dim
        self.fusion = nn.Sequential(nn.Linear(fused, d), nn.LayerNorm(d))
        self.temporal = TemporalEncoder(d, cfg.temporal.blocks, cfg.temporal.ffn_dim, cfg.temporal.dropout)
        self.pose_head = mlp([d, cfg.heads.hidden, NUM_BODY_JOINTS * 6])
        self.shape_head = mlp([d, cfg.heads.hidden, 10])
        self.contact_head = mlp([d, 128, 2]) if cfg.get("contact_head", False) else None
        with torch.no_grad():
            self.pose_head[-1].weight.mul_(0.01)
            self.pose_head[-1].bias.copy_(torch.tensor(IDENTITY_6D).repeat(NUM_BODY_JOINTS))
        self.smpl = SMPLJoints(smpl_path)

    def image_outputs(self, batch):
        if self.cached_image_features:
            feat, joints = batch["image_feat"], batch["joints_local_pred"]
            if self.training:
                joints = joints + torch.randn_like(joints) * self.feature_noise.joints_m
                feat = F.dropout(feat, self.feature_noise.image_feat_dropout)
            return {"image_feat": feat, "joints_local": joints}
        images = batch["images"]
        b, t = images.shape[:2]
        amp = self.image_amp and images.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = self.image_branch(images.flatten(0, 1))
        return {k: v.float().reshape(b, t, *v.shape[1:]) for k, v in out.items()}

    def joint_input(self, joints_global, batch):
        if self.joint_frame == "world":
            return joints_global.flatten(-2)
        r_head = batch["head_rot"]
        joints_head = joints_global @ r_head
        return torch.cat([joints_head.flatten(-2), matrix_to_rotation_6d(r_head)], -1)

    def forward(self, batch, states=None):
        feats, out = [], {}
        if self.use_image:
            img = self.image_outputs(batch)
            out.update(img)
            out["joints_global"] = align_to_world(img["joints_local"], batch["T_world_cam0"], batch["head_pos"])
            feats += [img["image_feat"], self.joint_feat(self.joint_input(out["joints_global"], batch))]
        if self.use_imu:
            feats.append(self.imu_branch((batch["imu"] - self.imu_mean) / self.imu_std))
        x = self.fusion(torch.cat(feats, -1))
        x, states = self.temporal(x, states)
        b, t = x.shape[:2]
        rot6d = self.pose_head(x).reshape(b, t, NUM_BODY_JOINTS, 6)
        betas = self.shape_head(x)
        rotmats = rotation_6d_to_matrix(rot6d)
        joints, global_rot = self.smpl(rotmats.flatten(0, 1), betas.flatten(0, 1))
        if self.contact_head is not None:
            out["contact_logits"] = self.contact_head(x)
        out.update({
            "rot6d": rot6d,
            "betas": betas,
            "joints_fk": joints.reshape(b, t, NUM_BODY_JOINTS, 3),
            "rot6d_global": matrix_to_rotation_6d(global_rot).reshape(b, t, NUM_BODY_JOINTS, 6),
            "states": states,
        })
        return out


def anchor_joints(joints, anchor):
    return joints - joints[..., HEAD : HEAD + 1, :] + anchor[..., None, :]
