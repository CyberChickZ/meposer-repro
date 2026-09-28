import numpy as np
import torch
from torch import nn

from .constants import NUM_BODY_JOINTS, PARENTS


class SMPLJoints(nn.Module):
    def __init__(self, npz_path, num_joints=NUM_BODY_JOINTS):
        super().__init__()
        m = np.load(npz_path)
        self.num_joints = num_joints
        self.register_buffer("v_template", torch.as_tensor(m["v_template"], dtype=torch.float32))
        self.register_buffer("shapedirs", torch.as_tensor(m["shapedirs"], dtype=torch.float32))
        self.register_buffer("j_regressor", torch.as_tensor(m["J_regressor"][:num_joints], dtype=torch.float32))
        self.parents = PARENTS[:num_joints]

    def rest_joints(self, betas):
        verts = self.v_template[None] + torch.einsum("vcb,nb->nvc", self.shapedirs, betas)
        return torch.einsum("jv,nvc->njc", self.j_regressor, verts)

    def forward(self, rotmats, betas):
        rest = self.rest_joints(betas)
        n, j = rotmats.shape[:2]
        assert j == self.num_joints
        global_rot = [rotmats[:, 0]]
        global_pos = [rest[:, 0]]
        for i in range(1, j):
            p = self.parents[i]
            global_rot.append(global_rot[p] @ rotmats[:, i])
            offset = (rest[:, i] - rest[:, p])[..., None]
            global_pos.append(global_pos[p] + (global_rot[p] @ offset)[..., 0])
        return torch.stack(global_pos, 1), torch.stack(global_rot, 1)
