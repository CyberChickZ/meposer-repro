import torch
import torch.nn.functional as F
from torch import nn

from .models.meposer import anchor_joints
from .smpl.constants import FOOT_JOINTS, HEAD


class MEPoserLoss(nn.Module):
    def __init__(self, cfg, image_joints_are_inputs=False):
        super().__init__()
        self.w = cfg
        self.smpl_w = cfg.smpl_terms
        self.image_joints_are_inputs = image_joints_are_inputs

    def forward(self, out, batch):
        parts = {}
        if "heatmap_logits" in out and "heatmaps" in batch:
            parts["heatmap"] = F.binary_cross_entropy_with_logits(out["heatmap_logits"], batch["heatmaps"])
        if "joints_local" in out and not self.image_joints_are_inputs:
            parts["local_joints"] = F.l1_loss(out["joints_local"], batch["joints_local"])
        if "joints_global" in out and not self.image_joints_are_inputs:
            parts["global_joints"] = F.l1_loss(out["joints_global"], batch["joints_world"] - batch["head_pos"][..., None, :])
        if "rot6d" in out:
            gt_anchor = batch["joints_world"][..., HEAD, :]
            pred = anchor_joints(out["joints_fk"], gt_anchor)
            gt = batch["joints_world"]
            smpl_terms = {
                "root_orient": F.l1_loss(out["rot6d"][..., 0, :], batch["rot6d_local"][..., 0, :]),
                "local_rot": F.l1_loss(out["rot6d"][..., 1:, :], batch["rot6d_local"][..., 1:, :]),
                "global_rot": F.l1_loss(out["rot6d_global"], batch["rot6d_global"]),
                "joints": F.l1_loss(pred, gt),
            }
            parts["smpl"] = sum(self.smpl_w[k] * v for k, v in smpl_terms.items())
            parts.update({f"smpl/{k}": v for k, v in smpl_terms.items()})
            if pred.shape[1] >= 3:
                acc_p = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
                acc_g = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
                parts["smooth"] = F.l1_loss(acc_p, acc_g)
            parts["shape"] = F.l1_loss(out["betas"], batch["betas"])
            if "contact" in batch and pred.shape[1] >= 2 and self.w.get("foot_skate", 0) > 0:
                feet = pred[:, :, FOOT_JOINTS]
                vel = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1).mean(-1)
                planted = batch["contact"][:, 1:]
                parts["foot_skate"] = (vel * planted).sum() / planted.sum().clamp_min(1.0)
            if "contact_logits" in out and "contact" in batch:
                parts["contact"] = F.binary_cross_entropy_with_logits(out["contact_logits"], batch["contact"])
        total = sum(self.w[k] * v for k, v in parts.items() if k in self.w and not k.startswith("smpl/"))
        parts["total"] = total
        return total, parts
