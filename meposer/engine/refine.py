import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from ..geometry.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix
from ..smpl.constants import HEAD, LEFT_WRIST, RIGHT_WRIST

ARM_JOINTS = [13, 14, 16, 17, 18, 19]
SIDES = [("left", LEFT_WRIST), ("right", RIGHT_WRIST)]


def _controller(annots, side):
    return R.from_quat(annots[f"{side}_hand_rot"][:, [1, 2, 3, 0]]).as_matrix().astype(np.float32), annots[f"{side}_hand_pos"].astype(np.float32)


def controller_wrist_offsets(processed_dir, subjects, smpl):
    from ..data.features import gt_world_joints
    offs = {s: [] for s, _ in SIDES}
    for subj in subjects:
        a = dict(np.load(f"{processed_dir}/{subj}/annots.npz"))
        joints, _ = gt_world_joints(a, smpl)
        for side, w in SIDES:
            rc, pc = _controller(a, side)
            offs[side].append(np.einsum("nji,nj->ni", rc, joints[:, w] - pc))
    return {s: np.concatenate(v).mean(0).astype(np.float32) for s, v in offs.items()}


def wrist_targets(processed_dir, subject, offsets, a, b):
    ann = dict(np.load(f"{processed_dir}/{subject}/annots.npz"))
    out = []
    for side, _ in SIDES:
        rc, pc = _controller(ann, side)
        out.append(pc[a:b] + np.einsum("nij,j->ni", rc[a:b], offsets[side]))
    return torch.as_tensor(np.stack(out, 1))


def refine_arms(smpl, rot6d, betas, head_anchor, targets, iters=60, lr=0.05, prior=0.1):
    rot6d = rot6d.detach()
    arm = rot6d[:, ARM_JOINTS].clone().requires_grad_(True)
    opt = torch.optim.Adam([arm], lr=lr)
    idx = torch.tensor(ARM_JOINTS)
    for _ in range(iters):
        full = rot6d.clone()
        full[:, idx] = arm
        joints, _ = smpl(rotation_6d_to_matrix(full), betas.detach())
        wrists = joints[:, [LEFT_WRIST, RIGHT_WRIST]] - joints[:, HEAD : HEAD + 1] + head_anchor[:, None]
        loss = (wrists - targets).norm(dim=-1).mean() + prior * (arm - rot6d[:, ARM_JOINTS]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    full = rot6d.clone()
    full[:, idx] = matrix_to_rotation_6d(rotation_6d_to_matrix(arm.detach()))
    with torch.no_grad():
        joints, grot = smpl(rotation_6d_to_matrix(full), betas.detach())
    return full, joints


BODY_JOINTS = [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 18, 19]


def project_kb4_torch(points_cam, params):
    fx, fy, cx, cy, k1, k2, k3, k4 = [float(p) for p in params]
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    r = torch.sqrt(x * x + y * y + 1e-12)
    th = torch.atan2(r, z)
    thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    s = thd / r
    return torch.stack([fx * x * s + cx, fy * y * s + cy], -1)


def refine_body(smpl, rot6d, betas, head_anchor, wrist_tg, peaks, conf, t_cam_world, cam_params, iters=80, lr=0.03,
                conf_thresh=0.3, w2d=1.0, wwrist=1.0, prior=0.1, f_norm=230.0):
    """Test-time fit: optimise limb rotations so that FK joints (anchored at the headset) reproject onto the stereo
    heatmap peaks (confidence-weighted, Huber in pixels / focal) and the wrists match the controller-implied wrists."""
    rot6d = rot6d.detach()
    idx = torch.tensor(BODY_JOINTS)
    var = rot6d[:, BODY_JOINTS].clone().requires_grad_(True)
    opt = torch.optim.Adam([var], lr=lr)
    w = torch.where(conf > conf_thresh, conf, torch.zeros_like(conf))[..., :22]
    for _ in range(iters):
        full = rot6d.clone()
        full[:, idx] = var
        joints, _ = smpl(rotation_6d_to_matrix(full), betas.detach())
        jw = joints - joints[:, HEAD : HEAD + 1] + head_anchor[:, None]
        loss = prior * (var - rot6d[:, BODY_JOINTS]).abs().mean()
        if wrist_tg is not None:
            loss = loss + wwrist * (jw[:, [LEFT_WRIST, RIGHT_WRIST]] - wrist_tg).norm(dim=-1).mean()
        for c in range(2):
            pc = (t_cam_world[:, c, None, :3, :3] @ jw[..., None])[..., 0] + t_cam_world[:, c, None, :3, 3]
            uv = project_kb4_torch(pc, cam_params[c])
            res = torch.nn.functional.huber_loss(uv / f_norm, peaks[:, c] / f_norm, reduction="none", delta=0.05).sum(-1)
            loss = loss + w2d * (w[:, c] * res).sum() / w[:, c].sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    full = rot6d.clone()
    full[:, idx] = matrix_to_rotation_6d(rotation_6d_to_matrix(var.detach()))
    with torch.no_grad():
        joints, _ = smpl(rotation_6d_to_matrix(full), betas.detach())
    return full, joints
