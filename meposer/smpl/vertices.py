import numpy as np
from scipy.spatial.transform import Rotation as R

from .constants import PARENTS

FOOT_JOINT_GROUPS = {"left": [7, 10], "right": [8, 11]}


def load_smpl_full(npz_path):
    m = np.load(npz_path)
    return {k: m[k].astype(np.float64) for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights")}


def sole_vertex_ids(smpl, fraction=0.25):
    w, v = smpl["weights"], smpl["v_template"]
    ids = {}
    for side, joints in FOOT_JOINT_GROUPS.items():
        foot = np.where(w[:, joints].sum(1) > 0.5)[0]
        y = v[foot, 1]
        ids[side] = foot[y <= np.quantile(y, fraction)]
    return ids


def posed_vertices(smpl, pose72, beta, trans, vids):
    """Linear blend skinning for a subset of vertices. pose72 (N, 72) axis-angle, beta (N, 10), trans (N, 3)."""
    n = len(pose72)
    rot = R.from_rotvec(pose72.reshape(-1, 3)).as_matrix().reshape(n, 24, 3, 3)
    v_shaped_all = smpl["v_template"][None] + np.einsum("vcb,nb->nvc", smpl["shapedirs"], beta)
    joints = np.einsum("jv,nvc->njc", smpl["J_regressor"], v_shaped_all)
    pose_feat = (rot[:, 1:] - np.eye(3)).reshape(n, 207)
    v = v_shaped_all[:, vids] + np.einsum("vcp,np->nvc", smpl["posedirs"][vids], pose_feat)
    g = np.zeros((n, 24, 4, 4))
    for i in range(24):
        local = np.zeros((n, 4, 4))
        local[:, :3, :3] = rot[:, i]
        local[:, :3, 3] = joints[:, i] - (joints[:, PARENTS[i]] if i > 0 else 0)
        local[:, 3, 3] = 1
        g[:, i] = local if i == 0 else g[:, PARENTS[i]] @ local
    g_rest = g.copy()
    g_rest[:, :, :3, 3] -= np.einsum("njab,njb->nja", g[:, :, :3, :3], joints)
    t = np.einsum("vj,njab->nvab", smpl["weights"][vids], g_rest)
    out = np.einsum("nvab,nvb->nva", t[..., :3, :3], v) + t[..., :3, 3]
    return out + trans[:, None]
