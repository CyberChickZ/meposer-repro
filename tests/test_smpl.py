from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from meposer.smpl.constants import PARENTS
from meposer.smpl.model import SMPLJoints

SMPL_PATH = Path(__file__).resolve().parents[1] / "assets" / "smpl" / "SMPL_NEUTRAL.npz"


def _numpy_fk(npz, pose72, beta):
    v = npz["v_template"] + npz["shapedirs"] @ beta
    j = npz["J_regressor"] @ v
    rots = R.from_rotvec(pose72.reshape(24, 3)).as_matrix()
    g_rot, g_pos = [rots[0]], [j[0]]
    for i in range(1, 22):
        p = PARENTS[i]
        g_rot.append(g_rot[p] @ rots[i])
        g_pos.append(g_pos[p] + g_rot[p] @ (j[i] - j[p]))
    return np.stack(g_pos), np.stack(g_rot)


@pytest.mark.skipif(not SMPL_PATH.exists(), reason="SMPL model not present")
def test_fk_matches_reference():
    npz = np.load(SMPL_PATH)
    model = SMPLJoints(SMPL_PATH)
    rng = np.random.default_rng(0)
    pose = rng.normal(scale=0.4, size=72).astype(np.float32)
    beta = rng.normal(scale=1.0, size=10).astype(np.float32)
    ref_pos, ref_rot = _numpy_fk(npz, pose, beta)
    rotmats = torch.as_tensor(R.from_rotvec(pose.reshape(24, 3)[:22]).as_matrix(), dtype=torch.float32)[None]
    pos, rot = model(rotmats, torch.as_tensor(beta)[None])
    assert torch.allclose(pos[0], torch.as_tensor(ref_pos, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(rot[0], torch.as_tensor(ref_rot, dtype=torch.float32), atol=1e-5)
