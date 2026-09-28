import torch
from scipy.spatial.transform import Rotation as R

from meposer.geometry import rotations as rot


def _random_rotvecs(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g) * 1.5


def test_axis_angle_matches_scipy():
    aa = _random_rotvecs()
    ours = rot.axis_angle_to_matrix(aa)
    ref = torch.as_tensor(R.from_rotvec(aa.numpy()).as_matrix(), dtype=torch.float32)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_quaternion_wxyz_matches_scipy():
    aa = _random_rotvecs(seed=1)
    q_xyzw = R.from_rotvec(aa.numpy()).as_quat()
    q_wxyz = torch.as_tensor(q_xyzw[:, [3, 0, 1, 2]], dtype=torch.float32)
    ref = torch.as_tensor(R.from_rotvec(aa.numpy()).as_matrix(), dtype=torch.float32)
    assert torch.allclose(rot.quaternion_wxyz_to_matrix(q_wxyz), ref, atol=1e-5)


def test_round_trips():
    aa = _random_rotvecs(seed=2)
    m = rot.axis_angle_to_matrix(aa)
    assert torch.allclose(rot.rotation_6d_to_matrix(rot.matrix_to_rotation_6d(m)), m, atol=1e-5)
    back = rot.matrix_to_axis_angle(m)
    assert torch.allclose(rot.axis_angle_to_matrix(back), m, atol=1e-4)
    assert torch.allclose(rot.relative_angle_deg(m, m), torch.zeros(len(m)), atol=0.1)
