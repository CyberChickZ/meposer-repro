import torch
import torch.nn.functional as F


def quaternion_wxyz_to_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    two = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        [
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def axis_angle_to_matrix(aa):
    angle = aa.norm(dim=-1, keepdim=True)
    axis = aa / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    s = torch.sin(angle)[..., None]
    c = torch.cos(angle)[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(k.shape)
    return eye + s * k + (1 - c) * (k @ k)


def matrix_to_quaternion_wxyz(m):
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
            ),
            0.0,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    best = q_abs.argmax(dim=-1)
    q = quat_candidates[F.one_hot(best, num_classes=4).bool()].reshape(m.shape[:-2] + (4,))
    return torch.where(q[..., :1] < 0, -q, q)


def quaternion_wxyz_to_axis_angle(q):
    norms = q[..., 1:].norm(dim=-1, keepdim=True)
    half = torch.atan2(norms, q[..., :1])
    angle = 2 * half
    eps = 1e-6
    small = angle.abs() < eps
    sin_half_over_angle = torch.empty_like(angle)
    sin_half_over_angle[~small] = torch.sin(half[~small]) / angle[~small]
    sin_half_over_angle[small] = 0.5 - (angle[small] ** 2) / 48
    return q[..., 1:] / sin_half_over_angle


def matrix_to_axis_angle(m):
    return quaternion_wxyz_to_axis_angle(matrix_to_quaternion_wxyz(m))


def matrix_to_rotation_6d(m):
    return m[..., :2, :].reshape(m.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def axis_angle_to_rotation_6d(aa):
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(d6):
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def relative_angle_deg(r1, r2):
    rel = r1.transpose(-1, -2) @ r2
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))
