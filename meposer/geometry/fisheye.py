import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]


def project_kb4(points_cam, params):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-9), 0.0)
    return np.stack([fx * x * scale + cx, fy * y * scale + cy], axis=-1)


def fit_kb4(points_cam, pixels, image_size, init_focal=230.0):
    w, h = image_size
    p0 = np.array([init_focal, init_focal, w / 2, h / 2, 0, 0, 0, 0], dtype=np.float64)
    res = least_squares(lambda p: (project_kb4(points_cam, p) - pixels).ravel(), p0, loss="soft_l1", f_scale=2.0)
    err = np.linalg.norm(project_kb4(points_cam, res.x) - pixels, axis=-1)
    return res.x, err
