from pathlib import Path

import numpy as np
import pytest
import torch

from meposer.config import load_config
from meposer.data.dataset import FrameDataset, WindowDataset, load_sequences
from meposer.data.sequence import contiguous_segments
from meposer.geometry.se3 import transform_points
from meposer.smpl.constants import HEAD

ROOT = Path(__file__).resolve().parents[1]
READY = (ROOT / "data/processed/index.json").exists() and (ROOT / "assets/calib/calib.json").exists()


def test_contiguous_segments():
    assert contiguous_segments([0, 1, 2, 5, 6, 9], min_length=2) == [(0, 3), (3, 5)]


@pytest.mark.skipif(not READY, reason="processed data / calibration not present")
def test_sequence_consistency():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg.data.processed = str(ROOT / "data/processed")
    cfg.data.calib = str(ROOT / "assets/calib/calib.json")
    cfg.data.smpl = str(ROOT / "assets/smpl/SMPL_NEUTRAL.npz")
    seq = load_sequences(cfg, ["0009"])[0]
    assert seq.imu.shape == (len(seq), 67)
    back = transform_points(seq.T_world_cam0, seq.joints_local)
    assert np.abs(back - seq.joints_world).max() < 1e-4
    assert np.linalg.norm(seq.head_anchor - seq.joints_world[:, HEAD], axis=1).mean() < 0.02
    assert np.abs(np.linalg.norm(seq.rot6d_local[..., :3], axis=-1) - 1).max() < 1e-4
    fd = FrameDataset([seq], 50, (40, 30), 1.5)
    item = fd[0]
    assert item["images"].dtype == torch.uint8 and item["images"].shape == (2, 240, 320)
    assert item["heatmaps"].shape == (44, 30, 40)
    wd = WindowDataset([seq], 32, 16)
    w = wd[0]
    assert w["imu"].shape == (32, 67) and w["rot6d_local"].shape == (32, 22, 6)


@pytest.mark.skipif(not READY, reason="processed data / calibration not present")
def test_yaw_augmentation_keeps_supervision_consistent():
    from meposer.data.augment import augment_window
    from meposer.config import Config
    from meposer.geometry.rotations import rotation_6d_to_matrix
    from meposer.smpl.model import SMPLJoints
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg.data.processed = str(ROOT / "data/processed")
    cfg.data.calib = str(ROOT / "assets/calib/calib.json")
    cfg.data.smpl = str(ROOT / "assets/smpl/SMPL_NEUTRAL.npz")
    seq = load_sequences(cfg, ["0009"], load_images=False)[0]
    w = WindowDataset([seq], 32, 16)[3]
    a = augment_window(w, list(cfg.data.imu_features), Config({"yaw_deg": 180, "acc_noise": 0, "pos_noise": 0}))
    smpl = SMPLJoints(cfg.data.smpl)
    joints, _ = smpl(rotation_6d_to_matrix(a["rot6d_local"]), a["betas"])
    rel = joints - joints[:, 15:16] + a["joints_world"][:, 15:16]
    assert (rel - a["joints_world"]).abs().max() < 1e-4
    assert torch.allclose(a["joints_world"].norm(dim=-1), w["joints_world"].norm(dim=-1), atol=1e-5)
    assert not torch.allclose(a["imu"], w["imu"])
    back = transform_points(a["T_world_cam0"].numpy(), a["joints_local"].numpy())
    assert np.abs(back - a["joints_world"].numpy()).max() < 1e-3
