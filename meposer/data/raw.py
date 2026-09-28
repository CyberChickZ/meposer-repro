from pathlib import Path

import numpy as np

IMU_KEYS = [
    "head_pos", "head_rot", "head_acc",
    "left_hand_pos", "left_hand_rot", "left_hand_acc",
    "right_hand_pos", "right_hand_rot", "right_hand_acc",
    "left_leg_rot", "left_leg_acc",
    "right_leg_rot", "right_leg_acc",
]
CAMERAS = ["cam0", "cam1"]
IMAGE_SIZE = (640, 480)


def list_subjects(raw_dir):
    raw_dir = Path(raw_dir)
    return sorted(p.name for p in raw_dir.iterdir() if (p / "annots").is_dir())


def annot_files(raw_dir, subject):
    return sorted((Path(raw_dir) / subject / "annots").glob("*.npy"))


def image_path(raw_dir, subject, cam, frame):
    return Path(raw_dir) / subject / "camera" / cam / f"{frame:06d}.png"


def load_annot(path):
    d = np.load(path, allow_pickle=True).item()
    smpl = d["smpl"]
    out = {
        "frame": int(Path(path).stem),
        "root_orient": np.asarray(smpl["R"], np.float32),
        "pose": np.asarray(smpl["pose"], np.float32),
        "shape": np.asarray(smpl["shape"], np.float32),
        "trans": np.asarray(smpl["t"], np.float32),
        "p2d": np.stack([np.asarray(p, np.float32) for p in d["p2d"]]),
        "extrinsics": np.stack([np.asarray(d["extrinsics"][c], np.float32) for c in CAMERAS]),
    }
    for k in IMU_KEYS:
        out[k] = np.asarray(d["IMU"][k], np.float32).reshape(-1)
    return out
