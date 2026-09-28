import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .raw import CAMERAS, IMU_KEYS, annot_files, image_path, list_subjects, load_annot

ANNOT_KEYS = ["frame", "root_orient", "pose", "shape", "trans", "p2d", "extrinsics"] + IMU_KEYS


def _read_resized(args):
    path, size = args
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


def _memmap_ok(path, shape):
    if not path.exists():
        return False
    try:
        return np.load(path, mmap_mode="r").shape == shape
    except (ValueError, OSError):
        return False


def prepare_subject(raw_dir, out_dir, subject, image_size, workers):
    out_dir = Path(out_dir) / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [load_annot(p) for p in annot_files(raw_dir, subject)]
    annots = {k: np.stack([r[k] for r in records]) for k in ANNOT_KEYS}
    np.savez(out_dir / "annots.npz", **annots)
    w, h = image_size
    n = len(records)
    mm_path = out_dir / f"images_{w}x{h}.u8"
    if not _memmap_ok(mm_path, (n, len(CAMERAS), h, w)):
        images = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8, shape=(n, len(CAMERAS), h, w))
        jobs = [(image_path(raw_dir, subject, c, int(r["frame"])), image_size) for r in records for c in CAMERAS]
        with Pool(workers) as pool:
            for idx, img in enumerate(tqdm(pool.imap(_read_resized, jobs, chunksize=16), total=len(jobs), desc=subject, leave=False)):
                images[idx // len(CAMERAS), idx % len(CAMERAS)] = img
        images.flush()
        del images
    meta = {"subject": subject, "num_frames": n, "image_size": [w, h], "cameras": CAMERAS, "images": mm_path.name}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def prepare(raw_dir, out_dir, image_size=(320, 240), workers=8, subjects=None, calib_path="assets/calib/calib.json",
            smpl_path="assets/smpl/SMPL_NEUTRAL.npz", recalibrate=False):
    from .calibrate import calibrate
    from .inspect import inspect

    subjects = subjects or list_subjects(raw_dir)
    metas = [prepare_subject(raw_dir, out_dir, s, image_size, workers) for s in tqdm(subjects, desc="subjects")]
    with open(Path(out_dir) / "index.json", "w") as f:
        json.dump({"image_size": list(image_size), "subjects": metas}, f, indent=2)
    if recalibrate or not Path(calib_path).exists():
        print(f"calibrating cameras from annotations -> {calib_path}", flush=True)
        calibrate(out_dir, calib_path, smpl_path=smpl_path)
    else:
        print(f"using the existing calibration {calib_path} (--recalibrate to refit it from these subjects)", flush=True)
    print("inspecting the prepared data", flush=True)
    report = inspect(out_dir, Path(out_dir) / "inspection", smpl_path)
    print(f"inspection report -> {report}", flush=True)
    return metas
