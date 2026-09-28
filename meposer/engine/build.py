import torch

from ..config import Config
from ..data.dataset import compute_feature_stats, load_sequences
from ..models.image_branch import StereoImageBranch
from ..models.meposer import MEPoser
from .common import heatmap_size, load_checkpoint


def apply_overrides(cfg_dict, overrides):
    from ..config import apply_overrides as _apply
    return _apply(Config.from_dict(cfg_dict), overrides)


def image_branch_from_checkpoint(ckpt):
    cfg = Config.from_dict(ckpt["cfg"])
    model = StereoImageBranch(cfg.model.image, tuple(ckpt["heatmap_size"]))
    model.load_state_dict(ckpt["model"])
    return model, cfg


def build_full_model(cfg, train_seqs=None, stats=None):
    cached = cfg.model.image_features is not None and "image" in cfg.model.modalities
    if stats is None:
        stats = compute_feature_stats(train_seqs)
    mean, std = stats
    model = MEPoser(cfg.model, mean.numel(), mean, std, cfg.data.smpl, heatmap_size(cfg), cached_image_features=cached)
    if cfg.model.get("init_from"):
        state = load_checkpoint(cfg.model.init_from)["model"]
        missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("imu_")}, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("image_branch.") or k.startswith("imu_") for k in missing), missing
    if not cached and model.image_branch is not None and cfg.model.image.get("init_from"):
        branch, _ = image_branch_from_checkpoint(load_checkpoint(cfg.model.image.init_from))
        model.image_branch.load_state_dict(branch.state_dict())
    return model


def full_model_from_checkpoint(ckpt, overrides=()):
    cfg = apply_overrides(ckpt["cfg"], overrides)
    model = build_full_model(cfg, stats=(torch.as_tensor(ckpt["imu_mean"]), torch.as_tensor(ckpt["imu_std"])))
    model.load_state_dict(ckpt["model"])
    return model, cfg


DEFAULT_STAGE1 = "checkpoints/stage1_image.pt"


def ensure_feature_cache(cfg, subjects):
    from pathlib import Path
    feature_dir = Path(cfg.model.image_features)
    missing = [s for s in subjects if not (feature_dir / f"{s}.npz").exists()]
    if not missing:
        return
    stage1 = cfg.model.get("stage1_checkpoint") or DEFAULT_STAGE1
    if not Path(stage1).exists():
        raise FileNotFoundError(f"cached image features for {missing} not found in {feature_dir} and no stage-1 checkpoint at {stage1} to rebuild them")
    print(f"image-feature cache missing for {missing}; rebuilding from {stage1} into {feature_dir}", flush=True)
    from .extract_features import run as extract
    extract(stage1, feature_dir, cfg=cfg, subjects=missing)


def sequences_for(cfg, split, load_images=True):
    subjects = {"train": cfg.data.train_subjects, "val": cfg.data.val_subjects}[split]
    feature_dir = cfg.model.image_features if "image" in cfg.model.modalities else None
    if feature_dir is not None:
        ensure_feature_cache(cfg, list(subjects))
    need_images = load_images and feature_dir is None
    return load_sequences(cfg, list(subjects), load_images=need_images, feature_dir=feature_dir)
