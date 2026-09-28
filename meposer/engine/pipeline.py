from pathlib import Path

from ..config import load_config, save_config
from . import extract_features, train_full, train_image

STAGES = ("image", "features", "full")


def _needs_image_stage(cfg):
    return "image" in cfg.model.modalities and not cfg.model.get("image_features") and not cfg.model.get("end_to_end", False)


def run(config_path, out_dir, overrides=(), stage="auto", resume=None):
    cfg = load_config(config_path, overrides)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    stage1_dir = out_dir / "stage1_image"
    feature_dir = stage1_dir / "features"
    if _needs_image_stage(cfg):
        if stage in ("auto", "image") and not (stage == "auto" and (stage1_dir / "best.pt").exists()):
            print(f"== stage 1: per-frame image branch -> {stage1_dir}", flush=True)
            train_image.run_cfg(cfg, stage1_dir, resume if stage == "image" else None)
        elif stage == "auto":
            print(f"== stage 1: reusing {stage1_dir / 'best.pt'} (pass --stage image to retrain)", flush=True)
        if stage == "image":
            return stage1_dir / "best.pt"
        if stage in ("auto", "features") and not (stage == "auto" and (feature_dir / "meta.json").exists()):
            print(f"== caching image-branch outputs for every frame -> {feature_dir}", flush=True)
            extract_features.run(stage1_dir / "best.pt", feature_dir, cfg=cfg)
        elif stage == "auto":
            print(f"== features: reusing {feature_dir}", flush=True)
        if stage == "features":
            return feature_dir
        cfg.model.image_features = str(feature_dir)
        cfg.model.stage1_checkpoint = str(stage1_dir / "best.pt")
    print(f"== stage 2: multimodal temporal model ({cfg.model.modalities}) -> {out_dir}", flush=True)
    return train_full.run_cfg(cfg, out_dir, resume)
