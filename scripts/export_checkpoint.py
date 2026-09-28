import argparse
from pathlib import Path

import torch

KEEP = {"kind", "cfg", "model", "heatmap_size", "imu_mean", "imu_std", "epoch", "val", "val_per_sequence", "best"}


def main():
    p = argparse.ArgumentParser(description="strip optimizer/scheduler state from a checkpoint for distribution")
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    slim = {k: v for k, v in ck.items() if k in KEEP}
    import os
    slim["source_run"] = os.path.relpath(Path(a.src).resolve().parent, Path.cwd())
    if slim.get("kind") == "full" and slim["cfg"]["model"].get("image_features"):
        slim["cfg"]["model"]["image_features"] = "checkpoints/image_features"
        slim["cfg"]["model"]["stage1_checkpoint"] = "checkpoints/stage1_image.pt"
    info = Path(a.src).parent / "run_info.json"
    if info.exists():
        import json
        slim["run_info"] = json.load(open(info))
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, a.dst)
    print(f"{a.src} ({Path(a.src).stat().st_size / 1e6:.1f} MB) -> {a.dst} ({Path(a.dst).stat().st_size / 1e6:.1f} MB), epoch {ck.get('epoch')}")


if __name__ == "__main__":
    main()
