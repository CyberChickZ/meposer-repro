import json
from pathlib import Path

import torch

from ..metrics.evaluate import format_table
from .build import full_model_from_checkpoint, sequences_for
from .common import load_checkpoint, resolve_device
from .infer import evaluate_sequences


def run(ckpt_path, split="val", out=None, overrides=()):
    ckpt = load_checkpoint(ckpt_path)
    model, cfg = full_model_from_checkpoint(ckpt, overrides)
    device = resolve_device(cfg.device)
    torch.set_num_threads(cfg.get("threads", 10))
    model = model.to(device).eval()
    seqs = sequences_for(cfg, split)
    names, per_seq, agg = evaluate_sequences(model, seqs, device, cfg)
    table = format_table(per_seq + [agg["frame_weighted"], agg["mean_over_sequences"]], names + ["frame-weighted", "mean-over-seq"])
    print(table)
    out = Path(out) if out else Path("runs/eval") / f"{Path(ckpt_path).parent.name}_{Path(ckpt_path).stem}_{split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"), "split": split, "subjects": names, "per_sequence": dict(zip(names, per_seq)),
              "aggregate": agg, "fps": cfg.data.fps, "protocol": "head-joint aligned (HMD-Poser), 22 SMPL joints, per-frame betas, causal full-sequence inference"}
    with open(str(out) + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(str(out) + ".md", "w") as f:
        f.write(table + "\n")
    print(f"wrote {out}.json and {out}.md")
    return result
