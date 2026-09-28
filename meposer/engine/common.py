import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..config import load_config, save_config


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    assert name in ("cpu", "cuda", "mps"), f"unsupported device {name!r}; use cpu, cuda, mps or auto"
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_run(config_path, out_dir, overrides):
    return setup_run_cfg(load_config(config_path, overrides), out_dir)


def _git(*args):
    try:
        import subprocess
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _cpu_name():
    try:
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3).stdout.strip() or platform_processor()
    except Exception:
        return platform_processor()


def platform_processor():
    import platform
    return platform.processor()


def write_run_info(out_dir, **extra):
    path = Path(out_dir) / "run_info.json"
    info = json.load(open(path)) if path.exists() else {}
    info.update(extra)
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return info


def setup_run_cfg(cfg, out_dir):
    import platform
    import sys
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.get("threads", 10))
    device = resolve_device(cfg.device)
    write_run_info(out_dir, command=" ".join(sys.argv), git_commit=_git("rev-parse", "--short", "HEAD"), git_dirty=bool(_git("status", "--porcelain")),
                   device=str(device), torch=torch.__version__, platform=platform.platform(), machine=platform.machine(), cpu=_cpu_name(),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"), seed=cfg.seed, train_subjects=list(cfg.data.train_subjects), val_subjects=list(cfg.data.val_subjects))
    return cfg, out_dir, device


def heatmap_size(cfg):
    w, h = cfg.data.image_size
    s = cfg.data.heatmap_stride
    return (w // s, h // s)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_optimizer(params, cfg):
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.milestones), gamma=cfg.gamma)
    return opt, sched


def save_checkpoint(path, **payload):
    torch.save(payload, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class RunLogger:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.log_path = self.out_dir / "log.txt"
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.t0 = time.time()

    def info(self, msg):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:7.0f}s] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def record(self, **row):
        row["time"] = time.time() - self.t0
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _scalar(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def fmt(parts):
    return " ".join(f"{k}={_scalar(v):.4f}" for k, v in parts.items())


def env_info():
    return {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
    }


def optimizer_step(loss, model, opt, grad_clip=0.0):
    if not torch.isfinite(loss):
        return False
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return True
