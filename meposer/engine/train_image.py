import time

import numpy as np
import torch

from ..config import load_config
from torch.utils.data import DataLoader

from ..data.dataset import FrameDataset
from ..losses import MEPoserLoss
from ..models.image_branch import StereoImageBranch
from ..models.meposer import align_to_world
from .build import sequences_for
from .common import RunLogger, _scalar, build_optimizer, env_info, fmt, heatmap_size, load_checkpoint, optimizer_step, save_checkpoint, setup_run_cfg, to_device, write_run_info


def _forward(model, batch, amp=False):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and batch["images"].is_cuda):
        out = model(batch["images"])
    out = {k: v.float() for k, v in out.items()}
    out["joints_global"] = align_to_world(out["joints_local"], batch["T_world_cam0"], batch["head_pos"])
    return out


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg=None):
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = _forward(model, batch, bool(cfg.model.image.get("amp", False)) if cfg else False)
        _, parts = loss_fn(out, batch)
        b = batch["images"].shape[0]
        parts["local_mpjpe_cm"] = (out["joints_local"] - batch["joints_local"]).norm(dim=-1).mean() * 100
        parts["global_mpjpe_cm"] = (out["joints_global"] - (batch["joints_world"] - batch["head_pos"][:, None])).norm(dim=-1).mean() * 100
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * b
        n += b
    model.train()
    out = {k: v / n for k, v in sums.items()}
    if not all(np.isfinite(v) for v in out.values()):
        print("WARNING: non-finite validation values; on MPS this has been observed with freshly initialised models, "
              "re-run with --set device=cpu to check", flush=True)
    return out


def run(config_path, out_dir, overrides=(), resume=None):
    return run_cfg(load_config(config_path, overrides), out_dir, resume)


def run_cfg(cfg, out_dir, resume=None):
    cfg, out_dir, device = setup_run_cfg(cfg, out_dir)
    log = RunLogger(out_dir)
    log.info(f"stage1 image branch | device={device} | {env_info()}")
    tcfg = cfg.train_image
    hs = heatmap_size(cfg)
    train_seqs = sequences_for(cfg, "train")
    val_seqs = sequences_for(cfg, "val")
    train_ds = FrameDataset(train_seqs, cfg.data.frame_stride, hs, cfg.data.heatmap_sigma)
    val_ds = FrameDataset(val_seqs, cfg.data.get("val_frame_stride", 10), hs, cfg.data.heatmap_sigma)
    log.info(f"train frames={len(train_ds)} (stride {cfg.data.frame_stride}) val frames={len(val_ds)} heatmap={hs}")
    train_loader = DataLoader(train_ds, tcfg.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True, persistent_workers=cfg.data.num_workers > 0)
    val_loader = DataLoader(val_ds, tcfg.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    model = StereoImageBranch(cfg.model.image, hs).to(device)
    loss_fn = MEPoserLoss(cfg.loss)
    opt, sched = build_optimizer(model.parameters(), tcfg)
    start_epoch, best = 0, float("inf")
    if resume:
        ck = load_checkpoint(resume)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", best)
        log.info(f"resumed from {resume} at epoch {start_epoch}")
    log.info(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    step, skipped = 0, 0
    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            out = _forward(model, batch, bool(cfg.model.image.get("amp", False)))
            loss, parts = loss_fn(out, batch)
            if not optimizer_step(loss, model, opt, tcfg.get("grad_clip", 0.0)):
                skipped += 1
                log.info(f"ep {epoch} it {i}: non-finite loss, step skipped (total skipped {skipped})")
                continue
            step += 1
            if i % tcfg.log_every == 0:
                rate = (i + 1) * tcfg.batch_size * 2 / (time.time() - t0)
                log.info(f"ep {epoch} it {i}/{len(train_loader)} lr={opt.param_groups[0]['lr']:.1e} {fmt(parts)} img/s={rate:.1f}")
                log.record(split="train", epoch=epoch, step=step, **{k: _scalar(v) for k, v in parts.items()})
        sched.step()
        val = validate(model, val_loader, loss_fn, device, cfg)
        log.info(f"ep {epoch} VAL {fmt(val)} epoch_time={time.time() - t0:.0f}s")
        log.record(split="val", epoch=epoch, step=step, **val)
        payload = dict(kind="image", cfg=cfg.to_dict(), model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), heatmap_size=hs, epoch=epoch, val=val, best=best)
        save_checkpoint(out_dir / "last.pt", **payload)
        if val["local_mpjpe_cm"] < best:
            best = val["local_mpjpe_cm"]
            payload["best"] = best
            save_checkpoint(out_dir / "best.pt", **payload)
            log.info(f"new best local_mpjpe_cm={best:.3f} -> best.pt")
    write_run_info(out_dir, finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_seconds=round(time.time() - log.t0), epochs=tcfg.epochs,
                   best_metric="local_mpjpe_cm", best_value=best, selected_checkpoint=str(out_dir / "best.pt"), skipped_steps=skipped)
    log.info("done")
    return out_dir / "best.pt"
