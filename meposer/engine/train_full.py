import time

import torch

from ..config import load_config
from torch.utils.data import DataLoader

from ..data.dataset import WindowDataset, compute_feature_stats
from ..losses import MEPoserLoss
from ..metrics.evaluate import format_table
from .build import build_full_model, sequences_for
from .common import RunLogger, _scalar, build_optimizer, env_info, fmt, heatmap_size, load_checkpoint, optimizer_step, save_checkpoint, setup_run_cfg, to_device, write_run_info
from .infer import evaluate_sequences


def run(config_path, out_dir, overrides=(), resume=None):
    return run_cfg(load_config(config_path, overrides), out_dir, resume)


def run_cfg(cfg, out_dir, resume=None):
    cfg, out_dir, device = setup_run_cfg(cfg, out_dir)
    log = RunLogger(out_dir)
    log.info(f"stage2 full model | device={device} | modalities={cfg.model.modalities} | cached_image_features={cfg.model.image_features} | {env_info()}")
    tcfg = cfg.train
    train_seqs = sequences_for(cfg, "train")
    val_seqs = sequences_for(cfg, "val")
    stats = compute_feature_stats(train_seqs)
    model = build_full_model(cfg, stats=stats).to(device)
    use_images = model.use_image and not model.cached_image_features
    hs = heatmap_size(cfg)
    train_ds = WindowDataset(train_seqs, cfg.data.window, cfg.data.window_stride, use_images, hs, cfg.data.heatmap_sigma,
                             augment=cfg.data.get("augment"), feature_names=list(cfg.data.imu_features))
    log.info(f"train windows={len(train_ds)} window={cfg.data.window} stride={cfg.data.window_stride} use_images={use_images} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    loader = DataLoader(train_ds, tcfg.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True, persistent_workers=cfg.data.num_workers > 0)
    loss_fn = MEPoserLoss(cfg.loss, image_joints_are_inputs=model.cached_image_features)
    scale = tcfg.get("backbone_lr_scale", 1.0)
    img = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("image_branch.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("image_branch.")]
    groups = [{"params": rest}] + ([{"params": img, "lr": tcfg.lr * scale}] if img else [])
    opt, sched = build_optimizer(groups, tcfg)
    accum = int(tcfg.get("accum_steps", 1))
    freeze_bn = bool(cfg.model.image.get("freeze_bn", False)) and model.image_branch is not None
    log.info(f"optimizer groups: rest lr={tcfg.lr}, image_branch lr={tcfg.lr * scale if img else None}; accum_steps={accum}; freeze_bn={freeze_bn}")
    start_epoch, best = 0, float("inf")
    if resume:
        ck = load_checkpoint(resume)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", best)
        log.info(f"resumed from {resume} at epoch {start_epoch}")
    step, skipped = 0, 0
    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        if freeze_bn:
            for m in model.image_branch.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            out = model(batch)
            loss, parts = loss_fn(out, batch)
            if accum == 1:
                if not optimizer_step(loss, model, opt, tcfg.get("grad_clip", 0.0)):
                    skipped += 1
                    log.info(f"ep {epoch} it {i}: non-finite loss, step skipped (total skipped {skipped})")
                    continue
            else:
                if not torch.isfinite(loss):
                    skipped += 1
                    log.info(f"ep {epoch} it {i}: non-finite loss, micro-batch skipped (total skipped {skipped})")
                    continue
                (loss / accum).backward()
                if (i + 1) % accum == 0:
                    if tcfg.get("grad_clip", 0.0):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            step += 1
            if i % tcfg.log_every == 0:
                log.info(f"ep {epoch} it {i}/{len(loader)} lr={opt.param_groups[0]['lr']:.1e} {fmt(parts)}")
                log.record(split="train", epoch=epoch, step=step, **{k: _scalar(v) for k, v in parts.items()})
        sched.step()
        train_time = time.time() - t0
        payload = dict(kind="full", cfg=cfg.to_dict(), model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                       imu_mean=stats[0], imu_std=stats[1], heatmap_size=hs, epoch=epoch, best=best)
        if (epoch + 1) % tcfg.val_every == 0 or epoch + 1 == tcfg.epochs:
            names, per_seq, agg = evaluate_sequences(model, val_seqs, device, cfg)
            val = agg["frame_weighted"]
            log.info(f"ep {epoch} VAL mpjpe={val['mpjpe']:.2f}cm pa={val['pa_mpjpe']:.2f} mpjre={val['mpjre']:.2f}deg jitter={val['pred_jitter']:.1f}/{val['gt_jitter']:.1f} train_time={train_time:.0f}s\n" + format_table(per_seq, names))
            log.record(split="val", epoch=epoch, step=step, **val)
            payload["val"] = val
            payload["val_per_sequence"] = dict(zip(names, per_seq))
            if val["mpjpe"] < best:
                best = val["mpjpe"]
                payload["best"] = best
                save_checkpoint(out_dir / "best.pt", **payload)
                log.info(f"new best mpjpe={best:.3f}cm -> best.pt")
        save_checkpoint(out_dir / "last.pt", **payload)
    write_run_info(out_dir, finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_seconds=round(time.time() - log.t0), epochs=tcfg.epochs,
                   best_metric="mpjpe", best_value=best, selected_checkpoint=str(out_dir / "best.pt"), skipped_steps=skipped)
    log.info("done")
    return out_dir / "best.pt"
