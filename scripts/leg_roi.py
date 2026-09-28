"""Adaptive lower-body ROI: crop the legs from the native 640x480 fisheye images, localise 8 leg joints at higher
resolution, and write full-image peaks that the stereo 2D fit can use (--set refine_2d_features=<out>).

train:  crops centred on the GT leg keypoints with random shift/scale (simulating an imperfect prediction)
infer:  crops centred on the leg joints of a stage-2 model's prediction, projected with the headset-derived camera pose
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import RegNet_Y_400MF_Weights, regnet_y_400mf

LEG = [1, 2, 4, 5, 7, 8, 10, 11]
CROP, HM = 128, 32
MEAN, STD = 0.45, 0.225


def roi_from_points(uv, rng=None, jitter=0.0):
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.3 + 24
    if rng is not None and jitter:
        cx += rng.normal() * jitter * side
        cy += rng.normal() * jitter * side
        side *= float(np.exp(rng.normal() * jitter))
    side = float(np.clip(side, 64, 480))
    return cx, cy, side


def crop(img, cx, cy, side):
    s = CROP / side
    m = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float32)
    return cv2.warpAffine(img, m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0), m


class LegNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2 if pretrained else None)
        conv = net.stem[0]
        mono = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        mono.weight.data.copy_(conv.weight.data.sum(1, keepdim=True))
        net.stem[0] = mono
        self.stem, self.b1, self.b2, self.b3 = net.stem, net.trunk_output.block1, net.trunk_output.block2, net.trunk_output.block3
        self.l1, self.l2, self.l3 = nn.Conv2d(48, 64, 1), nn.Conv2d(104, 64, 1), nn.Conv2d(208, 64, 1)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, len(LEG), 1))

    def forward(self, x):
        x = (x.float() / 255.0 - MEAN) / STD
        c1 = self.b1(self.stem(x))
        c2 = self.b2(c1)
        c3 = self.b3(c2)
        up = lambda t, ref: nn.functional.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        p = self.l1(c1) + up(self.l2(c2), c1) + up(self.l3(c3), c1)
        return self.head(p)


def render(uv_crop, vis, sigma=1.0):
    ys, xs = np.mgrid[0:HM, 0:HM].astype(np.float32)
    c = uv_crop * (HM / CROP)
    hm = np.exp(-((xs[None] - c[:, 0, None, None]) ** 2 + (ys[None] - c[:, 1, None, None]) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    hm[~vis] = 0
    return hm


class CropSet(Dataset):
    def __init__(self, processed, subjects, jitter=0.15, stride=1):
        self.items, self.imgs, self.p2d = [], {}, {}
        for s in subjects:
            d = Path(processed) / s
            meta = json.load(open(d / "meta.json"))
            self.imgs[s] = np.load(d / meta["images"], mmap_mode="r")
            p = np.load(d / "annots.npz")["p2d"][:, :, LEG]
            self.p2d[s] = p
            for i in range(0, len(p), stride):
                for c in range(2):
                    v = (p[i, c, :, 0] >= 0) & (p[i, c, :, 0] < 640) & (p[i, c, :, 1] >= 0) & (p[i, c, :, 1] < 480)
                    if v.sum() >= 2:
                        self.items.append((s, i, c))
        self.jitter = jitter

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s, i, c = self.items[k]
        rng = np.random.default_rng()
        p = self.p2d[s][i, c]
        v = (p[:, 0] >= 0) & (p[:, 0] < 640) & (p[:, 1] >= 0) & (p[:, 1] < 480)
        cx, cy, side = roi_from_points(p[v], rng, self.jitter)
        im, m = crop(np.asarray(self.imgs[s][i, c]), cx, cy, side)
        uv = p @ m[:, :2].T + m[:, 2]
        vin = v & (uv[:, 0] >= 0) & (uv[:, 0] < CROP) & (uv[:, 1] >= 0) & (uv[:, 1] < CROP)
        return torch.from_numpy(im)[None], torch.from_numpy(render(uv, vin)), torch.from_numpy(vin)


def decode(hm):
    n, j, h, w = hm.shape
    flat = hm.reshape(n, j, -1)
    idx = flat.argmax(-1)
    conf = flat.max(-1).values
    x, y = (idx % w).float(), (idx // w).float()
    xs, ys = x.clone(), y.clone()
    for b in range(n):
        for k in range(j):
            xi, yi = int(x[b, k]), int(y[b, k])
            patch = hm[b, k, max(0, yi - 1): yi + 2, max(0, xi - 1): xi + 2].clamp_min(0)
            gy, gx = torch.meshgrid(torch.arange(max(0, yi - 1), min(h, yi + 2)), torch.arange(max(0, xi - 1), min(w, xi + 2)), indexing="ij")
            s = patch.sum()
            if s > 0:
                xs[b, k], ys[b, k] = (patch * gx).sum() / s, (patch * gy).sum() / s
    return torch.stack([xs, ys], -1) * (CROP / HM), conf


def train(a):
    dev = torch.device(a.device)
    ds = CropSet(a.processed, a.train, jitter=0.15)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    net = LegNet().to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        remap = {"stem.": "stem.", "stages.0.": "b1.", "stages.1.": "b2.", "stages.2.": "b3."}
        own = net.state_dict()
        load = {}
        for k, v in sd.items():
            for src, dst in remap.items():
                if k.startswith(src) and dst + k[len(src):] in own and own[dst + k[len(src):]].shape == v.shape:
                    load[dst + k[len(src):]] = v
        net.load_state_dict(load, strict=False)
        print(f"initialised {len(load)} tensors of the backbone from {a.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.epochs * 0.6), int(a.epochs * 0.85)], 0.1)
    for ep in range(a.epochs):
        net.train()
        tot = 0
        for i, (x, y, v) in enumerate(dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = net(x)
            loss = nn.functional.binary_cross_entropy_with_logits(out.float(), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        print(f"epoch {ep} loss {tot / (i + 1):.4f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), Path(a.out) / "legnet.pt")


@torch.no_grad()
def infer(a):
    from meposer.data.calibrate import load_calib
    from meposer.engine.build import full_model_from_checkpoint, sequences_for
    from meposer.engine.common import load_checkpoint
    from meposer.engine.infer import run_sequence
    from meposer.geometry.fisheye import project_kb4
    from meposer.geometry.se3 import transform_points
    from meposer.models.meposer import anchor_joints
    dev = torch.device(a.device)
    net = LegNet(pretrained=False).to(dev).eval()
    net.load_state_dict(torch.load(a.weights or Path(a.out) / "legnet.pt", map_location=dev))
    calib = load_calib("assets/calib/calib.json")
    model, cfg = full_model_from_checkpoint(load_checkpoint(a.ckpt), ["device=cpu"] + a.set)
    model.eval()
    out_dir = Path(a.out) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    for subj in a.val:
        cfg.data.val_subjects = [subj]
        seq = sequences_for(cfg, "val")[0]
        imgs = np.load(Path(a.processed) / subj / json.load(open(Path(a.processed) / subj / "meta.json"))["images"], mmap_mode="r")
        base = seq.image_features
        peaks = base["peaks"].astype(np.float32) * (640.0 / 40.0)
        conf = base["conf"].astype(np.float32).copy()
        for (s0, s1), out in run_sequence(model, seq, torch.device("cpu"), cfg):
            pred = anchor_joints(out["joints_fk"], torch.as_tensor(seq.head_anchor[s0:s1])).numpy()
            for c, cam in enumerate(("cam0", "cam1")):
                pc = transform_points(seq.T_cam_world[s0:s1, c].astype(np.float64), pred[:, LEG].astype(np.float64))
                uv = project_kb4(pc, calib["cameras"][cam]["params"])
                crops, ms, frames = [], [], []
                for t in range(s1 - s0):
                    ok = (pc[t, :, 2] > 0.05) & (uv[t, :, 0] > -80) & (uv[t, :, 0] < 720) & (uv[t, :, 1] > -80) & (uv[t, :, 1] < 560)
                    if ok.sum() < 2:
                        continue
                    cx, cy, side = roi_from_points(uv[t, ok])
                    im, m = crop(np.asarray(imgs[s0 + t, c]), cx, cy, side)
                    crops.append(im); ms.append(m); frames.append(s0 + t)
                for b in range(0, len(crops), 512):
                    x = torch.from_numpy(np.stack(crops[b:b + 512]))[:, None].to(dev)
                    hm = torch.sigmoid(net(x).float()).cpu()
                    pk, cf = decode(hm)
                    for r in range(len(pk)):
                        m = ms[b + r]
                        minv = cv2.invertAffineTransform(m)
                        full = pk[r].numpy() @ minv[:, :2].T + minv[:, 2]
                        fi = frames[b + r]
                        peaks[fi, c, LEG] = full
                        conf[fi, c, LEG] = cf[r].numpy()
        np.savez(out_dir / f"{subj}.npz", peaks=peaks, conf=conf)
        p2d = seq.p2d
        vis = (p2d[..., 0] >= 0) & (p2d[..., 0] < 640) & (p2d[..., 1] >= 0) & (p2d[..., 1] < 480)
        legvis = vis[:, :, LEG]
        e_new = np.linalg.norm(peaks[:, :, LEG] - p2d[:, :, LEG], axis=-1)[legvis]
        e_old = np.linalg.norm(base["peaks"][:, :, LEG] * 16.0 - p2d[:, :, LEG], axis=-1)[legvis]
        print(f"{subj}: leg-joint 2D error median {np.median(e_old):.1f} -> {np.median(e_new):.1f} px, p90 {np.percentile(e_old, 90):.1f} -> {np.percentile(e_new, 90):.1f}", flush=True)
    json.dump({"heatmap_size": [640, 480], "note": "peaks in full-image pixels; leg joints from the adaptive ROI network"}, open(out_dir / "meta.json", "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "infer"])
    p.add_argument("--processed", default="data/processed_640")
    p.add_argument("--train", nargs="*", default=["0000", "0002", "0003", "0004", "0005", "0007", "0008", "0009"])
    p.add_argument("--val", nargs="*", default=["0001", "0006"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--ckpt")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default="runs/leg_roi_fold1")
    p.add_argument("--init", help="stage-1 image-branch checkpoint to initialise the backbone from")
    p.add_argument("--weights", help="trained leg network (default: <out>/legnet.pt), e.g. checkpoints/leg_roi.pt")
    a = p.parse_args()
    train(a) if a.cmd == "train" else infer(a)
