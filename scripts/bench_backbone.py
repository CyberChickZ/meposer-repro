import argparse
import time
from pathlib import Path

import torch
from torchvision.models import regnet_y_400mf


def bench(device, bs, res, steps=3):
    m = regnet_y_400mf(weights=None)
    net = torch.nn.Sequential(m.stem, m.trunk_output).to(device)
    opt = torch.optim.Adam(net.parameters())
    x = torch.randn(bs, 3, *res, device=device)
    sync = torch.mps.synchronize if device == "mps" else (torch.cuda.synchronize if device == "cuda" else (lambda: None))
    mem = None
    for i in range(steps + 1):
        sync(); t = time.time()
        if device == "mps" and i == steps:
            base = torch.mps.current_allocated_memory()
        out = net(x); loss = out.mean()
        if device == "mps" and i == steps:
            sync(); mem = (torch.mps.current_allocated_memory() - base) / 1e6 / bs
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); sync()
        dt = time.time() - t
    return bs / dt, mem


def main():
    p = argparse.ArgumentParser(description="training throughput and activation memory of the RegNetY-400MF backbone on this machine")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="reports/results/benchmark.md")
    a = p.parse_args()
    rows = ["| device | resolution | batch | img/s (fwd+bwd+Adam) | forward activations MB per image |", "|---|---|---|---|---|"]
    for bs, res in [(64, (240, 320)), (16, (240, 320)), (32, (480, 640)), (8, (480, 640))]:
        try:
            ips, mem = bench(a.device, bs, res)
            rows.append(f"| {a.device} | {res[1]}x{res[0]} | {bs} | {ips:.0f} | {'-' if mem is None else f'{mem:.0f}'} |")
            print(rows[-1], flush=True)
        except RuntimeError as e:
            rows.append(f"| {a.device} | {res[1]}x{res[0]} | {bs} | failed: {str(e)[:60]} | |")
    Path(a.out).write_text(f"Measured by `scripts/bench_backbone.py` (torch {torch.__version__}); memory = tensors allocated during the forward pass and kept for the backward pass (activations), divided by the batch.\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
