import argparse
import json
from pathlib import Path

import numpy as np
import torch

from meposer.engine.build import full_model_from_checkpoint, sequences_for
from meposer.engine.common import load_checkpoint
from meposer.engine.infer import evaluate_sequences


def main():
    p = argparse.ArgumentParser(description="re-evaluate every fold of K-fold runs (last.pt) with and without the test-time wrist IK")
    p.add_argument("runs", nargs="+", help="name=dir")
    p.add_argument("--out", default="reports/results/crossval_refine.md")
    a = p.parse_args()
    torch.set_num_threads(6)
    rows, res = [], {}
    for item in a.runs:
        name, _, d = item.partition("=")
        folds = sorted(Path(d).glob("fold*/last.pt"))
        per = {False: {}, True: {}}
        for ck in folds:
            ckpt = load_checkpoint(ck)
            for refine in (False, True):
                model, cfg = full_model_from_checkpoint(ckpt, ["device=cpu", f"refine_wrists={str(refine).lower()}"])
                model.eval()
                names, per_seq, _ = evaluate_sequences(model, sequences_for(cfg, "val"), torch.device("cpu"), cfg)
                for n, m in zip(names, per_seq):
                    per[refine][n] = m
            print(name, ck.parent.name, "done", flush=True)
        res[name] = per
    subs = sorted(set().union(*[set(d[False]) for d in res.values()]))
    common = sorted(set.intersection(*[set(d[False]) for d in res.values()]))
    for name, per in res.items():
        for refine in (False, True):
            cells = [f"{per[refine][s]['mpjpe']:.2f}" if s in per[refine] else "-" for s in subs]
            v = np.array([per[refine][s]["mpjpe"] for s in common]); h = np.array([per[refine][s]["handpe"] for s in common])
            rows.append(f"| {name} | {'IK' if refine else '-'} | " + " | ".join(cells) + f" | {v.mean():.2f} +- {v.std():.2f} | {h.mean():.2f} |")
    table = "| model | wrist IK | " + " | ".join(subs) + " | MPJPE mean +- std (subjects common to all rows) | HandPE mean |\n|" + "---|" * (len(subs) + 4) + "\n" + "\n".join(rows)
    Path(a.out).write_text("Test-time wrist IK: shoulders/collars/elbows optimised so that the FK wrists match controller pose x one fixed offset estimated on the fold's training subjects. MPJPE cm per held-out subject, last epoch of each fold.\n\n" + table + "\n")
    json.dump({k: {str(r): v for r, v in d.items()} for k, d in res.items()}, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print(table)


if __name__ == "__main__":
    main()
