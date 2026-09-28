import argparse


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _add_set(sp):
    sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any config key, e.g. train.lr=1e-4 'model.modalities=[imu]'")


def main(argv=None):
    p = argparse.ArgumentParser(prog="meposer", description="MEPoser reproduction: prepare -> train -> evaluate -> infer")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="raw EMHI subset -> per-subject npz + image memmaps, camera calibration, data report")
    sp.add_argument("--raw", default="data/raw")
    sp.add_argument("--out", default="data/processed")
    sp.add_argument("--image-size", type=_size, default="320x240")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--subjects", nargs="*")
    sp.add_argument("--calib", default="assets/calib/calib.json")
    sp.add_argument("--recalibrate", action="store_true", help="refit the fisheye calibration even if --calib exists")
    sp.add_argument("--smpl", default="assets/smpl/SMPL_NEUTRAL.npz")

    sp = sub.add_parser("inspect", help="fixed dataset-inspection procedure: inventory, preview video, statistics, physics checks, cameras, paper checklist")
    sp.add_argument("--processed", default="data/processed")
    sp.add_argument("--out", default="data/processed/inspection")
    sp.add_argument("--smpl", default="assets/smpl/SMPL_NEUTRAL.npz")
    sp.add_argument("--subjects", nargs="*")

    sp = sub.add_parser("train", help="run every training stage the config needs (image branch -> feature cache -> temporal model), skipping finished ones")
    sp.add_argument("-c", "--config", required=True)
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--stage", choices=["auto", "image", "features", "full"], default="auto", help="force a single stage (default: run what is missing)")
    sp.add_argument("--resume", help="checkpoint (last.pt) to resume the final stage from")
    _add_set(sp)

    sp = sub.add_parser("crossval", help="K-fold cross-validation by subject: retrains every stage per fold and aggregates per-subject metrics")
    sp.add_argument("-c", "--config", required=True)
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--fold", type=int, help="run a single fold index")
    sp.add_argument("--aggregate-only", action="store_true")
    _add_set(sp)

    sp = sub.add_parser("ledger", help="summarise every run under runs/ (config diff vs default, split, seed, device, runtime, best epoch, metrics)")
    sp.add_argument("--runs", default="runs")
    sp.add_argument("--base", default="configs/default.yaml")
    sp.add_argument("--out", default="reports/results/EXPERIMENTS.md")

    sp = sub.add_parser("evaluate", help="sequence-level protocol metrics for a checkpoint")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--split", default="val", choices=["val", "train"])
    sp.add_argument("--out", help="output prefix (default: runs/eval/<run>_<ckpt>_<split>)")
    _add_set(sp)

    sp = sub.add_parser("infer", help="predict per-frame SMPL pose/shape/translation + joints for one subject -> npz")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--subject", required=True)
    sp.add_argument("--out", required=True)
    _add_set(sp)

    sp = sub.add_parser("visualize", help="qualitative figure + per-frame error curve for one subject")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--subject", required=True)
    sp.add_argument("--frames", nargs="*", type=int)
    sp.add_argument("--out", default="reports/figures")
    sp.add_argument("--video", type=float, default=0, metavar="SECONDS", help="also render an mp4: stereo frames with GT/pred 2D joints, 3D GT vs pred skeleton, error curve")
    sp.add_argument("--video-start", type=int, help="first frame id of the clip (default: sequence start)")
    _add_set(sp)

    args = p.parse_args(argv)
    if args.cmd == "prepare":
        from .data.prepare import prepare
        prepare(args.raw, args.out, args.image_size, args.workers, args.subjects, args.calib, args.smpl, args.recalibrate)
    elif args.cmd == "inspect":
        from .data.inspect import inspect
        print(inspect(args.processed, args.out, args.smpl, args.subjects))
    elif args.cmd == "train":
        from .engine.pipeline import run
        run(args.config, args.out, args.set, args.stage, args.resume)
    elif args.cmd == "crossval":
        from .engine.crossval import aggregate, run
        aggregate(args.out) if args.aggregate_only else run(args.config, args.out, args.set, args.folds, args.fold)
    elif args.cmd == "ledger":
        from .engine.ledger import build_ledger
        out, n = build_ledger(args.runs, args.base, args.out)
        print(f"{n} runs -> {out}")
    elif args.cmd == "evaluate":
        from .engine.evaluate import run
        run(args.ckpt, args.split, args.out, args.set)
    elif args.cmd == "infer":
        from .engine.predict import run
        run(args.ckpt, args.subject, args.out, args.set)
    elif args.cmd == "visualize":
        from .engine.visualize import run
        run(args.ckpt, args.subject, args.frames, args.out, args.set, args.video, args.video_start)


if __name__ == "__main__":
    main()
