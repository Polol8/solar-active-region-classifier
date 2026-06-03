"""
Solar Active Region Detector — main pipeline entry point.

Runs the full pipeline in sequence:
    1. download    — Fetch HMI magnetograms from JSOC
    2. preprocess  — Convert FITS → normalised PNG
    3. labels      — Query HEK and generate YOLO label files
    4. dataset     — Split into train / val / test
    5. train       — Fine-tune YOLOv11
    6. evaluate    — Report mAP on the test split

Individual steps can be run in isolation via their own modules:
    uv run python -m src.download   --help
    uv run python -m src.preprocess --help
    uv run python -m src.labels     --help
    uv run python -m src.dataset    --help
    uv run python -m src.train      --help
    uv run python -m src.predict    --help

Usage (full pipeline):
    uv run python main.py --email you@example.com \\
                          --start 2014-01-01 --end 2014-06-30

Usage (training only, data already prepared):
    uv run python main.py --skip-download --skip-preprocess --skip-labels --skip-split \\
                          --email dummy@x.com --start 2014-01-01 --end 2014-01-01
"""

import argparse

from src import log as rlog

_STAGE_NAMES = ["Download", "Preprocess", "Labels", "Split", "Train", "Evaluate"]


def run(args):
    rlog.banner()

    tracker = rlog.PipelineTracker(_STAGE_NAMES)

    # Mark skipped stages up front so they appear in the first panel render
    if args.skip_download:
        tracker.skip(0)
    if args.skip_preprocess:
        tracker.skip(1)
    if args.skip_labels:
        tracker.skip(2)
    if args.skip_split:
        tracker.skip(3)

    # ── 1 Download ────────────────────────────────────────────────────────────
    if not args.skip_download:
        rlog.stage(1, "Download")
        tracker.start(0)
        from src.download import download
        download(
            start=args.start, end=args.end, email=args.email,
            output_dir=args.raw_dir, cadence_hours=args.cadence,
        )
        tracker.done(0)

    # ── 2 Preprocess ──────────────────────────────────────────────────────────
    if not args.skip_preprocess:
        rlog.stage(2, "Preprocess  FITS → PNG")
        tracker.start(1)
        from src.preprocess import process_directory
        process_directory(args.raw_dir, args.images_dir)
        tracker.done(1)

    # ── 3 Labels ──────────────────────────────────────────────────────────────
    if not args.skip_labels:
        rlog.stage(3, "Labels  HEK → YOLO")
        tracker.start(2)
        from src.labels import generate_all_labels
        generate_all_labels(args.images_dir, args.labels_dir)
        tracker.done(2)

    # ── 4 Dataset split ───────────────────────────────────────────────────────
    if not args.skip_split:
        rlog.stage(4, "Dataset split")
        tracker.start(3)
        from src.dataset import split_dataset
        split_dataset(args.images_dir, args.labels_dir, args.datasets_dir)
        tracker.done(3)

    # ── 5 Train ───────────────────────────────────────────────────────────────
    rlog.stage(5, "Train")
    tracker.start(4)
    from src.train import train
    best_weights = train(
        data_yaml=args.data_yaml, model_weights=args.weights,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        name=args.run_name,
    )
    tracker.done(4)

    # ── 6 Evaluate ────────────────────────────────────────────────────────────
    rlog.stage(6, "Evaluate")
    tracker.start(5)
    from src.train import evaluate
    evaluate(args.data_yaml, best_weights, imgsz=args.imgsz, split="test")
    tracker.done(5)

    rlog.console.print()
    rlog.success("[bold]Pipeline complete.[/bold]")


def _parse_args():
    p = argparse.ArgumentParser(description="Solar AR Detection — full pipeline")

    p.add_argument("--email",   required=True, help="Email registered with JSOC")
    p.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    p.add_argument("--cadence", type=float, default=6.0)

    p.add_argument("--raw-dir",      default="data/raw")
    p.add_argument("--images-dir",   default="data/images")
    p.add_argument("--labels-dir",   default="data/labels")
    p.add_argument("--datasets-dir", default="datasets")
    p.add_argument("--data-yaml",    default="configs/solar.yaml")

    p.add_argument("--weights",  default="yolo11m.pt")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--imgsz",    type=int, default=1024)
    p.add_argument("--batch",    type=int, default=8)
    p.add_argument("--run-name", default="solar_ar",
                   help="Name for the training run directory under runs/")

    p.add_argument("--skip-download",   action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-labels",     action="store_true")
    p.add_argument("--skip-split",      action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
