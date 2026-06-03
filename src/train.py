"""
Fine-tune YOLOv11 on the solar active-region dataset.

Model choice — YOLOv11m (medium)
  The medium variant (20 M parameters) provides a good balance between
  detection accuracy and training speed on CPU.  Switch to 'yolo11n.pt'
  (nano, 2.6 M params) for faster experiments or 'yolo11l.pt' (large,
  25 M params) if GPU memory allows and higher mAP is required.

Transfer learning from COCO
  Starting from COCO-pretrained weights means the backbone already knows
  how to detect edges, textures and blobs — features that transfer well to
  magnetogram bright/dark patch detection — so the network only needs to
  adapt its head to four solar-specific classes.

Augmentation choices specific to magnetograms
  Standard image-augmentation practices must be reconsidered for magnetograms:

  fliplr=0 (horizontal flip disabled)
    Mirroring reverses the east–west orientation of the solar disk.  Because
    leading sunspots in a bipolar group tend to be closer to the equator and
    have the same polarity as the dominant hemisphere polarity (Joy's Law and
    Hale's Polarity Law), a horizontally flipped magnetogram would represent
    a physically impossible or opposite-hemisphere active region.

  flipud=0 (vertical flip disabled)
    Flipping vertically swaps solar north and south, inverting the statistical
    latitude tilt of bipolar groups (Joy's Law).  This would confuse the model
    about the north–south orientation of the solar disk.

  hsv_h=0, hsv_s=0 (no hue/saturation shift)
    Magnetograms are single-channel greyscale images.  Hue and saturation
    augmentations are meaningless and would produce invalid data.

  hsv_v=0.02 (slight brightness jitter retained)
    Small brightness variations simulate day-to-day differences in HMI
    calibration and solar irradiance without distorting the magnetic polarity.

  cos_lr=True (cosine learning-rate schedule)
    Cosine annealing decays the learning rate smoothly from lr0 to near-zero,
    which typically yields better final accuracy than a step decay schedule for
    fine-tuning tasks with relatively few epochs.

  patience=20 (early stopping)
    If the validation mAP does not improve for 20 consecutive epochs, training
    stops automatically.  Prevents overfitting when the dataset is small (a
    1-month run produces only ~80 training images).

Usage:
    python -m src.train --data configs/solar.yaml --epochs 100 --batch 8
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from src import log as rlog

# Resolve the repository root so the 'runs/' directory is always written
# next to this repository, regardless of the working directory from which
# the script is invoked (e.g. from inside a virtual environment).
_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL  = "yolo11m.pt"   # YOLOv11 medium — COCO pretrained
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ  = 1024           # matches IMAGE_SIZE in preprocess.py
DEFAULT_BATCH  = 8


def train(
    data_yaml: str,
    model_weights: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    project: str = "runs",
    name: str = "solar_ar",
    resume: bool = False,
):
    """Fine-tune YOLOv11 on the solar active-region dataset.

    Returns the absolute path to the best checkpoint (best.pt).
    """
    # Always use an absolute project path so YOLO does not accidentally write
    # checkpoints inside the virtual environment directory (a known YOLO quirk
    # when the current working directory is not the project root)
    abs_project = str((_REPO_ROOT / project).resolve())

    # Detect available hardware and warn early if only CPU is available
    if torch.cuda.is_available():
        device_label = f"CUDA  ({torch.cuda.get_device_name(0)})"
    else:
        device_label = "CPU only  —  training will be slow; consider a GPU or reduce --imgsz/--epochs"
        rlog.warn("No CUDA GPU detected. Running on CPU.")

    rlog.kv_table([
        ("Model",    f"{model_weights}  (COCO pretrained)"),
        ("Device",   device_label),
        ("Dataset",  f"{data_yaml}  ·  4 classes"),
        ("Config",   f"{epochs} epochs  ·  batch {batch}  ·  {imgsz} px"),
        ("Augment",  "fliplr=0  flipud=0  (polarity semantics preserved)"),
        ("Output",   f"{abs_project}/{name}/"),
    ])

    model = YOLO(model_weights)
    model.train(
        data       = data_yaml,
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        project    = abs_project,
        name       = name,
        resume     = resume,
        # --- Magnetogram-specific augmentation settings (see module docstring) ---
        fliplr     = 0.0,   # no horizontal flip — would invert polarity layout
        flipud     = 0.0,   # no vertical flip   — would invert Joy's Law tilt
        hsv_h      = 0.0,   # no hue shift       — greyscale image, meaningless
        hsv_s      = 0.0,   # no saturation shift — same reason
        hsv_v      = 0.02,  # tiny brightness jitter — safe for calibration noise
        # --- Training stability ---
        cos_lr     = True,  # cosine LR schedule (smoother convergence)
        patience   = 20,    # early stopping after 20 non-improving epochs
        save_period= 10,    # checkpoint every 10 epochs (for long runs)
        plots      = True,  # save confusion matrix, PR curve, etc.
    )

    best_ckpt = Path(abs_project) / name / "weights" / "best.pt"
    rlog.success(f"Best weights saved → [cyan]{best_ckpt}[/cyan]")
    return str(best_ckpt)


def evaluate(data_yaml: str, weights: str, imgsz: int = DEFAULT_IMGSZ, split: str = "test"):
    """Run YOLO validation on the given split and print mAP metrics."""
    rlog.kv_table([
        ("Weights", weights),
        ("Split",   split),
    ])

    model   = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split)

    # mAP50    — mean Average Precision at IoU threshold 0.50
    #            (standard for solar/remote-sensing detection tasks)
    # mAP50-95 — averaged over IoU thresholds 0.50–0.95 in steps of 0.05
    #            (COCO-style metric, stricter on localisation precision)
    rlog.metrics_table([
        ("mAP50",    f"{metrics.box.map50:.4f}"),
        ("mAP50-95", f"{metrics.box.map:.4f}"),
    ])
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11 on solar active-region data")
    p.add_argument("--data",    default="configs/solar.yaml")
    p.add_argument("--weights", default=DEFAULT_MODEL)
    p.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    p.add_argument("--project", default="runs")
    p.add_argument("--name",    default="solar_ar")
    p.add_argument("--resume",  action="store_true",
                   help="Resume training from the last checkpoint")
    p.add_argument("--eval",    action="store_true",
                   help="Only evaluate (skip training); requires --weights pointing to a trained model")
    p.add_argument("--split",   default="test", choices=["train", "val", "test"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.eval:
        evaluate(args.data, args.weights, imgsz=args.imgsz, split=args.split)
    else:
        train(
            args.data,
            model_weights = args.weights,
            epochs        = args.epochs,
            imgsz         = args.imgsz,
            batch         = args.batch,
            project       = args.project,
            name          = args.name,
            resume        = args.resume,
        )
