"""
Run inference with a trained YOLOv11 model on new solar magnetograms.

Accepts:
  - A single PNG/FITS file
  - A directory of images

Output:
  - Annotated PNG images saved alongside the inputs (or to --output dir).
  - A CSV summary with per-detection class, confidence, and bounding box.

Usage:
    python -m src.predict --weights runs/solar_ar/weights/best.pt \
                          --source data/images/ --output results/
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src import log as rlog

CLASS_NAMES = ["Alpha", "Beta", "BetaGamma", "BetaGammaDelta"]
CLASS_COLORS = {
    0: (100, 200, 255),   # Alpha — light blue
    1: (100, 255, 100),   # Beta — green
    2: (255, 200,  50),   # BetaGamma — amber
    3: (255,  60,  60),   # BetaGammaDelta — red (highest flare risk)
}


def _draw_detections(image_path: Path, boxes, output_path: Path):
    """Overlay detection boxes on the image and save to output_path."""
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for box in boxes:
        cls_id       = int(box.cls[0])
        conf         = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        color        = CLASS_COLORS.get(cls_id, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(y1 - 20, 0)), f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                  fill=color, font=font)
    img.save(str(output_path))


def predict(
    weights: str,
    source: str,
    output_dir: str | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 1024,
    save_csv: bool = True,
):
    """Run inference and save annotated images + optional CSV. Returns list of detection dicts."""
    source_path = Path(source)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = source_path.parent if source_path.is_file() else source_path

    inputs = ([source_path] if source_path.is_file()
              else sorted(source_path.glob("*.png")) + sorted(source_path.glob("*.jpg")))

    rlog.kv_table([
        ("Weights", weights),
        ("Source",  f"{source}  ({len(inputs)} images)"),
        ("Conf ≥",  f"{conf_threshold}  ·  IoU ≤ {iou_threshold}"),
        ("Output",  str(out_path)),
    ])

    model          = YOLO(weights)
    all_detections = []
    class_counts: Counter[str] = Counter()

    with rlog.make_progress("Detecting") as progress:
        task = progress.add_task("", total=len(inputs))
        for img_file in inputs:
            results = model.predict(str(img_file), conf=conf_threshold,
                                    iou=iou_threshold, imgsz=imgsz, verbose=False)
            boxes = results[0].boxes
            _draw_detections(img_file, boxes, out_path / f"{img_file.stem}_pred.png")

            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                all_detections.append({
                    "image": img_file.name, "class_id": cls_id,
                    "class_name": cls_name, "confidence": round(conf, 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                })
                class_counts[cls_name] += 1
            progress.advance(task)

    rlog.success(
        f"{len(all_detections)} detections across {len(inputs)} images"
    )

    # Per-class breakdown
    if class_counts:
        parts = "  ·  ".join(f"{n} {cls}" for cls, n in class_counts.most_common())
        rlog.info(f"[dim]{parts}[/dim]")

    if save_csv and all_detections:
        csv_path = out_path / "detections.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_detections[0].keys())
            writer.writeheader()
            writer.writerows(all_detections)
        rlog.info(f"CSV saved → [cyan]{csv_path}[/cyan]")

    return all_detections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Inference with trained solar AR detector")
    p.add_argument("--weights", required=True)
    p.add_argument("--source",  required=True)
    p.add_argument("--output",  default=None)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--imgsz",   type=int,   default=1024)
    p.add_argument("--no-csv",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict(weights=args.weights, source=args.source, output_dir=args.output,
            conf_threshold=args.conf, iou_threshold=args.iou,
            imgsz=args.imgsz, save_csv=not args.no_csv)
