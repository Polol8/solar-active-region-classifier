"""
Draw ground-truth YOLO label boxes on top of preprocessed PNGs, for visual QA.

Why this exists
  labels.py projects HEK catalogue positions into pixel coordinates through
  several unverifiable steps (WCS linear approximation, HMI camera-orientation
  correction, differential-rotation correction).  A wrong assumption anywhere
  in that chain produces boxes that are geometrically valid (in [0, 1], right
  shape) but land on the wrong part of the disk — something no unit test can
  catch, because unit tests only check the arithmetic is internally
  consistent, not that it matches physical reality.  This script renders the
  boxes over the actual magnetogram so a human can confirm they sit on real
  sunspot groups before trusting the labels for training.

Usage:
    python -m src.debug_overlay --images data/images --labels data/labels \
                                --output debug/overlay --limit 20
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src import log as rlog
from src.predict import CLASS_COLORS, CLASS_NAMES


def draw_overlay(image_path: Path, label_path: Path, output_path: Path) -> int:
    """Draw every YOLO box in label_path onto image_path, save to output_path.

    Returns the number of boxes drawn (0 if label_path is missing/empty).
    """
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    w, h = img.size
    n = 0

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id, cx, cy, bw, bh = line.split()
            cls_id = int(cls_id)
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            name  = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(y1 - 20, 0)), name, fill=color, font=font)
            n += 1

    img.save(str(output_path))
    return n


def process_directory(images_dir: str, labels_dir: str, output_dir: str, limit: int | None = None):
    """Draw label overlays for every PNG in images_dir with a matching label file."""
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = sorted(img_path.glob("*.png"))
    if limit:
        images = images[:limit]

    rlog.kv_table([
        ("Images", f"{img_path}/  ({len(images)} files)"),
        ("Labels", f"{lbl_path}/"),
        ("Output", f"{out_path}/"),
    ])

    total_boxes  = 0
    missing_lbls = 0

    with rlog.make_progress("Drawing overlays") as progress:
        task = progress.add_task("", total=len(images))
        for img_file in images:
            lbl_file = lbl_path / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                missing_lbls += 1
            out_file = out_path / f"{img_file.stem}_overlay.png"
            total_boxes += draw_overlay(img_file, lbl_file, out_file)
            progress.advance(task)

    rlog.success(f"{total_boxes} boxes drawn across {len(images)} images -> [cyan]{out_path}/[/cyan]")
    if missing_lbls:
        rlog.warn(f"{missing_lbls}/{len(images)} images had no label file (shown with no boxes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Draw YOLO ground-truth labels on PNGs for visual QA")
    p.add_argument("--images", default="data/images")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--output", default="debug/overlay")
    p.add_argument("--limit",  type=int, default=None, help="Only process the first N images (sorted)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_directory(args.images, args.labels, args.output, limit=args.limit)
