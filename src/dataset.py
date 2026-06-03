"""
Split processed images + labels into train / val / test sets and
copy them into the datasets/ directory tree expected by YOLO.

Splitting strategy:
  - Images are sorted chronologically.
  - Split is done in time order (not shuffled) to avoid data leakage from
    adjacent frames belonging to the same active region.
  - Default split: 70% train / 15% val / 15% test.

Usage:
    python -m src.dataset --images data/images --labels data/labels \
                          --output datasets --train 0.70 --val 0.15
"""

import argparse
import shutil
from pathlib import Path

from rich.table import Table

from src import log as rlog


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """
    Copy image/label pairs into datasets/{train,val,test}/{images,labels}/.

    Args:
        images_dir:  Directory containing PNG files.
        labels_dir:  Directory containing matching YOLO .txt label files.
        output_dir:  Root of the YOLO dataset tree (datasets/).
        train_frac:  Fraction for training set.
        val_frac:    Fraction for validation set; remainder goes to test.
    """
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)
    out_path = Path(output_dir)

    # Collect matched (image, label) pairs, sorted chronologically by filename
    images = sorted(img_path.glob("*.png"))
    pairs  = []
    for img in images:
        lbl = lbl_path / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            rlog.warn(f"No label for {img.name} — skipping")

    n = len(pairs)
    if n == 0:
        rlog.error("No matched image/label pairs found. Aborting.")
        return

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train: n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    rlog.kv_table([
        ("Total",    f"{n} image/label pairs"),
        ("Strategy", "chronological (no shuffle)"),
        ("Output",   str(out_path)),
    ])

    with rlog.make_progress("Copying files") as progress:
        task = progress.add_task("", total=n)
        for split_name, split_pairs in splits.items():
            img_dest = out_path / split_name / "images"
            lbl_dest = out_path / split_name / "labels"
            img_dest.mkdir(parents=True, exist_ok=True)
            lbl_dest.mkdir(parents=True, exist_ok=True)
            for img, lbl in split_pairs:
                shutil.copy2(img, img_dest / img.name)
                shutil.copy2(lbl, lbl_dest / lbl.name)
                progress.advance(task)

    # Summary table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Split", style="cyan", min_width=8)
    table.add_column("Images", justify="right")
    table.add_column("%", justify="right", style="dim")
    for name, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        table.add_row(name, str(count), f"{100 * count / n:.0f}")
    rlog.console.print()
    rlog.console.print("  [bold]Dataset split[/bold]")
    rlog.console.print(table)

    rlog.success(f"Dataset ready at [cyan]{out_path}/[/cyan]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Split processed data into train/val/test sets")
    p.add_argument("--images", default="data/images", help="Processed PNG images dir")
    p.add_argument("--labels", default="data/labels", help="YOLO label .txt dir")
    p.add_argument("--output", default="datasets",    help="Output dataset root dir")
    p.add_argument("--train",  type=float, default=0.70, help="Train fraction (default 0.70)")
    p.add_argument("--val",    type=float, default=0.15, help="Val fraction (default 0.15)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_dataset(args.images, args.labels, args.output, args.train, args.val)
