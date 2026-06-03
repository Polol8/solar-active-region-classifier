"""
Tests for src/dataset.py — dataset splitting and file-copy logic.

All tests use tmp_path; no real data files are required.
"""

import pytest
from pathlib import Path

from src.dataset import split_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pairs(tmp_path: Path, n: int):
    """Create n dummy PNG + YOLO label pairs in separate source directories."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    for i in range(n):
        # Use zero-padded filenames so lexicographic order = chronological order
        stem = f"frame_{i:04d}"
        (img_dir / f"{stem}.png").write_bytes(b"\x89PNG")   # minimal content
        (lbl_dir / f"{stem}.txt").write_text(f"1 0.5 0.5 0.1 0.1\n")

    return img_dir, lbl_dir


# ---------------------------------------------------------------------------
# Split counts
# ---------------------------------------------------------------------------

class TestSplitCounts:
    def test_default_split_70_15_15(self, tmp_path):
        img_dir, lbl_dir = _make_pairs(tmp_path, 100)
        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out))

        n_train = len(list((out / "train" / "images").glob("*.png")))
        n_val   = len(list((out / "val"   / "images").glob("*.png")))
        n_test  = len(list((out / "test"  / "images").glob("*.png")))

        assert n_train == 70
        assert n_val   == 15
        assert n_test  == 15

    def test_all_images_accounted_for(self, tmp_path):
        n = 47
        img_dir, lbl_dir = _make_pairs(tmp_path, n)
        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out))

        total = sum(
            len(list((out / split / "images").glob("*.png")))
            for split in ("train", "val", "test")
        )
        assert total == n

    def test_custom_split_fractions(self, tmp_path):
        img_dir, lbl_dir = _make_pairs(tmp_path, 200)
        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out),
                      train_frac=0.80, val_frac=0.10)

        n_train = len(list((out / "train" / "images").glob("*.png")))
        n_val   = len(list((out / "val"   / "images").glob("*.png")))
        n_test  = len(list((out / "test"  / "images").glob("*.png")))

        assert n_train == 160
        assert n_val   == 20
        assert n_test  == 20


# ---------------------------------------------------------------------------
# Chronological order preserved
# ---------------------------------------------------------------------------

class TestChronologicalOrder:
    def test_train_gets_earliest_frames(self, tmp_path):
        img_dir, lbl_dir = _make_pairs(tmp_path, 10)
        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out),
                      train_frac=0.70, val_frac=0.15)

        train_stems = sorted(p.stem for p in (out / "train" / "images").glob("*.png"))
        val_stems   = sorted(p.stem for p in (out / "val"   / "images").glob("*.png"))
        test_stems  = sorted(p.stem for p in (out / "test"  / "images").glob("*.png"))

        # train frames should all come before val frames, which come before test frames
        assert max(train_stems) < min(val_stems)
        assert max(val_stems)   < min(test_stems)


# ---------------------------------------------------------------------------
# Label files mirror image files
# ---------------------------------------------------------------------------

class TestLabelMirroring:
    def test_every_image_has_a_label(self, tmp_path):
        img_dir, lbl_dir = _make_pairs(tmp_path, 30)
        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out))

        for split in ("train", "val", "test"):
            imgs = {p.stem for p in (out / split / "images").glob("*.png")}
            lbls = {p.stem for p in (out / split / "labels").glob("*.txt")}
            assert imgs == lbls, f"Mismatch in {split}: images={imgs}, labels={lbls}"

    def test_label_content_preserved(self, tmp_path):
        img_dir, lbl_dir = _make_pairs(tmp_path, 5)
        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out))

        for split in ("train", "val", "test"):
            for lbl in (out / split / "labels").glob("*.txt"):
                assert lbl.read_text().strip() != "", f"Empty label in {split}/{lbl.name}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_images_without_labels_are_skipped(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir(); lbl_dir.mkdir()

        # 5 images, only 3 have labels
        for i in range(5):
            (img_dir / f"frame_{i:04d}.png").write_bytes(b"\x89PNG")
        for i in range(3):
            (lbl_dir / f"frame_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1")

        out = tmp_path / "datasets"
        split_dataset(str(img_dir), str(lbl_dir), str(out))

        total = sum(
            len(list((out / split / "images").glob("*.png")))
            for split in ("train", "val", "test")
        )
        assert total == 3   # only the 3 paired images are copied

    def test_output_dirs_created_if_missing(self, tmp_path):
        img_dir, lbl_dir = _make_pairs(tmp_path, 10)
        out = tmp_path / "new" / "nested" / "datasets"   # does not exist yet
        split_dataset(str(img_dir), str(lbl_dir), str(out))

        assert (out / "train" / "images").is_dir()
        assert (out / "val"   / "labels").is_dir()
        assert (out / "test"  / "images").is_dir()
