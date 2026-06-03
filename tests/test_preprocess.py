"""
Tests for src/preprocess.py — FITS-to-PNG conversion and normalisation logic.

sunpy.map.Map is mocked throughout; no FITS files on disk are required.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from src.preprocess import fits_to_png, IMAGE_SIZE, B_CLIP


def _make_mock_map(data: np.ndarray, date_obs="2014-01-01T00:00:00.000"):
    """Return a MagicMock that behaves like a sunpy.map.Map."""
    m = MagicMock()
    m.data = data
    m.date = date_obs

    meta = {
        "crpix1": data.shape[1] / 2,
        "crpix2": data.shape[0] / 2,
        "cdelt1": 0.504,
        "cdelt2": 0.504,
        "crval1": 0.0,
        "crval2": 0.0,
    }
    m.meta = meta
    return m


# ---------------------------------------------------------------------------
# Normalisation arithmetic (pure functions, no I/O)
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_zero_maps_to_midpoint(self):
        # A pixel at 0 G should map to 127 (neutral grey)
        value = np.array([0.0], dtype=np.float32)
        normalised = np.clip(value, -B_CLIP, B_CLIP)
        normalised = (normalised + B_CLIP) / (2 * B_CLIP) * 255.0
        assert abs(normalised[0] - 127.5) < 0.5

    def test_positive_clip_maps_to_255(self):
        value = np.array([B_CLIP * 2], dtype=np.float32)   # above clip
        result = np.clip(value, -B_CLIP, B_CLIP)
        result = (result + B_CLIP) / (2 * B_CLIP) * 255.0
        assert result[0] == 255.0

    def test_negative_clip_maps_to_0(self):
        value = np.array([-B_CLIP * 2], dtype=np.float32)
        result = np.clip(value, -B_CLIP, B_CLIP)
        result = (result + B_CLIP) / (2 * B_CLIP) * 255.0
        assert result[0] == 0.0

    def test_nan_replaced_by_zero_before_clip(self):
        data = np.array([np.nan, 500.0, -500.0], dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0)
        assert not np.isnan(data).any()
        assert data[0] == 0.0


# ---------------------------------------------------------------------------
# fits_to_png  (I/O mocked)
# ---------------------------------------------------------------------------

class TestFitsToPng:
    def _run(self, tmp_path, data, date_obs="2014-01-01T00:00:00.000"):
        fits_path = tmp_path / "test.fits"
        fits_path.touch()                 # file must exist for Path.name to work

        mock_map = _make_mock_map(data, date_obs)
        with patch("sunpy.map.Map", return_value=mock_map):
            path, err = fits_to_png(fits_path, tmp_path)
        return path

    def test_returns_png_path(self, tmp_path):
        data = np.zeros((512, 512), dtype=np.float32)
        result = self._run(tmp_path, data)
        assert result is not None
        assert result.suffix == ".png"
        assert result.exists()

    def test_output_has_correct_size(self, tmp_path):
        data = np.ones((512, 512), dtype=np.float32) * 500.0
        self._run(tmp_path, data)
        png = tmp_path / "test.png"
        img = Image.open(png)
        assert img.size == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_is_grayscale(self, tmp_path):
        data = np.zeros((512, 512), dtype=np.float32)
        self._run(tmp_path, data)
        img = Image.open(tmp_path / "test.png")
        assert img.mode == "L"

    def test_sidecar_json_written(self, tmp_path):
        data = np.zeros((512, 512), dtype=np.float32)
        self._run(tmp_path, data)
        json_path = tmp_path / "test.json"
        assert json_path.exists()

    def test_sidecar_contains_wcs_keys(self, tmp_path):
        data = np.zeros((512, 512), dtype=np.float32)
        self._run(tmp_path, data)
        meta = json.loads((tmp_path / "test.json").read_text())
        required = {"orig_width", "orig_height", "target_size", "scale_x", "scale_y",
                    "crpix1", "crpix2", "cdelt1", "cdelt2", "crval1", "crval2"}
        assert required.issubset(meta.keys())

    def test_scale_factors_correct(self, tmp_path):
        data = np.zeros((4096, 4096), dtype=np.float32)
        self._run(tmp_path, data)
        meta = json.loads((tmp_path / "test.json").read_text())
        assert abs(meta["scale_x"] - IMAGE_SIZE / 4096) < 1e-6
        assert abs(meta["scale_y"] - IMAGE_SIZE / 4096) < 1e-6

    def test_nan_pixels_become_midgrey(self, tmp_path):
        data = np.full((128, 128), np.nan, dtype=np.float32)
        self._run(tmp_path, data)
        img = np.array(Image.open(tmp_path / "test.png"))
        # NaN → 0 G → normalised midpoint (~127)
        assert abs(int(img.mean()) - 127) <= 2

    def test_uniform_positive_field_is_bright(self, tmp_path):
        data = np.full((128, 128), B_CLIP, dtype=np.float32)
        self._run(tmp_path, data)
        img = np.array(Image.open(tmp_path / "test.png"))
        assert img.mean() > 200

    def test_uniform_negative_field_is_dark(self, tmp_path):
        data = np.full((128, 128), -B_CLIP, dtype=np.float32)
        self._run(tmp_path, data)
        img = np.array(Image.open(tmp_path / "test.png"))
        assert img.mean() < 55

    def test_returns_none_on_bad_fits(self, tmp_path):
        bad_fits = tmp_path / "corrupt.fits"
        bad_fits.touch()
        with patch("sunpy.map.Map", side_effect=Exception("corrupt")):
            path, err = fits_to_png(bad_fits, tmp_path)
        assert path is None
        assert err is not None
