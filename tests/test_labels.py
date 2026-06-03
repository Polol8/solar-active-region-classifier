"""
Tests for src/labels.py — label generation logic.

No network calls are made: generate_label now accepts events directly,
so tests pass pre-built event lists without any mocking.
"""

import json
import pytest

from src.labels import (
    _normalise_mtwilson,
    _hpc_to_pixel,
    _estimate_box_size,
    generate_label,
    MIN_BOX_NORM,
)


# ---------------------------------------------------------------------------
# _normalise_mtwilson
# ---------------------------------------------------------------------------

class TestNormaliseMtwilson:
    def test_alpha_variants(self):
        assert _normalise_mtwilson("Alpha") == 0
        assert _normalise_mtwilson("alpha") == 0
        assert _normalise_mtwilson("A") == 0
        assert _normalise_mtwilson("a") == 0

    def test_beta_variants(self):
        assert _normalise_mtwilson("Beta") == 1
        assert _normalise_mtwilson("BETA") == 1
        assert _normalise_mtwilson("B") == 1

    def test_betagamma_variants(self):
        assert _normalise_mtwilson("BetaGamma") == 2
        assert _normalise_mtwilson("Beta-Gamma") == 2
        assert _normalise_mtwilson("BG") == 2
        assert _normalise_mtwilson("Gamma") == 2       # pure Gamma → BetaGamma
        assert _normalise_mtwilson("BetaDelta") == 2
        assert _normalise_mtwilson("Beta-Delta") == 2

    def test_betagammadelta_variants(self):
        assert _normalise_mtwilson("BetaGammaDelta") == 3
        assert _normalise_mtwilson("Beta-Gamma-Delta") == 3
        assert _normalise_mtwilson("BGD") == 3

    def test_unknown_returns_none(self):
        assert _normalise_mtwilson("Unknown") is None
        assert _normalise_mtwilson("") is None
        assert _normalise_mtwilson(None) is None

    def test_whitespace_stripped(self):
        assert _normalise_mtwilson("  Beta  ") == 1


# ---------------------------------------------------------------------------
# _hpc_to_pixel
# ---------------------------------------------------------------------------

class TestHpcToPixel:
    def test_disk_centre_maps_to_image_centre(self, wcs_meta):
        # HPC (0, 0) arcsec should land exactly at the image centre
        cx, cy = _hpc_to_pixel(0.0, 0.0, wcs_meta)
        # crpix is 2048.5 (1-based) → 0-based index 2047.5, scaled ×0.25 = 511.875
        expected_x = (wcs_meta["crpix1"] - 1) * wcs_meta["scale_x"]
        # Y is flipped: orig_height - 1 - native_py, then scaled
        native_py = wcs_meta["crpix2"] - 1   # 2047.5
        expected_y = (wcs_meta["orig_height"] - 1 - native_py) * wcs_meta["scale_y"]
        assert abs(cx - expected_x) < 0.01
        assert abs(cy - expected_y) < 0.01

    def test_positive_hpc_x_moves_right(self, wcs_meta):
        cx0, _ = _hpc_to_pixel(0.0, 0.0, wcs_meta)
        cx1, _ = _hpc_to_pixel(100.0, 0.0, wcs_meta)
        assert cx1 > cx0

    def test_positive_hpc_y_moves_up_in_image(self, wcs_meta):
        # Solar north = positive HPC_Y → smaller row index in PNG (row 0 is top)
        _, cy0 = _hpc_to_pixel(0.0, 0.0, wcs_meta)
        _, cy1 = _hpc_to_pixel(0.0, 100.0, wcs_meta)
        assert cy1 < cy0   # higher on disk → smaller pixel row

    def test_symmetry_east_west(self, wcs_meta):
        cx_east, _ = _hpc_to_pixel(200.0, 0.0, wcs_meta)
        cx_west, _ = _hpc_to_pixel(-200.0, 0.0, wcs_meta)
        cx_centre, _ = _hpc_to_pixel(0.0, 0.0, wcs_meta)
        assert abs((cx_centre - cx_west) - (cx_east - cx_centre)) < 0.1


# ---------------------------------------------------------------------------
# _estimate_box_size
# ---------------------------------------------------------------------------

class TestEstimateBoxSize:
    def test_larger_area_gives_larger_box(self, wcs_meta):
        small = _estimate_box_size(50, wcs_meta)
        large = _estimate_box_size(500, wcs_meta)
        assert large > small

    def test_minimum_area_enforced(self, wcs_meta):
        # area=0 should still return a positive box (minimum clamped to 10 MSH internally)
        box = _estimate_box_size(0, wcs_meta)
        assert box > 0

    def test_scales_with_square_root_of_area(self, wcs_meta):
        # Doubling area should scale box by sqrt(2), approximately
        b1 = _estimate_box_size(100, wcs_meta)
        b4 = _estimate_box_size(400, wcs_meta)
        ratio = b4 / b1
        assert abs(ratio - 2.0) < 0.2   # sqrt(4) = 2, allow 10 % tolerance


# ---------------------------------------------------------------------------
# generate_label  (integration-level, HEK mocked)
# ---------------------------------------------------------------------------

def _make_hek_event(hpc_x, hpc_y, area_msh, mtwilson_cls):
    return {
        "hpc_x": hpc_x,
        "hpc_y": hpc_y,
        "ar_area": area_msh,
        "ar_mtwilsoncls": mtwilson_cls,
    }


class TestGenerateLabel:
    def test_writes_label_for_valid_event(self, tmp_path, sidecar_file):
        event = _make_hek_event(0.0, 0.0, 200, "Beta")
        n = generate_label(sidecar_file, tmp_path, events=[event])

        assert n == 1
        label_file = tmp_path / f"{sidecar_file.stem}.txt"
        assert label_file.exists()

        line = label_file.read_text().strip()
        parts = line.split()
        assert len(parts) == 5
        cls_id, cx, cy, w, h = int(parts[0]), *[float(p) for p in parts[1:]]
        assert cls_id == 1          # Beta → 1
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert w >= MIN_BOX_NORM
        assert h >= MIN_BOX_NORM

    def test_empty_label_when_no_events(self, tmp_path, sidecar_file):
        n = generate_label(sidecar_file, tmp_path, events=[])

        assert n == 0
        label_file = tmp_path / f"{sidecar_file.stem}.txt"
        assert label_file.exists()
        assert label_file.read_text() == ""

    def test_skips_event_with_unknown_class(self, tmp_path, sidecar_file):
        event = _make_hek_event(0.0, 0.0, 100, "UnknownClass")
        n = generate_label(sidecar_file, tmp_path, events=[event])

        assert n == 0

    def test_skips_event_outside_image(self, tmp_path, sidecar_file):
        event = _make_hek_event(5000.0, 5000.0, 100, "Beta")
        n = generate_label(sidecar_file, tmp_path, events=[event])

        assert n == 0

    def test_multiple_events_written(self, tmp_path, sidecar_file):
        events = [
            _make_hek_event(-200.0,  300.0, 150, "Alpha"),
            _make_hek_event( 400.0, -200.0, 800, "BetaGammaDelta"),
        ]
        n = generate_label(sidecar_file, tmp_path, events=events)

        assert n == 2
        lines = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip().splitlines()
        assert len(lines) == 2
        assert int(lines[0].split()[0]) == 0   # Alpha
        assert int(lines[1].split()[0]) == 3   # BetaGammaDelta

    def test_all_normalised_values_in_range(self, tmp_path, sidecar_file):
        events = [_make_hek_event(x, y, 200, "Beta")
                  for x, y in [(-400, 400), (400, -400), (0, 0)]]
        generate_label(sidecar_file, tmp_path, events=events)

        lines = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip().splitlines()
        for line in lines:
            _, cx, cy, w, h = [float(v) for v in line.split()]
            assert 0.0 <= cx <= 1.0, f"cx out of range: {cx}"
            assert 0.0 <= cy <= 1.0, f"cy out of range: {cy}"
            assert w > 0
            assert h > 0
