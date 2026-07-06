"""
Tests for src/labels.py — label generation logic.

No network calls are made: generate_label now accepts events directly,
so tests pass pre-built event lists without any mocking.
"""

import json
import pytest

from collections import Counter
from datetime import datetime

from src.labels import (
    _normalise_mtwilson,
    _hpc_to_pixel,
    _estimate_box_size,
    _filter_authoritative,
    _sharp_lookup,
    _nearest_spoca_bbox,
    _parse_hpc_bbox,
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

    def test_typo_tolerant_variants(self):
        # Real NOAA SRS formatting artefacts (doubled letter) that don't
        # survive an exact dictionary lookup — classified by substring
        # presence instead of being silently dropped.
        assert _normalise_mtwilson("ALPHAGAMMA-DELTA") == 3
        assert _normalise_mtwilson("BETAAGAMMA-DELTA") == 3


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

def _make_hek_event(hpc_x, hpc_y, area_msh, mtwilson_cls, frm_name="NOAA SWPC Observer", **extra):
    ev = {
        "hpc_x": hpc_x,
        "hpc_y": hpc_y,
        "ar_area": area_msh,
        "ar_mtwilsoncls": mtwilson_cls,
        "frm_name": frm_name,
    }
    ev.update(extra)
    return ev


# ---------------------------------------------------------------------------
# _parse_hpc_bbox
# ---------------------------------------------------------------------------

class TestParseHpcBbox:
    def test_parses_real_wkt(self):
        pts = [(-885.39, -170.3982), (-878.922, -169.6044),
               (-880.152, -162.6804), (-886.632, -163.476)]
        wkt = "POLYGON((" + ",".join(f"{x} {y}" for x, y in pts) + f",{pts[0][0]} {pts[0][1]}))"
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        result = _parse_hpc_bbox(wkt)
        assert result == pytest.approx((max(xs) - min(xs), max(ys) - min(ys)))

    def test_returns_none_for_malformed_string(self):
        assert _parse_hpc_bbox("not a polygon") is None

    def test_returns_none_for_empty_or_missing(self):
        assert _parse_hpc_bbox("") is None
        assert _parse_hpc_bbox(None) is None


# ---------------------------------------------------------------------------
# _filter_authoritative
# ---------------------------------------------------------------------------

class TestFilterAuthoritative:
    def test_keeps_only_noaa_source(self):
        events = [
            {"frm_name": "HMI SHARP",         "ar_noaanum": 1},
            {"frm_name": "SPoCA",              "ar_noaanum": 2},
            {"frm_name": "NOAA SWPC Observer", "ar_noaanum": 3},
        ]
        kept = _filter_authoritative(events)
        assert len(kept) == 1
        assert kept[0]["ar_noaanum"] == 3

    def test_deduplicates_by_noaanum(self):
        events = [
            {"frm_name": "NOAA SWPC Observer", "ar_noaanum": 11934},
            {"frm_name": "NOAA SWPC Observer", "ar_noaanum": 11934},
            {"frm_name": "NOAA SWPC Observer", "ar_noaanum": 11935},
        ]
        kept = _filter_authoritative(events)
        assert len(kept) == 2

    def test_events_without_noaanum_all_pass(self):
        events = [
            {"frm_name": "NOAA SWPC Observer"},
            {"frm_name": "NOAA SWPC Observer"},
        ]
        assert len(_filter_authoritative(events)) == 2


# ---------------------------------------------------------------------------
# _sharp_lookup
# ---------------------------------------------------------------------------

class TestSharpLookup:
    def test_picks_time_closest_sharp_event(self):
        events = [
            {"frm_name": "HMI SHARP", "ar_noaanum": 100, "event_starttime": "2014-01-01T00:00:00", "hpc_bbox": "A"},
            {"frm_name": "HMI SHARP", "ar_noaanum": 100, "event_starttime": "2014-01-01T16:00:00", "hpc_bbox": "B"},
            {"frm_name": "NOAA SWPC Observer", "ar_noaanum": 100, "hpc_bbox": "noaa box"},
        ]
        lookup = _sharp_lookup(events, datetime(2014, 1, 1, 17, 58))
        assert lookup[100]["hpc_bbox"] == "B"   # 16:00 report is closer to 17:58 than 00:00

    def test_ignores_events_without_noaanum(self):
        events = [{"frm_name": "HMI SHARP", "hpc_bbox": "POLYGON((x))"}]
        assert _sharp_lookup(events, None) == {}

    def test_ignores_non_sharp_sources(self):
        events = [{"frm_name": "SPoCA", "ar_noaanum": 1, "hpc_bbox": "POLYGON((x))"}]
        assert _sharp_lookup(events, None) == {}

    def test_falls_back_to_first_when_image_time_missing(self):
        events = [
            {"frm_name": "HMI SHARP", "ar_noaanum": 5, "hpc_bbox": "A"},
            {"frm_name": "HMI SHARP", "ar_noaanum": 5, "hpc_bbox": "B"},
        ]
        assert _sharp_lookup(events, None)[5]["hpc_bbox"] == "A"


# ---------------------------------------------------------------------------
# _nearest_spoca_bbox
# ---------------------------------------------------------------------------

class TestNearestSpocaBbox:
    def test_picks_closest_within_radius(self):
        events = [
            {"frm_name": "SPoCA", "hpc_x": 500.0, "hpc_y": 500.0, "hpc_bbox": "far"},
            {"frm_name": "SPoCA", "hpc_x": 100.0, "hpc_y": 50.0,  "hpc_bbox": "near"},
        ]
        assert _nearest_spoca_bbox(events, 110.0, 55.0) == "near"

    def test_returns_none_outside_radius(self):
        events = [{"frm_name": "SPoCA", "hpc_x": 500.0, "hpc_y": 500.0, "hpc_bbox": "far"}]
        assert _nearest_spoca_bbox(events, 0.0, 0.0) is None

    def test_ignores_non_spoca_sources(self):
        events = [{"frm_name": "HMI SHARP", "hpc_x": 0.0, "hpc_y": 0.0, "hpc_bbox": "sharp box"}]
        assert _nearest_spoca_bbox(events, 0.0, 0.0) is None

    def test_returns_none_for_empty_events(self):
        assert _nearest_spoca_bbox([], 0.0, 0.0) is None


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

    def test_non_authoritative_source_is_dropped(self, tmp_path, sidecar_file):
        event = _make_hek_event(0.0, 0.0, 200, "Beta", frm_name="HMI SHARP")
        n = generate_label(sidecar_file, tmp_path, events=[event])
        assert n == 0

    def test_falls_back_to_area_estimate_without_bbox(self, tmp_path, sidecar_file):
        event = _make_hek_event(0.0, 0.0, 200, "Beta")
        generate_label(sidecar_file, tmp_path, events=[event])
        line = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip()
        _, cx, cy, w, h = [float(v) for v in line.split()]
        assert w == pytest.approx(h)   # area-based estimate is always square

    def test_hpc_bbox_produces_nonsquare_box(self, tmp_path, sidecar_file):
        # Box size comes from a SHARP event cross-matched by ar_noaanum, not
        # from the NOAA event's own hpc_bbox (see SHARP_FRM_SUBSTR) — the
        # NOAA event carries the classification/position, SHARP the size.
        noaa_event = _make_hek_event(0.0, 0.0, 0, "Beta", ar_noaanum=12345)
        sharp_event = {
            "frm_name": "HMI SHARP", "ar_noaanum": 12345,
            "hpc_bbox": "POLYGON((-100 -50,100 -50,100 50,-100 50,-100 -50))",
        }
        n = generate_label(sidecar_file, tmp_path, events=[noaa_event, sharp_event])
        assert n == 1
        line = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip()
        _, cx, cy, w, h = [float(v) for v in line.split()]
        assert w > h   # bbox is 200 arcsec wide x 100 arcsec tall

    def test_spoca_fallback_used_when_no_sharp_match(self, tmp_path, sidecar_file):
        # No SHARP event present for this region — size should fall back to
        # the nearest SPoCA detection (matched by position) instead of the
        # square area-based estimate.
        noaa_event = _make_hek_event(100.0, 50.0, 0, "Beta", ar_noaanum=999)
        spoca_event = {
            "frm_name": "SPoCA", "hpc_x": 105.0, "hpc_y": 52.0,
            "hpc_bbox": "POLYGON((-50 -20,150 -20,150 20,-50 20,-50 -20))",
        }
        n = generate_label(sidecar_file, tmp_path, events=[noaa_event, spoca_event])
        assert n == 1
        line = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip()
        _, cx, cy, w, h = [float(v) for v in line.split()]
        assert w > h   # bbox is 200 arcsec wide x 40 arcsec tall

    def test_spoca_ignored_when_outside_match_radius(self, tmp_path, sidecar_file):
        noaa_event = _make_hek_event(100.0, 50.0, 200, "Beta", ar_noaanum=998)
        spoca_far = {
            "frm_name": "SPoCA", "hpc_x": 900.0, "hpc_y": 900.0,
            "hpc_bbox": "POLYGON((-50 -20,150 -20,150 20,-50 20,-50 -20))",
        }
        generate_label(sidecar_file, tmp_path, events=[noaa_event, spoca_far])
        line = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip()
        _, cx, cy, w, h = [float(v) for v in line.split()]
        assert w == pytest.approx(h)   # too far — falls back to area estimate

    def test_noaa_own_bbox_is_ignored_for_sizing(self, tmp_path, sidecar_file):
        # Regression guard: a NOAA event's own hpc_bbox must NOT be used for
        # box size (it's a lifetime-tracking envelope, not a physical
        # extent) — without a matching SHARP event, fall back to the area
        # estimate instead of trusting NOAA's degenerate/elongated bbox.
        event = _make_hek_event(0.0, 0.0, 200, "Beta",
                                 hpc_bbox="POLYGON((-500 -5,500 -5,500 5,-500 5,-500 -5))")
        generate_label(sidecar_file, tmp_path, events=[event])
        line = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip()
        _, cx, cy, w, h = [float(v) for v in line.split()]
        assert w == pytest.approx(h)   # area-based estimate, not the NOAA bbox

    def test_position_prefers_time_matched_sharp_over_noaa(self, tmp_path, sidecar_file, wcs_meta):
        # SHARP's automated centroid, refreshed every few hours, should win
        # over NOAA's once-daily position — and among several SHARP reports,
        # the one closest to the image's own date_obs should be picked.
        noaa_event = _make_hek_event(0.0, 0.0, 0, "Beta", ar_noaanum=777)
        sharp_wrong_time = {
            "frm_name": "HMI SHARP", "ar_noaanum": 777,
            "event_starttime": "2013-06-01T00:00:00",
            "hpc_x": 500.0, "hpc_y": 500.0,
        }
        sharp_matched = {
            "frm_name": "HMI SHARP", "ar_noaanum": 777,
            "event_starttime": wcs_meta["date_obs"],
            "hpc_x": 100.0, "hpc_y": 50.0,
        }
        n = generate_label(sidecar_file, tmp_path, events=[noaa_event, sharp_wrong_time, sharp_matched])
        assert n == 1
        line = (tmp_path / f"{sidecar_file.stem}.txt").read_text().strip()
        _, cx, cy, w, h = [float(v) for v in line.split()]

        expected_cx, expected_cy = _hpc_to_pixel(100.0, 50.0, wcs_meta)
        img_size = wcs_meta["target_size"]
        assert cx == pytest.approx(expected_cx / img_size, abs=1e-4)
        assert cy == pytest.approx(expected_cy / img_size, abs=1e-4)

    def test_unmapped_counter_increments_on_unknown_class(self, tmp_path, sidecar_file):
        counter = Counter()
        event = _make_hek_event(0.0, 0.0, 100, "TotallyUnknown")
        generate_label(sidecar_file, tmp_path, events=[event], unmapped_counter=counter)
        assert counter["TotallyUnknown"] == 1

    def test_network_error_returns_sentinel_and_writes_no_file(self, tmp_path, sidecar_file, monkeypatch):
        import urllib.request

        def _raise(*args, **kwargs):
            raise TimeoutError("mock network failure")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)

        n = generate_label(sidecar_file, tmp_path, events=None)

        assert n == -1
        assert not (tmp_path / f"{sidecar_file.stem}.txt").exists()
