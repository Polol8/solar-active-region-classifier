"""Shared fixtures for the solar-classifier test suite."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs for heavy scientific dependencies.
# This lets the test suite run without installing astropy / sunpy.
# The tests themselves mock the specific callables they care about.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []        # mark as package so sub-imports resolve
    return mod


def _register_stubs():
    stubs = [
        "astropy",
        "astropy.units",
        "astropy.time",
        "astropy.utils",
        "astropy.utils.iers",
        "sunpy",
        "sunpy.map",
        "sunpy.net",
        "sunpy.net.attrs",
        "sunpy.net.attrs.hek",
    ]
    for name in stubs:
        if name not in sys.modules:
            mod = _make_stub(name)
            sys.modules[name] = mod

    # astropy.units needs a `u.min` / `u.hour` sentinel
    u = sys.modules["astropy.units"]
    u.min  = MagicMock(name="u.min")
    u.hour = MagicMock(name="u.hour")

    # astropy.utils.iers.conf needs auto_download / auto_max_age attributes
    iers_conf = MagicMock(name="iers_conf")
    iers_conf.auto_download = True
    iers_conf.auto_max_age  = None
    sys.modules["astropy.utils.iers"].conf = iers_conf
    sys.modules["astropy.utils"].iers = sys.modules["astropy.utils.iers"]

    # astropy.time.Time must be callable (returns a mock with isot)
    t_cls = MagicMock(name="Time")
    t_instance = MagicMock()
    t_instance.isot = "2014-01-01T00:00:00.000"
    t_cls.return_value = t_instance
    sys.modules["astropy.time"].Time = t_cls

    # sunpy.net: Fido and attrs (hek.EventType / hek.OBS / hek.AR)
    Fido = MagicMock(name="Fido")
    sys.modules["sunpy.net"].Fido = Fido
    attrs = sys.modules["sunpy.net.attrs"]
    attrs.Time = MagicMock(name="attrs.Time")
    attrs.hek = sys.modules["sunpy.net.attrs.hek"]
    attrs.hek.EventType = MagicMock(name="EventType")
    attrs.hek.OBS = MagicMock(name="OBS")

    # sunpy.map.Map must be importable (tests patch it individually)
    sys.modules["sunpy.map"].Map = MagicMock(name="Map")

    # Wire up attribute chains so `sunpy.map` is reachable via parent modules
    sys.modules["sunpy"].map = sys.modules["sunpy.map"]
    sys.modules["sunpy"].net = sys.modules["sunpy.net"]
    sys.modules["sunpy.net"].attrs = sys.modules["sunpy.net.attrs"]


_register_stubs()


@pytest.fixture
def wcs_meta():
    """Realistic WCS sidecar metadata for a 4096×4096 HMI image resized to 1024×1024."""
    return {
        "fits_file": "hmi_m_2014_01_01.fits",
        "date_obs": "2014-01-01T00:00:00.000",
        "orig_width": 4096,
        "orig_height": 4096,
        "target_size": 1024,
        "scale_x": 0.25,
        "scale_y": 0.25,
        # Reference pixel at disk centre (1-based FITS)
        "crpix1": 2048.5,
        "crpix2": 2048.5,
        "cdelt1": 0.504,   # arcsec/pixel — typical HMI value
        "cdelt2": 0.504,
        "crval1": 0.0,     # disk centre at (0, 0) arcsec
        "crval2": 0.0,
    }


@pytest.fixture
def sidecar_file(tmp_path, wcs_meta):
    """Write a WCS sidecar JSON to a temp file and return its Path."""
    path = tmp_path / "hmi_m_2014_01_01.json"
    path.write_text(json.dumps(wcs_meta))
    return path
