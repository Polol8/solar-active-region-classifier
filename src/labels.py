"""
Generate YOLO-format labels for solar active regions using the HEK catalogue.

Background — what is the HEK?
  The Heliophysics Event Knowledgebase (HEK) is a database maintained by LMSAL
  (Lockheed Martin Solar and Astrophysics Laboratory) that aggregates solar event
  detections from many automated pipelines.  We use it to find Active Region (AR)
  events and their Mount Wilson magnetic-complexity classification.

Background — Mount Wilson classification
  The Mount Wilson scheme describes the magnetic topology of sunspot groups:
    Alpha  — a single dominant polarity (unipolar), low flare probability.
    Beta   — clear bipolar separation between positive and negative spots.
    BetaGamma — bipolar with a complex polarity-inversion line running between
                spots; substantially higher flare probability than Beta.
    BetaGammaDelta — same as BetaGamma but with one or more umbrae of opposite
                     polarity sharing a penumbra (delta configuration); the
                     highest probability of producing X-class flares.

Background — coordinate systems
  HMI magnetograms use the standard solar coordinate system:
    HPC (Heliocentric Projected, arcsec) — origin at disk centre, X increases
    toward solar west, Y increases toward solar north.
    FITS stores arrays with row 0 at the bottom (south), but PNG/numpy store
    row 0 at the top, so the Y axis must be flipped when projecting HPC
    coordinates onto pixel coordinates.

Background — MSH (Millionths of Solar Hemisphere)
  The HEK reports active-region areas in MSH.  One MSH = 10⁻⁶ of the area of
  one solar hemisphere (≈ 3.04 × 10⁹ km²).  Typical ARs range from ~10 MSH
  (small sunspot) to ~2000 MSH (very large complex group).

Bulk-query strategy (1 query per source per day)
  We make ONE HEK query and ONE JSOC SHARP query per calendar day, cache
  both in memory, and match records to individual image timestamps by
  checking a ±30-minute window around the observation time.  For a 1-month
  run this reduces hundreds of per-image requests to 31 × 2 = 62 total.

Why JSOC SHARP for bounding boxes?
  SHARP (Spaceweather HMI Active Region Patches, hmi.sharp_720s) are
  pixel-precise cutouts of the HMI full-disk images, produced by the JSOC
  from the same data we download.  The SHARP FITS header stores:
    NAXIS1, NAXIS2  — cutout dimensions in native HMI pixels (= bbox size).
    CRVAL1, CRVAL2  — HPC centre of the cutout in arcseconds.
  This gives us exact bounding boxes without any area estimation.  We query
  only metadata (jsoc_info.cgi, no authentication, no file download) and
  match each HEK AR event to the closest SHARP patch by HPC distance.

Implementation note — no sunpy.net dependency
  sunpy.net makes network calls during module import, blocking the terminal
  silently.  We bypass it entirely and query the HEK REST endpoint directly via
  urllib.request.urlopen with an explicit timeout parameter so a slow or
  unresponsive server raises socket.timeout instead of hanging forever.

YOLO label format (one line per bounding box):
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    All five values are normalised to [0, 1] relative to image dimensions.

Usage:
    python -m src.labels --images data/images --labels data/labels
"""

import argparse
import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src import log as rlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HEK REST endpoint.  Querying with cosec=2 returns JSON.
HEK_URL          = "https://www.lmsal.com/hek/her"

# Maximum seconds to wait for a single HEK HTTP response.  If the server
# does not reply within this window, socket.timeout is raised and the day
# is skipped (images from that day get empty label files).
HEK_TIMEOUT_SEC  = 45

# Half-width of the time window used to associate a cached day-query event
# with a specific image.  An event is considered "active" at image time T
# if its event_starttime ≤ T + window  AND  event_endtime ≥ T − window.
# 30 minutes is conservative for HMI magnetograms sampled every 6 hours.
HEK_WINDOW_MIN   = 30

# Maximum rows returned per day query.  January 2014 had ~50–80 AR reports
# per day; 300 is safe headroom without sending oversized responses.
HEK_RESULT_LIMIT = 300

# Fields requested from the HEK per event row.  Only fetching what we need
# keeps the JSON payload small and the parsing fast.
HEK_RETURN = ",".join([
    "ar_mtwilsoncls",   # Mount Wilson class (primary classification field)
    "frm_specificid",   # fallback class field used by some pipeline versions
    "ar_noaanum",       # NOAA AR number — used to match HEK events to SHARP records
    "hpc_x", "hpc_y",  # centroid position in arcseconds (HPC frame)
    "ar_area",          # area in MSH (fallback sizing only)
    "event_starttime",  # start of the AR's tracked lifetime
    "event_endtime",    # end of the AR's tracked lifetime
])

# Minimum normalised bounding-box side length.
MIN_BOX_NORM         = 0.015

# IoU threshold for ground-truth NMS applied after all boxes are computed
# for one image.  Overlapping boxes above this threshold are collapsed to the
# largest one, preventing multiple identical labels for the same SHARP patch.
NMS_IOU_THRESH       = 0.5

# ---------------------------------------------------------------------------
# JSOC / SHARP constants
# ---------------------------------------------------------------------------

# JSOC metadata endpoint.  jsoc_info.cgi is a public read-only REST API —
# no email registration or authentication is required (unlike jsoc_fetch.cgi).
JSOC_INFO_URL   = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info"

# SHARP series and keywords we request.
# CRVAL1/CRVAL2 in hmi.sharp_720s are always (0,0) — just the coordinate origin
# definition, not the AR position.  NAXIS1/NAXIS2 are segment-level and not
# queryable via rs_list.  Instead we use the heliographic bounding box
# (LON_MIN/MAX, LAT_MIN/MAX) and match by NOAA AR number.
SHARP_SERIES    = "hmi.sharp_720s"
SHARP_KEYS      = "T_REC,HARPNUM,NOAA_ARS,LON_MIN,LON_MAX,LAT_MIN,LAT_MAX"

# Per-day SHARP query: max records to accept.
# ~20 active regions × 120 cadence steps/day = 2400; 5000 is safe headroom.
SHARP_MAX_RECORDS = 5000

# Timeout for JSOC jsoc_info requests.
JSOC_TIMEOUT_SEC = 45

# ---------------------------------------------------------------------------
# Fallback constants (used when SHARP has no coverage for an AR)
# ---------------------------------------------------------------------------

# Fallback area (MSH) per Mount Wilson class — used only when SHARP metadata
# is unavailable.  Physically motivated medians from Solar Cycle 24 statistics.
_CLASS_DEFAULT_AREA_MSH = {0: 80, 1: 150, 2: 300, 3: 500}  # Alpha … BGD

# HMI plate scale: the solar radius subtends ~960 arcseconds as seen from
# Earth.  Used to convert physical AR area (MSH) → pixel area.
SOLAR_RADIUS_ARCSEC  = 960.0

# YOLO class names in index order (must match configs/solar.yaml).
CLASS_NAMES = ["Alpha", "Beta", "BetaGamma", "BetaGammaDelta"]

# Mapping from raw HEK string → integer class index.  Multiple aliases exist
# because different HEK pipelines (NOAA SRS, SHARP, SOON) use slightly
# different capitalisation and separators.
MTWILSON_MAP = {
    # Alpha — single dominant polarity
    "alpha": 0, "a": 0,
    # Beta — clean bipolar pair
    "beta": 1,  "b": 1,
    # BetaGamma — bipolar with complex inversion line
    "betagamma": 2, "beta-gamma": 2, "bg": 2,
    # Gamma and BetaDelta are rare but share BetaGamma's complexity level
    "gamma": 2,     "betadelta": 2,  "beta-delta": 2,
    # BetaGammaDelta — highest complexity, delta umbrae present
    "betagammadelta": 3, "beta-gamma-delta": 3, "bgd": 3,
}

# ---------------------------------------------------------------------------
# Pure helpers — no I/O, fully unit-testable
# ---------------------------------------------------------------------------

def _normalise_mtwilson(raw: str) -> int | None:
    """Return a class index for a raw Mount Wilson string, or None if unknown.

    First tries an exact lookup after normalising whitespace/underscores/slashes.
    Falls back to component-based parsing to handle HEK artefacts where the
    NOAA SRS Zurich-type letter gets concatenated into the Mount Wilson field
    (e.g. 'BETAA' = Beta + Zurich-type 'A', 'BETAAGAMMA' = Beta-Gamma,
    'ALPHAGAMMA-DELTA' = Gamma-Delta complex).

    Rule: take the most complex component present:
      delta anywhere → 3 (BetaGammaDelta)
      gamma anywhere → 2 (BetaGamma)
      beta  anywhere → 1 (Beta)
      alpha/anything → 0 (Alpha)
    """
    if not raw:
        return None
    key = re.sub(r"[\s_/]", "", raw.strip().lower())
    exact = MTWILSON_MAP.get(key)
    if exact is not None:
        return exact
    # Fuzzy fallback: most complex component wins
    if "delta" in key:
        return 3
    if "gamma" in key:
        return 2
    if "beta" in key:
        return 1
    if "alpha" in key:
        return 0
    return None


def _hpc_to_pixel(hpc_x_arcsec: float, hpc_y_arcsec: float, meta: dict) -> tuple[float, float]:
    """Convert a heliocentric projected position (arcsec) to pixel coordinates
    in the *resized* PNG image, using the linear WCS parameters stored in the
    sidecar JSON written by preprocess.py.

    The WCS projection used here is the simple linear (TAN) approximation,
    valid to within a few pixels for positions within ~800 arcsec of disk
    centre — which covers almost all active regions except those near the limb.

    FITS convention: pixel (1, 1) is at the *bottom-left* corner of the image
    and CRPIX stores 1-based indices.  numpy/PIL use 0-based row 0 = top,
    so we subtract 1 from CRPIX and flip the Y axis.
    """
    # Translate HPC arcseconds to native-resolution pixel coordinates (0-based)
    px_native = (meta["crpix1"] - 1) + (hpc_x_arcsec - meta["crval1"]) / meta["cdelt1"]
    py_native = (meta["crpix2"] - 1) + (hpc_y_arcsec - meta["crval2"]) / meta["cdelt2"]

    # Flip Y: in FITS row 0 is the solar south limb; in PNG row 0 is the top
    py_native = meta["orig_height"] - 1 - py_native

    # Scale from native 4096×4096 HMI resolution to the resized PNG
    return float(px_native * meta["scale_x"]), float(py_native * meta["scale_y"])


def _estimate_box_size(area_msh: float, meta: dict) -> float:
    """Estimate the bounding-box side length (pixels, in the resized image)
    from the active-region area expressed in Millionths of Solar Hemisphere.

    Derivation:
      1. Convert MSH → pixel² using the plate scale (arcsec/pixel) and the
         known angular size of one solar hemisphere (π × R_sun²).
      2. Assume the region is circular to get an equivalent radius.
      3. Multiply by 2 for the full diameter, then by 1.5 as a safety margin
         that accounts for the non-circular shape of real active regions and
         the fact that the HEK area can underestimate the full extent.
      4. Scale from native HMI resolution to the resized PNG dimensions.

    A floor of 10 MSH is applied before the conversion so that tiny ARs
    (reported as 0–1 MSH by some pipelines) still produce a visible box.
    """
    r_sun_pix  = SOLAR_RADIUS_ARCSEC / meta["cdelt1"]   # solar radius in native px
    # Area of one hemisphere in pixel² at native resolution
    hemi_pix2  = np.pi * r_sun_pix ** 2
    # Convert MSH area to pixel² — floor prevents zero-area boxes
    area_pix2  = max(area_msh, 10.0) * 1e-6 * hemi_pix2
    # Circular-equivalent diameter with 3× margin.  The HEK area reports the
    # magnetic footprint; the full visible AR extent (penumbra + network) is
    # typically 2–3× larger, and YOLO needs boxes that enclose the region.
    box_native = 2 * np.sqrt(area_pix2 / np.pi) * 3.0
    # Scale to the resized PNG
    return float(box_native * meta["scale_x"])


# ---------------------------------------------------------------------------
# HEK network helpers — stdlib urllib only, no sunpy.net
# ---------------------------------------------------------------------------

def _query_day_hek(day: str) -> list:
    """Query the HEK REST API for all AR events on a given calendar day.

    Args:
        day: ISO date string 'YYYY-MM-DD'.

    Returns:
        A list of event dicts (possibly empty).  Never raises — network errors
        are caught and logged as warnings, returning an empty list so the
        pipeline continues with unlabelled images for that day.

    The request URL uses the standard HEK search parameters:
      cosec=2      → return JSON (not XML)
      event_type=ar → active-region events only
      x1/x2/y1/y2  → full-disk bounding box in arcseconds (±1200 arcsec ≈
                      slightly beyond the solar limb at ~960 arcsec radius)
      result_limit  → cap at HEK_RESULT_LIMIT to avoid oversized responses
      return        → comma-separated list of fields to include in each row
    """
    params = urllib.parse.urlencode({
        "cosec":           "2",
        "cmd":             "search",
        "type":            "column",
        "event_type":      "ar",
        "event_starttime": f"{day}T00:00:00",
        "event_endtime":   f"{day}T23:59:59",
        "event_coordsys":  "helioprojective",
        "x1": "-1200", "x2": "1200",   # full-disk ± limb in arcsec
        "y1": "-1200", "y2": "1200",
        "result_limit":    str(HEK_RESULT_LIMIT),
        "return":          HEK_RETURN,
    })
    url = f"{HEK_URL}?{params}"

    try:
        # The explicit timeout parameter is the correct way to prevent urllib
        # from blocking forever on a slow or unresponsive server.
        with urllib.request.urlopen(url, timeout=HEK_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("result", [])
    except Exception as exc:
        rlog.warn(f"HEK {day}: {type(exc).__name__} — skipping")
        return []


def _parse_hek_time(raw: str) -> datetime | None:
    """Parse a HEK timestamp string to a datetime, returning None on failure.

    HEK uses two common formats: 'YYYY-MM-DD HH:MM:SS' (most pipelines) and
    'YYYY-MM-DDTHH:MM:SS' (ISO 8601).  We try both.
    """
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(raw)[:19], fmt)
        except ValueError:
            continue
    return None


def _events_for_image(day_events: list, date_obs: str) -> list:
    """Filter a day's cached events to those active within ±HEK_WINDOW_MIN
    of the image observation time.

    An event is considered active at time T if:
        event_starttime ≤ T + window  AND  event_endtime ≥ T − window

    This correctly handles both:
      - Long-lived SHARP patches that span multiple days (their window is
        wide enough to cover our ±30-minute filter easily).
      - NOAA SRS point-in-time reports where start == end; these match if
        the report falls within the 30-minute window around the image.

    If a timestamp cannot be parsed, the event is included conservatively
    (better to produce a spurious box than silently miss a real region).
    """
    try:
        t = datetime.strptime(date_obs[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Unparseable image timestamp — include all day's events as a fallback
        return day_events

    window = timedelta(minutes=HEK_WINDOW_MIN)
    t_lo, t_hi = t - window, t + window

    matched = []
    for ev in day_events:
        t_start = _parse_hek_time(ev.get("event_starttime")) or t
        t_end   = _parse_hek_time(ev.get("event_endtime"))   or t
        if t_start <= t_hi and t_end >= t_lo:
            matched.append(ev)
    return matched


# ---------------------------------------------------------------------------
# JSOC SHARP network helpers
# ---------------------------------------------------------------------------

def _parse_jsoc_time(raw: str) -> datetime | None:
    """Parse a JSOC timestamp like '2014.01.01_17:58:00_TAI' to datetime."""
    if not raw:
        return None
    parts = str(raw).split("_")  # ['2014.01.01', '17:58:00', 'TAI']
    if len(parts) < 2:
        return None
    date_part = parts[0].replace(".", "-")  # '2014-01-01'
    try:
        return datetime.strptime(f"{date_part} {parts[1]}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _query_day_jsoc_sharp(day: str) -> list[dict]:
    """Query JSOC jsoc_info.cgi for all SHARP patches on a given calendar day.

    Args:
        day: ISO date string 'YYYY-MM-DD'.

    Returns:
        A list of SHARP record dicts (possibly empty).  Never raises — errors
        are caught and logged, returning [] so the pipeline falls back to the
        area-based size estimate for that day.

    Uses jsoc_info.cgi (public, no auth) rather than jsoc_fetch.cgi (requires
    email registration).  Only metadata is fetched — no FITS files downloaded.
    """
    jsoc_day = day.replace("-", ".")      # '2014.01.01'
    # [] before time range means "all HARPNUMs" in this bi-dimensional series
    ds = f"{SHARP_SERIES}[][{jsoc_day}_00:00:00/1d@720s]"
    params = urllib.parse.urlencode({
        "op":  "rs_list",
        "ds":  ds,
        "key": SHARP_KEYS,
        "max": str(SHARP_MAX_RECORDS),
    })
    url = f"{JSOC_INFO_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=JSOC_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # jsoc_info returns column-major format: each keyword has its own
        # "values" list; record i = {kw["name"]: kw["values"][i] for all kw}.
        kws = data.get("keywords", [])
        if not kws or not kws[0].get("values"):
            return []
        names  = [kw["name"]   for kw in kws]
        cols   = [kw["values"] for kw in kws]
        return [dict(zip(names, [col[i] for col in cols]))
                for i in range(len(cols[0]))]
    except Exception as exc:
        rlog.warn(f"JSOC SHARP {day}: {type(exc).__name__} — skipping (will use fallback sizing)")
        return []


def _sharp_for_time(sharp_records: list[dict], date_obs: str) -> list[dict]:
    """Filter SHARP records to those within ±HEK_WINDOW_MIN of the image time.

    Unlike HEK events (which have a start/end lifetime), each SHARP record is
    a snapshot at a specific T_REC.  We include it if |T_REC − date_obs| ≤ window.
    """
    try:
        t = datetime.strptime(date_obs[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return sharp_records

    window = timedelta(minutes=HEK_WINDOW_MIN)
    matched = []
    for rec in sharp_records:
        t_rec = _parse_jsoc_time(rec.get("T_REC", ""))
        if t_rec is None or abs(t_rec - t) <= window:
            matched.append(rec)
    return matched


def _noaa_nums(raw: str) -> set[str]:
    """Parse a NOAA AR number string (possibly comma-separated) to a set.

    Normalises by stripping whitespace, 'AR' prefix, and leading zeros so that
    '11944', 'AR11944', and '011944' all produce {'11944'}.
    """
    if not raw or str(raw).strip().upper() in ("", "MISSING", "NONE"):
        return set()
    nums: set[str] = set()
    for part in str(raw).split(","):
        n = part.strip().upper().lstrip("AR").lstrip("0")
        if n.isdigit():
            nums.add(n)
    return nums


def _sharp_box_arcsec(sharp_rec: dict) -> tuple[float, float] | None:
    """Compute (width_arcsec, height_arcsec) from SHARP heliographic bounding box.

    LON_MIN/LON_MAX and LAT_MIN/LAT_MAX are the Stonyhurst heliographic extent
    of the SHARP patch in degrees.  Converting to arcseconds via the linear
    approximation (valid within ~800 arcsec of disk centre):

        arcsec_per_deg = π × R_sun_arcsec / 180
        width  = ΔLon × cos(lat_centre) × arcsec_per_deg
        height = ΔLat × arcsec_per_deg

    Returns None if any value is missing or suspiciously zero (MISSING placeholder).
    """
    try:
        lon_min = float(sharp_rec["LON_MIN"])
        lon_max = float(sharp_rec["LON_MAX"])
        lat_min = float(sharp_rec["LAT_MIN"])
        lat_max = float(sharp_rec["LAT_MAX"])
    except (KeyError, TypeError, ValueError):
        return None

    # JSOC returns 0.0 as a placeholder for MISSING values
    if lon_min == lon_max == 0.0 and lat_min == lat_max == 0.0:
        return None

    lat_c = (lat_min + lat_max) / 2.0
    arcsec_per_deg = np.pi * SOLAR_RADIUS_ARCSEC / 180.0

    w_arcsec = abs(lon_max - lon_min) * np.cos(np.radians(lat_c)) * arcsec_per_deg
    h_arcsec = abs(lat_max - lat_min) * arcsec_per_deg
    return float(w_arcsec), float(h_arcsec)


def _match_sharp_by_noaa(noaa_num: str, sharp_records: list[dict]) -> dict | None:
    """Return the SHARP record for the same NOAA AR, or None if not found.

    Picks the record with the smallest |T_REC − image_time| among all records
    that share the NOAA AR number.  The caller already filtered sharp_records to
    ±HEK_WINDOW_MIN, so we just take the first one with a matching NOAA number.
    """
    target = _noaa_nums(noaa_num)
    if not target:
        return None
    for rec in sharp_records:
        if target & _noaa_nums(rec.get("NOAA_ARS", "")):
            return rec
    return None


# ---------------------------------------------------------------------------
# Ground-truth NMS
# ---------------------------------------------------------------------------

def _nms(
    boxes: list[tuple[int, float, float, float, float]],
    iou_thresh: float = NMS_IOU_THRESH,
) -> list[int]:
    """Return the indices of boxes to keep after greedy non-maximum suppression.

    Args:
        boxes: list of (class_idx, cx, cy, w, h) — all normalised to [0, 1].
        iou_thresh: boxes whose IoU with an already-kept box exceeds this are
                    suppressed.  0.5 keeps distinct ARs that share a SHARP
                    patch but removes near-identical duplicate labels.

    Priority: larger area first (SHARP boxes are larger than fallback squares,
    so they naturally win when the two overlap).
    """
    if len(boxes) <= 1:
        return list(range(len(boxes)))

    # Convert (cx, cy, w, h) → (x1, y1, x2, y2, area)
    coords: list[tuple[float, float, float, float, float]] = []
    for _, cx, cy, w, h in boxes:
        coords.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, w * h))

    # Process largest-first so SHARP-matched (bigger) boxes beat fallback (smaller)
    order = sorted(range(len(boxes)), key=lambda i: -coords[i][4])

    keep: list[int] = []
    suppressed: set[int] = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        x1i, y1i, x2i, y2i, ai = coords[i]
        for j in order:
            if j in suppressed or j <= i:
                continue
            x1j, y1j, x2j, y2j, aj = coords[j]
            iw = max(0.0, min(x2i, x2j) - max(x1i, x1j))
            ih = max(0.0, min(y2i, y2j) - max(y1i, y1j))
            inter = iw * ih
            union = ai + aj - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed.add(j)

    return sorted(keep)


# ---------------------------------------------------------------------------
# Label writing
# ---------------------------------------------------------------------------

def generate_label(
    sidecar_path: Path,
    label_dir: Path,
    events: list | None = None,
    sharp_records: list[dict] | None = None,
) -> tuple[int, int, int]:
    """Write one YOLO .txt label file for the image described by sidecar_path.

    Args:
        sidecar_path:  Path to the WCS sidecar JSON written by preprocess.py.
        label_dir:     Directory where the .txt label file will be written.
        events:        Pre-fetched HEK event list (class + centroid).
                       Pass None only for standalone use.
        sharp_records: Pre-fetched JSOC SHARP records filtered to this image's
                       time window.  Used for pixel-precise bounding boxes.
                       Pass None to fall back to area-based estimation.

    Returns:
        (n_boxes, n_sharp, n_fallback) — total boxes written plus how many
        used SHARP for sizing vs. the area-based fallback.

    YOLO format: each line is '<class_id> <cx> <cy> <w> <h>' with all four
    geometry values normalised to [0, 1] relative to image dimensions.
    """
    meta     = json.loads(sidecar_path.read_text())
    img_size = meta["target_size"]

    if events is None:
        day_evs = _query_day_hek(meta["date_obs"][:10])
        events  = _events_for_image(day_evs, meta["date_obs"])

    if not events:
        (label_dir / f"{sidecar_path.stem}.txt").write_text("")
        return 0, 0, 0

    # Sort events highest-complexity-first so that when two NOAA ARs share the
    # same SHARP HARP patch the most complex class wins the slot.
    def _ev_cls(ev: dict) -> int:
        raw = ev.get("ar_mtwilsoncls") or ev.get("frm_specificid") or ""
        c = _normalise_mtwilson(str(raw))
        return c if c is not None else -1

    events = sorted(events, key=_ev_cls, reverse=True)

    # used_harps: once a SHARP patch (HARPNUM) has been assigned a label the
    # patch is "spent" — subsequent NOAA ARs in the same patch are skipped so
    # that the full-disk image never has two boxes that are identical in size
    # and heavily overlap just because two sub-regions share one HARP.
    used_harps: set[str] = set()

    n_sharp    = 0
    n_fallback = 0
    candidates: list[tuple[int, float, float, float, float]] = []
    # (cls_idx, cx_n, cy_n, w_n, h_n)

    for ev in events:
        raw_cls = ev.get("ar_mtwilsoncls") or ev.get("frm_specificid") or ""
        cls_idx = _normalise_mtwilson(str(raw_cls))
        if cls_idx is None:
            continue

        try:
            hpc_x = float(ev["hpc_x"])
            hpc_y = float(ev["hpc_y"])
        except (KeyError, TypeError, ValueError):
            continue

        # --- Bounding box: SHARP (preferred) or area estimate (fallback) ----
        cx, cy = _hpc_to_pixel(hpc_x, hpc_y, meta)

        noaa_raw  = str(ev.get("ar_noaanum") or "")
        sharp_rec = _match_sharp_by_noaa(noaa_raw, sharp_records or [])
        box_arcsec = _sharp_box_arcsec(sharp_rec) if sharp_rec is not None else None

        if box_arcsec is not None:
            harp_id = str(sharp_rec.get("HARPNUM", ""))  # type: ignore[union-attr]
            if harp_id and harp_id in used_harps:
                # This HARP is already represented by a higher-complexity AR —
                # skip to prevent duplicate same-patch labels.
                continue
            if harp_id:
                used_harps.add(harp_id)
            w_arcsec, h_arcsec = box_arcsec
            w_px = w_arcsec / abs(meta["cdelt1"]) * meta["scale_x"]
            h_px = h_arcsec / abs(meta["cdelt2"]) * meta["scale_y"]
            n_sharp += 1
        else:
            # No SHARP match — skip rather than emit an unreliable area-estimate box.
            continue
        # ---------------------------------------------------------------------

        cx_n = cx / img_size
        cy_n = cy / img_size
        w_n  = w_px / img_size
        h_n  = h_px / img_size

        if not (0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0):
            continue

        w_n = min(w_n, min(2 * cx_n, 2 * (1 - cx_n)))
        h_n = min(h_n, min(2 * cy_n, 2 * (1 - cy_n)))
        w_n = max(w_n, MIN_BOX_NORM)
        h_n = max(h_n, MIN_BOX_NORM)

        candidates.append((cls_idx, cx_n, cy_n, w_n, h_n))

    # Final NMS pass — removes any remaining spatial overlaps between boxes
    # from distinct HARPs that happen to be physically adjacent.
    keep = _nms(candidates)
    removed = len(candidates) - len(keep)
    n_sharp    = max(0, n_sharp    - removed)
    n_fallback = max(0, n_fallback - removed)

    lines = [
        f"{candidates[i][0]} {candidates[i][1]:.6f} {candidates[i][2]:.6f} "
        f"{candidates[i][3]:.6f} {candidates[i][4]:.6f}"
        for i in keep
    ]

    (label_dir / f"{sidecar_path.stem}.txt").write_text("\n".join(lines))
    return len(lines), n_sharp, n_fallback


# ---------------------------------------------------------------------------
# Disk-cached day fetcher (used by generate_all_labels)
# ---------------------------------------------------------------------------

def _fetch_day(day: str, cache_dir: Path) -> tuple[str, list[dict], list[dict]]:
    """Load HEK events and SHARP records for one day, with disk cache.

    On first call for a given day the results are fetched over the network and
    written to cache_dir/hek_YYYY-MM-DD.json and sharp_YYYY-MM-DD.json.
    Subsequent calls (e.g. when re-running label generation after tweaking
    logic) read the cached files instantly without any network traffic.

    Returns (day, hek_events, sharp_records).  Both lists may be empty on
    network failure; the label writer handles this with the area fallback.
    """
    hek_file   = cache_dir / f"hek_{day}.json"
    sharp_file = cache_dir / f"sharp_{day}.json"

    if hek_file.exists():
        hek: list[dict] = json.loads(hek_file.read_text(encoding="utf-8"))
    else:
        hek = _query_day_hek(day)
        hek_file.write_text(json.dumps(hek, ensure_ascii=False), encoding="utf-8")

    if sharp_file.exists():
        sharp: list[dict] = json.loads(sharp_file.read_text(encoding="utf-8"))
    else:
        sharp = _query_day_jsoc_sharp(day)
        sharp_file.write_text(json.dumps(sharp, ensure_ascii=False), encoding="utf-8")

    return day, hek, sharp


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_labels(images_dir: str, labels_dir: str):
    """Generate YOLO labels for every sidecar JSON found in images_dir.

    Workflow:
      1. Read all sidecar JSONs and group them by calendar day (YYYY-MM-DD).
      2. Fetch HEK + SHARP for all days in parallel (up to 8 concurrent HTTP
         requests).  Results are written to data/.cache/ so that re-runs skip
         the network entirely.
      3. For each image, filter cached events to ±HEK_WINDOW_MIN of the
         observation time and write the YOLO label file.
      4. Print a box-source and class-distribution summary.
    """
    img_path  = Path(images_dir)
    lbl_path  = Path(labels_dir)
    cache_dir = img_path.parent / ".cache"
    lbl_path.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sidecars = sorted(img_path.glob("*.json"))
    if not sidecars:
        rlog.warn(f"No sidecar JSON files found in {img_path}")
        return

    day_map: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for sc in sidecars:
        meta = json.loads(sc.read_text())
        day_map[meta["date_obs"][:10]].append((sc, meta))

    days      = sorted(day_map.keys())
    n_days    = len(days)
    n_workers = min(n_days, 8)

    # Count cached days for the info table
    cached = sum(
        1 for d in days
        if (cache_dir / f"hek_{d}.json").exists()
        and (cache_dir / f"sharp_{d}.json").exists()
    )
    cache_note = f"all cached" if cached == n_days else f"{cached}/{n_days} days cached"

    rlog.kv_table([
        ("Images",   f"{img_path}/  ({len(sidecars)} files, {n_days} days)"),
        ("Labels",   f"{lbl_path}/"),
        ("Cache",    f"{cache_dir}/  ({cache_note})"),
        ("Strategy", f"HEK + JSOC SHARP  ·  {n_workers} parallel fetches  ·  ±{HEK_WINDOW_MIN} min match"),
    ])

    # --- Phase 1: fetch all days (parallel, disk-cached) ---
    day_data: dict[str, tuple[list[dict], list[dict]]] = {}
    hek_timeouts   = 0
    sharp_timeouts = 0

    with rlog.make_progress("Fetching catalogue data") as progress:
        fetch_task = progress.add_task("days", total=n_days)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fetch_day, day, cache_dir): day for day in days}
            for fut in as_completed(futures):
                day, hek, sharp = fut.result()
                day_data[day] = (hek, sharp)
                if not hek:
                    hek_timeouts += 1
                if not sharp:
                    sharp_timeouts += 1
                progress.advance(fetch_task)

    # --- Phase 2: write label files (sequential — fast, pure CPU/disk) ---
    total_boxes    = 0
    total_sharp    = 0
    total_fallback = 0

    with rlog.make_progress("Writing labels") as progress:
        img_task = progress.add_task("images", total=len(sidecars))
        for day in days:
            day_events, day_sharp = day_data[day]
            for sc, meta in day_map[day]:
                img_events = _events_for_image(day_events, meta["date_obs"])
                img_sharp  = _sharp_for_time(day_sharp,   meta["date_obs"])
                n, ns, nf  = generate_label(sc, lbl_path,
                                            events=img_events,
                                            sharp_records=img_sharp)
                total_boxes    += n
                total_sharp    += ns
                total_fallback += nf
                progress.advance(img_task)

    rlog.success(f"{total_boxes} boxes across {len(sidecars)} images")
    rlog.kv_table([
        ("Box size source", ""),
        ("  SHARP (precise)", f"{total_sharp} boxes ({100*total_sharp/max(total_boxes,1):.0f}%)"),
        ("  area fallback",   f"{total_fallback} boxes ({100*total_fallback/max(total_boxes,1):.0f}%)"),
    ])
    if hek_timeouts:
        rlog.warn(f"HEK: {hek_timeouts}/{n_days} days skipped")
    if sharp_timeouts:
        rlog.warn(f"JSOC SHARP: {sharp_timeouts}/{n_days} days with no data (fallback used)")

    # Tally boxes by class from the written label files to produce a
    # distribution table — useful for spotting severe class imbalance
    # before training
    counts: Counter[int] = Counter()
    for lbl in lbl_path.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    rlog.class_table(counts, CLASS_NAMES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def diagnose(day: str, max_records: int = 3) -> None:
    """Query both HEK and JSOC SHARP for *day* and print raw fields.

    Shows whether SHARP records can be matched to HEK events by NOAA AR number
    and whether the heliographic bounding box produces sensible arcsec sizes.

    Usage:
        python -m src.labels --diagnose 2014-01-07
    """
    import math

    # --- HEK ---
    hek_events = _query_day_hek(day)
    rlog.console.print(f"\n[bold cyan]HEK — {len(hek_events)} AR events for {day}[/bold cyan]")
    hek_fields = ["ar_mtwilsoncls", "frm_name", "ar_noaanum", "hpc_x", "hpc_y", "ar_area"]
    for i, ev in enumerate(hek_events[:max_records]):
        rlog.console.print(f"  [bold]Event {i + 1}[/bold]")
        for f in hek_fields:
            val = ev.get(f)
            colour = "green" if val not in (None, "", "None") else "red"
            rlog.console.print(f"    {f:20s} [{colour}]{val!r}[/{colour}]")
    if len(hek_events) > max_records:
        rlog.console.print(f"  … {len(hek_events) - max_records} more events omitted")

    # --- JSOC SHARP ---
    sharp_recs = _query_day_jsoc_sharp(day)
    rlog.console.print(f"\n[bold cyan]JSOC SHARP — {len(sharp_recs)} records for {day}[/bold cyan]")
    sharp_fields = ["T_REC", "HARPNUM", "NOAA_ARS", "LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"]
    seen: set[str] = set()
    shown = 0
    for rec in sharp_recs:
        harp = str(rec.get("HARPNUM", ""))
        if harp in seen:
            continue
        seen.add(harp)
        rlog.console.print(f"  [bold]HARP {harp}[/bold]")
        for f in sharp_fields:
            val = rec.get(f)
            colour = "green" if val not in (None, "", "MISSING", "0.000000") else "red"
            rlog.console.print(f"    {f:12s} [{colour}]{val!r}[/{colour}]")
        # Also show computed box size in arcsec
        box = _sharp_box_arcsec(rec)
        if box:
            w, h = box
            rlog.console.print(f"    {'box':12s} [green]{w:.0f} x {h:.0f} arcsec[/green]")
        else:
            rlog.console.print(f"    {'box':12s} [red]could not compute[/red]")
        shown += 1
        if shown >= max_records:
            break
    if len(seen) < len({str(r.get("HARPNUM")) for r in sharp_recs}):
        rlog.console.print(f"  … more HARPs omitted")

    # --- Matching test ---
    rlog.console.print(f"\n[bold cyan]Match test (first {max_records} HEK events with ar_noaanum)[/bold cyan]")
    matched = shown_m = 0
    for ev in hek_events:
        noaa_raw = str(ev.get("ar_noaanum") or "")
        if not noaa_raw or noaa_raw in ("", "None"):
            continue
        cls_raw = ev.get("ar_mtwilsoncls") or ev.get("frm_specificid") or ""
        if _normalise_mtwilson(str(cls_raw)) is None:
            continue
        # Filter SHARP to ±HEK_WINDOW_MIN of noon on the day (approximate)
        noon_dt = datetime.strptime(f"{day} 12:00:00", "%Y-%m-%d %H:%M:%S")
        nearby = [r for r in sharp_recs
                  if (t := _parse_jsoc_time(str(r.get("T_REC", "")))) is not None
                  and abs((t - noon_dt).total_seconds()) <= HEK_WINDOW_MIN * 60 * 4]
        rec = _match_sharp_by_noaa(noaa_raw, nearby or sharp_recs)
        box = _sharp_box_arcsec(rec) if rec else None
        status = "[green]SHARP match[/green]" if box else "[yellow]fallback[/yellow]"
        rlog.console.print(f"  NOAA {noaa_raw:>6s}  {cls_raw:>20s}  {status}")
        matched += (1 if box else 0)
        shown_m += 1
        if shown_m >= max_records:
            break
    rlog.console.print(f"  ({matched}/{shown_m} showed matched)")


def _parse_args():
    p = argparse.ArgumentParser(description="Generate YOLO labels from HEK catalogue")
    p.add_argument("--images",   default="data/images")
    p.add_argument("--labels",   default="data/labels")
    p.add_argument("--diagnose", metavar="YYYY-MM-DD",
                   help="Print raw HEK + JSOC SHARP fields for one day and exit (no labels written)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.diagnose:
        diagnose(args.diagnose)
    else:
        generate_all_labels(args.images, args.labels)
