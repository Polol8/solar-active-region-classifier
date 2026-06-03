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

Bulk-query strategy (1 HEK query per day)
  Making one HEK query per image (N queries) is fragile: each HTTP round-trip
  takes 2–10 s and may hang indefinitely.  Instead we make ONE query per
  calendar day, cache the day's events, and match them to individual image
  timestamps by checking whether the event's lifetime overlaps a ±30-minute
  window around the image observation time.  For a 1-month run this reduces
  117 queries to 31.

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
    "hpc_x", "hpc_y",  # centroid position in arcseconds (HPC frame)
    "ar_area",          # area in Millionths of Solar Hemisphere (MSH)
    "event_starttime",  # start of the AR's tracked lifetime
    "event_endtime",    # end of the AR's tracked lifetime
])

# Minimum normalised bounding-box side length.  Prevents degenerate 1-pixel
# boxes for very small ARs where the area estimate rounds to near-zero.
MIN_BOX_NORM         = 0.02

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

    Strips whitespace, underscores and slashes (common HEK formatting artefacts)
    before the dictionary lookup, so 'Beta-Gamma', 'betagamma', 'BG' all map
    to index 2.
    """
    if not raw:
        return None
    key = re.sub(r"[\s_/]", "", raw.strip().lower())
    return MTWILSON_MAP.get(key)


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
    # Circular-equivalent diameter with 1.5× safety margin
    box_native = 2 * np.sqrt(area_pix2 / np.pi) * 1.5
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
# Label writing
# ---------------------------------------------------------------------------

def generate_label(sidecar_path: Path, label_dir: Path, events: list | None = None) -> int:
    """Write one YOLO .txt label file for the image described by sidecar_path.

    Args:
        sidecar_path: Path to the WCS sidecar JSON written by preprocess.py.
        label_dir:    Directory where the .txt label file will be written.
        events:       Pre-fetched HEK event list.  Pass None only for
                      standalone use — then a fresh day query is made.

    Returns:
        Number of bounding boxes written (0 = background image with no ARs).

    YOLO format: each line is '<class_id> <cx> <cy> <w> <h>' with all four
    geometry values normalised to [0, 1] relative to image dimensions.
    Boxes whose centre falls outside the image, or that would have
    zero/negative clamped width/height, are silently discarded.
    """
    meta     = json.loads(sidecar_path.read_text())
    img_size = meta["target_size"]

    if events is None:
        # Standalone fallback: query HEK for this specific day and filter
        day_evs = _query_day_hek(meta["date_obs"][:10])
        events  = _events_for_image(day_evs, meta["date_obs"])

    if not events:
        # Write an empty file — YOLO treats this as a background (no-object) image
        (label_dir / f"{sidecar_path.stem}.txt").write_text("")
        return 0

    lines = []
    for ev in events:
        # ar_mtwilsoncls is the canonical Mount Wilson field.  frm_specificid
        # is a fallback used by some NOAA SRS and older SHARP pipeline entries.
        raw_cls = ev.get("ar_mtwilsoncls") or ev.get("frm_specificid") or ""
        cls_idx = _normalise_mtwilson(str(raw_cls))
        if cls_idx is None:
            # Event has no recognisable Mount Wilson class — skip it.
            # This is common for events detected by non-magnetic pipelines
            # (e.g. EUV-based detectors that report AR positions but no
            # magnetic classification).
            continue

        try:
            hpc_x = float(ev["hpc_x"])
            hpc_y = float(ev["hpc_y"])
        except (KeyError, TypeError, ValueError):
            continue  # missing or non-numeric position — skip

        try:
            area_msh = float(ev.get("ar_area") or 0)
        except (ValueError, TypeError):
            area_msh = 0.0  # missing area → use the 10 MSH floor in _estimate_box_size

        cx, cy = _hpc_to_pixel(hpc_x, hpc_y, meta)
        box_px = _estimate_box_size(area_msh, meta)

        # Normalise to [0, 1] relative to image side length
        cx_n = cx / img_size
        cy_n = cy / img_size
        w_n  = max(box_px / img_size, MIN_BOX_NORM)
        h_n  = max(box_px / img_size, MIN_BOX_NORM)

        # Discard events whose projected centre falls off-disk (behind the limb
        # or at extremely high latitudes), which can happen near solar maximum
        # when active regions emerge close to the east/west limb
        if not (0.0 <= cx_n <= 1.0 and 0.0 <= cy_n <= 1.0):
            continue

        # Clamp the box so it does not extend beyond the image boundary.
        # min(2*cx_n, 2*(1-cx_n)) is the maximum box width that keeps the
        # centre-anchored box within [0, 1].
        w_n = min(w_n, min(2 * cx_n, 2 * (1 - cx_n)))
        h_n = min(h_n, min(2 * cy_n, 2 * (1 - cy_n)))

        lines.append(f"{cls_idx} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

    (label_dir / f"{sidecar_path.stem}.txt").write_text("\n".join(lines))
    return len(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_labels(images_dir: str, labels_dir: str):
    """Generate YOLO labels for every sidecar JSON found in images_dir.

    Workflow:
      1. Read all sidecar JSONs and group them by calendar day (YYYY-MM-DD).
      2. For each unique day, make one HEK REST request covering 00:00–23:59.
      3. Cache the day's events in memory.
      4. For each image in that day, filter cached events to ±HEK_WINDOW_MIN
         around the image timestamp and write the YOLO label file.
      5. After all images, print a class-distribution summary table.
    """
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)
    lbl_path.mkdir(parents=True, exist_ok=True)

    sidecars = sorted(img_path.glob("*.json"))
    if not sidecars:
        rlog.warn(f"No sidecar JSON files found in {img_path}")
        return

    # Group sidecars by calendar day so that images from the same day share
    # a single HEK query result (bulk-query strategy described in module doc)
    day_map: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for sc in sidecars:
        meta = json.loads(sc.read_text())
        day_map[meta["date_obs"][:10]].append((sc, meta))

    n_days = len(day_map)
    rlog.kv_table([
        ("Images",   f"{img_path}/  ({len(sidecars)} files, {n_days} days)"),
        ("Labels",   f"{lbl_path}/"),
        ("Strategy", f"1 HEK query/day  ·  {HEK_TIMEOUT_SEC}s timeout  ·  ±{HEK_WINDOW_MIN} min match"),
    ])

    total_boxes = 0
    timed_out   = 0

    with rlog.make_progress("Generating labels") as progress:
        day_task = progress.add_task("[dim]HEK queries[/dim]", total=n_days)
        img_task = progress.add_task("writing labels",         total=len(sidecars))

        for day in sorted(day_map.keys()):
            day_events = _query_day_hek(day)
            if not day_events:
                timed_out += 1
            progress.advance(day_task)

            for sc, meta in day_map[day]:
                img_events   = _events_for_image(day_events, meta["date_obs"])
                total_boxes += generate_label(sc, lbl_path, events=img_events)
                progress.advance(img_task)

    rlog.success(f"{total_boxes} boxes across {len(sidecars)} images")
    if timed_out:
        rlog.warn(f"{timed_out}/{n_days} days skipped (HEK timeout or empty)")

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

def _parse_args():
    p = argparse.ArgumentParser(description="Generate YOLO labels from HEK catalogue")
    p.add_argument("--images", default="data/images")
    p.add_argument("--labels", default="data/labels")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_all_labels(args.images, args.labels)
