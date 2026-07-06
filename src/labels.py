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
  117 queries to 31.  Those per-day queries are independent of each other, so
  generate_all_labels fans them out across HEK_MAX_CONCURRENT_QUERIES worker
  threads instead of making them one at a time.

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

# generate_all_labels fans out day-queries to this many worker threads.
# _query_day_hek uses blocking urllib (not asyncio), so a thread pool — not
# an event loop — is what actually overlaps the network waits.  Kept modest
# to stay polite to the HEK server (matches the same default used for JSOC
# file downloads in src/download.py).
HEK_MAX_CONCURRENT_QUERIES = 5

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
    "frm_name",         # detection pipeline name — used to keep only the
                         # authoritative source (see AUTHORITATIVE_FRM_SUBSTR)
    "ar_noaanum",        # NOAA AR number — used to de-duplicate repeated
                         # reports of the same physical region
    "hpc_x", "hpc_y",  # centroid position in arcseconds (HPC frame).  Despite
                         # NOAA's SRS report nominally applying at 00:00 UT,
                         # this field empirically already tracks close to the
                         # actual image time (verified against real magnetogram
                         # features) — do NOT re-derive/rotate it from
                         # event_starttime, that was tried and made positions
                         # measurably worse (shifted off the real feature).
    "hpc_bbox",         # WKT polygon bounding box in HPC arcsec.  Reliable as
                         # a physical extent from the HMI SHARP and SPoCA
                         # pipelines; the NOAA pipeline's own hpc_bbox spans
                         # the region's full tracked lifetime, not its
                         # instantaneous size (see SHARP_FRM_SUBSTR) —
                         # fetched for every event regardless of source so
                         # SHARP/SPoCA values can be cross-matched onto the
                         # NOAA event (by ar_noaanum for SHARP, by position
                         # for SPoCA — see SPOCA_FRM_SUBSTR)
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

# HEK aggregates several independent detection pipelines under event_type=ar
# (HMI SHARP patches, SPoCA, NOAA SWPC's SRS reports).  Only the NOAA SWPC
# pipeline reliably populates ar_mtwilsoncls — the others report position/area
# only.  A substring match (rather than an exact match) tolerates the label
# having changed across HEK's history (e.g. "NOAA SWPC Observer").
AUTHORITATIVE_FRM_SUBSTR = "noaa"

# Substring identifying the HMI SHARP pipeline, whose hpc_bbox is the actual
# magnetic-patch cutout extent (confirmed against live HEK data: aspect
# ratios ~0.4-3.2, sizes of tens to hundreds of arcsec — physically
# plausible).  This is DIFFERENT from the NOAA pipeline's own hpc_bbox,
# which spans that region's full tracked lifetime and is dominated by
# rotational drift (confirmed aspect ratios up to 47:1, i.e. not a size
# measurement at all) — so box *size* is sourced from SHARP, cross-matched
# by ar_noaanum, while box *classification* stays sourced from NOAA.
SHARP_FRM_SUBSTR = "sharp"

# SPoCA's hpc_bbox is also a reliable, physically plausible axis-aligned
# extent (confirmed against live HEK data: aspect ratios ~0.4-1.4) — used
# as a secondary size fallback for regions SHARP doesn't track (SHARP only
# tracks patches above its own automatic detection threshold; confirmed
# empirically that ~37% of authoritative NOAA events have no SHARP match).
# Unlike SHARP, SPoCA carries no ar_noaanum, so it can't be joined by
# number — matched to the NOAA event by spatial proximity instead (see
# _nearest_spoca_bbox).
SPOCA_FRM_SUBSTR = "spoca"

# Maximum distance (arcsec) between a NOAA event's position and a SPoCA
# detection for them to be considered the same region.  Real active
# regions are rarely closer than this to each other, so a match within
# this radius is very unlikely to be a coincidentally-nearby different
# region.
SPOCA_MATCH_RADIUS_ARCSEC = 150.0

# Safety margin applied to SHARP's/SPoCA's measured hpc_bbox extent.
# Smaller than _estimate_box_size's 1.5x margin because this is a measured
# extent, not an estimate derived from area alone.
HPC_BBOX_MARGIN = 1.1

# YOLO class names in index order (must match configs/solar.yaml).
CLASS_NAMES = ["Alpha", "Beta", "BetaGamma", "BetaGammaDelta"]

# Mapping from raw HEK string → integer class index.  Multiple aliases exist
# because different HEK pipelines (NOAA SRS, SHARP, SOON) use slightly
# different capitalisation and separators.
# Exact-match table.  Keys are pre-normalised (letters only, lowercase) since
# _normalise_mtwilson strips everything else before the lookup — hyphenated
# variants like "Beta-Gamma" collapse to "betagamma" and don't need a
# separate entry here.
MTWILSON_MAP = {
    # Alpha — single dominant polarity
    "alpha": 0, "a": 0,
    # Beta — clean bipolar pair
    "beta": 1,  "b": 1,
    # BetaGamma — bipolar with complex inversion line
    "betagamma": 2, "bg": 2,
    # Gamma and BetaDelta are rare but share BetaGamma's complexity level
    "gamma": 2,     "betadelta": 2,
    # BetaGammaDelta — highest complexity, delta umbrae present
    "betagammadelta": 3, "bgd": 3,
}

# ---------------------------------------------------------------------------
# Pure helpers — no I/O, fully unit-testable
# ---------------------------------------------------------------------------

def _normalise_mtwilson(raw: str) -> int | None:
    """Return a class index for a raw Mount Wilson string, or None if unknown.

    Strips everything but letters (whitespace, underscores, slashes, hyphens
    — common HEK formatting artefacts) before the dictionary lookup, so
    'Beta-Gamma', 'betagamma', 'BG' all map to index 2.

    If the exact-match lookup fails, falls back to a substring classification
    (highest complexity tier first) to tolerate formatting artefacts seen in
    real NOAA SRS data that a dictionary lookup can't anticipate — e.g. a
    doubled letter like 'BETAAGAMMA-DELTA' — rather than silently dropping
    the event.
    """
    if not raw:
        return None
    key = re.sub(r"[^a-z]", "", raw.strip().lower())
    if not key:
        return None

    if key in MTWILSON_MAP:
        return MTWILSON_MAP[key]

    if "delta" in key:
        return 3 if "gamma" in key else 2
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
    # Circular-equivalent diameter with 1.5× safety margin
    box_native = 2 * np.sqrt(area_pix2 / np.pi) * 1.5
    # Scale to the resized PNG
    return float(box_native * meta["scale_x"])


def _filter_authoritative(events: list) -> list:
    """Keep only events from the authoritative classification source, and
    drop duplicate reports of the same numbered active region.

    HEK aggregates several independent pipelines under event_type='ar'; only
    frm_name containing "NOAA" reliably populates ar_mtwilsoncls (see
    AUTHORITATIVE_FRM_SUBSTR).  Events sharing the same ar_noaanum within
    that source are duplicate reports of the same region — keep the first.
    """
    seen_noaanum = set()
    kept = []
    for ev in events:
        frm_name = str(ev.get("frm_name") or "")
        if AUTHORITATIVE_FRM_SUBSTR not in frm_name.lower():
            continue
        noaanum = ev.get("ar_noaanum")
        if noaanum:
            if noaanum in seen_noaanum:
                continue
            seen_noaanum.add(noaanum)
        kept.append(ev)
    return kept


def _sharp_lookup(events: list, image_time: datetime | None) -> dict:
    """Map ar_noaanum -> the HMI SHARP event whose report time is closest to
    image_time.

    Unlike NOAA's once-daily report, HEK carries several SHARP observations
    per day for the same region (confirmed against live HEK data: ~4-hour
    spaced windows spanning the full day) — using whichever one HEK happens
    to list first (as an earlier version of this code did) picks a
    position/size up to ~24h stale relative to the image, which is exactly
    the kind of error this function exists to avoid.  SHARP is used for
    both position and size (see SHARP_FRM_SUBSTR); NOAA remains the only
    source of Mount Wilson classification.
    """
    by_noaanum: dict = defaultdict(list)
    for ev in events:
        if SHARP_FRM_SUBSTR not in str(ev.get("frm_name") or "").lower():
            continue
        noaanum = ev.get("ar_noaanum")
        if noaanum:
            by_noaanum[noaanum].append(ev)

    if image_time is None:
        return {noaanum: evs[0] for noaanum, evs in by_noaanum.items()}

    def _time_delta(ev):
        t = _parse_hek_time(ev.get("event_starttime"))
        return abs((image_time - t).total_seconds()) if t else float("inf")

    return {noaanum: min(evs, key=_time_delta) for noaanum, evs in by_noaanum.items()}


def _nearest_spoca_bbox(events: list, hpc_x: float, hpc_y: float) -> str | None:
    """Return the hpc_bbox of the closest HMI SPoCA detection to (hpc_x,
    hpc_y), within SPOCA_MATCH_RADIUS_ARCSEC, or None if none is close
    enough (or no SPoCA events are present).

    SPoCA carries no ar_noaanum (see SPOCA_FRM_SUBSTR), so it can't be
    joined by number like SHARP — spatial proximity to the already-resolved
    NOAA/SHARP position is used instead.
    """
    best_bbox = None
    best_dist = SPOCA_MATCH_RADIUS_ARCSEC
    for ev in events:
        if SPOCA_FRM_SUBSTR not in str(ev.get("frm_name") or "").lower():
            continue
        try:
            ex, ey = float(ev["hpc_x"]), float(ev["hpc_y"])
        except (KeyError, TypeError, ValueError):
            continue
        dist = ((ex - hpc_x) ** 2 + (ey - hpc_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_bbox = ev.get("hpc_bbox")
    return best_bbox


_HPC_BBOX_RE = re.compile(r"POLYGON\s*\(\(([^)]+)\)\)", re.IGNORECASE)


def _parse_hpc_bbox(bbox_wkt: str | None) -> tuple[float, float] | None:
    """Parse a HEK hpc_bbox WKT polygon string into an axis-aligned
    (width_arcsec, height_arcsec) bounding box, or None if unparseable.

    HEK returns a closed ring of 'x y' vertex pairs, e.g.
    'POLYGON((-885.39 -170.3982,-878.922 -169.6044,...,-885.39 -170.3982))'.
    The polygon isn't guaranteed to be axis-aligned, so this takes the
    envelope (min/max of the vertices) rather than an exact rotated
    rectangle — sufficient for an axis-aligned YOLO box.
    """
    if not bbox_wkt:
        return None
    m = _HPC_BBOX_RE.search(bbox_wkt)
    if not m:
        return None
    try:
        pts = [tuple(map(float, p.split())) for p in m.group(1).split(",")]
        xs, ys = zip(*pts)
    except (ValueError, IndexError):
        return None
    return max(xs) - min(xs), max(ys) - min(ys)


# ---------------------------------------------------------------------------
# HEK network helpers — stdlib urllib only, no sunpy.net
# ---------------------------------------------------------------------------

def _query_day_hek(day: str) -> list | None:
    """Query the HEK REST API for all AR events on a given calendar day.

    Args:
        day: ISO date string 'YYYY-MM-DD'.

    Returns:
        A list of event dicts on success — possibly empty, meaning the query
        succeeded and genuinely found no AR events that day.  Returns None if
        the request itself failed (network error, timeout, malformed
        response).  Callers MUST distinguish these two cases: None means
        "unknown, don't write a label", not "confirmed background image".
        Never raises — network errors are caught and logged as warnings.

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
        return None


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

def generate_label(
    sidecar_path: Path,
    label_dir: Path,
    events: list | None = None,
    unmapped_counter: Counter | None = None,
) -> int:
    """Write one YOLO .txt label file for the image described by sidecar_path.

    Args:
        sidecar_path: Path to the WCS sidecar JSON written by preprocess.py.
        label_dir:    Directory where the .txt label file will be written.
        events:       Pre-fetched HEK event list.  Pass None only for
                      standalone use — then a fresh day query is made.
        unmapped_counter: Optional Counter to accumulate raw Mount Wilson
                      strings that failed to normalise, for an aggregate
                      end-of-run report (see generate_all_labels).

    Returns:
        Number of bounding boxes written (0 = background image with no ARs).
        -1 if events is None and the standalone HEK query failed (network
        error) — no label file is written in that case, so dataset.py's
        existing skip-if-missing logic excludes the image instead of it
        being mislabelled as a confirmed background observation.

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
        if day_evs is None:
            # Network/HEK failure — do not write a label file; the image
            # should be treated as unlabelled, not a confirmed background.
            return -1
        events = _events_for_image(day_evs, meta["date_obs"])

    # SHARP is a reliable source of both position and size (see
    # SHARP_FRM_SUBSTR); NOAA's own hpc_bbox is not, and NOAA's once-daily
    # report is coarser in time than SHARP's several-times-a-day cadence.
    # Build the lookup from the full, unfiltered event list — keyed to
    # whichever SHARP report is closest in time to this image — before
    # narrowing down to the authoritative (NOAA) events used for
    # classification.
    image_time       = _parse_hek_time(meta.get("date_obs"))
    sharp_by_noaanum = _sharp_lookup(events, image_time)
    all_events       = events   # kept for the SPoCA fallback below (no ar_noaanum to join on)
    events           = _filter_authoritative(events)

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
            if unmapped_counter is not None and raw_cls:
                unmapped_counter[str(raw_cls)] += 1
            continue

        try:
            hpc_x = float(ev["hpc_x"])
            hpc_y = float(ev["hpc_y"])
        except (KeyError, TypeError, ValueError):
            continue  # missing or non-numeric position — skip

        # Prefer the time-matched SHARP report's own position over NOAA's
        # once-daily position when available — SHARP's automated centroid,
        # refreshed every few hours, is closer to the image's exact
        # observation time than NOAA's single daily report (see
        # _sharp_lookup).  Falls back to NOAA's position when no SHARP
        # report exists for this region (e.g. very small/new regions).
        sharp_ev = sharp_by_noaanum.get(ev.get("ar_noaanum"))
        if sharp_ev is not None:
            try:
                hpc_x = float(sharp_ev["hpc_x"])
                hpc_y = float(sharp_ev["hpc_y"])
            except (KeyError, TypeError, ValueError):
                pass  # malformed SHARP position — keep NOAA's

        try:
            area_msh = float(ev.get("ar_area") or 0)
        except (ValueError, TypeError):
            area_msh = 0.0  # missing area → use the 10 MSH floor in _estimate_box_size

        cx, cy = _hpc_to_pixel(hpc_x, hpc_y, meta)

        # Prefer SHARP's measured bounding box over the area-based estimate
        # (ar_area is essentially never populated in practice) — see
        # SHARP_FRM_SUBSTR for why NOAA's own hpc_bbox is not used here.
        # SPoCA is a secondary fallback for the ~37% of regions SHARP
        # doesn't track (see SPOCA_FRM_SUBSTR), matched by position since
        # it carries no ar_noaanum to join on.
        bbox_wkt = sharp_ev.get("hpc_bbox") if sharp_ev else None
        if not bbox_wkt:
            bbox_wkt = _nearest_spoca_bbox(all_events, hpc_x, hpc_y)
        bbox_arcsec = _parse_hpc_bbox(bbox_wkt)
        if bbox_arcsec is not None:
            w_arcsec, h_arcsec = bbox_arcsec
            box_w_px = w_arcsec / meta["cdelt1"] * meta["scale_x"] * HPC_BBOX_MARGIN
            box_h_px = h_arcsec / meta["cdelt2"] * meta["scale_y"] * HPC_BBOX_MARGIN
        else:
            box_w_px = box_h_px = _estimate_box_size(area_msh, meta)

        # Normalise to [0, 1] relative to image side length
        cx_n = cx / img_size
        cy_n = cy / img_size
        w_n  = max(box_w_px / img_size, MIN_BOX_NORM)
        h_n  = max(box_h_px / img_size, MIN_BOX_NORM)

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
        ("Strategy", f"{HEK_MAX_CONCURRENT_QUERIES} concurrent HEK queries  ·  "
                     f"{HEK_TIMEOUT_SEC}s timeout  ·  ±{HEK_WINDOW_MIN} min match"),
    ])

    total_boxes = 0
    failed_days = []
    unmapped: Counter = Counter()
    days = sorted(day_map.keys())

    with rlog.make_progress("Generating labels") as progress:
        day_task = progress.add_task("[dim]HEK queries[/dim]", total=n_days)
        img_task = progress.add_task("writing labels",         total=len(sidecars))

        # Fan the day-queries out to a thread pool — _query_day_hek blocks on
        # urllib, so threads (not asyncio) are what let these network waits
        # overlap.  Fetch everything first, then write labels sequentially
        # (fast — no network I/O left at that point).
        day_events: dict[str, list | None] = {}
        with ThreadPoolExecutor(max_workers=HEK_MAX_CONCURRENT_QUERIES) as executor:
            futures = {executor.submit(_query_day_hek, day): day for day in days}
            for future in as_completed(futures):
                day_events[futures[future]] = future.result()
                progress.advance(day_task)

        for day in days:
            events_for_day = day_events[day]

            if events_for_day is None:
                # Network/HEK failure — skip every image for this day rather
                # than writing a false "confirmed background" empty label.
                failed_days.append(day)
                progress.advance(img_task, advance=len(day_map[day]))
                continue

            for sc, meta in day_map[day]:
                img_events   = _events_for_image(events_for_day, meta["date_obs"])
                total_boxes += generate_label(sc, lbl_path, events=img_events,
                                               unmapped_counter=unmapped)
                progress.advance(img_task)

    rlog.success(f"{total_boxes} boxes across {len(sidecars)} images")
    if failed_days:
        n_skipped_imgs = sum(len(day_map[d]) for d in failed_days)
        rlog.warn(f"{len(failed_days)}/{n_days} days failed (HEK network error) — "
                  f"{n_skipped_imgs} images left unlabelled (excluded from dataset)")
    if unmapped:
        top = ", ".join(f"{cls!r} x{n}" for cls, n in unmapped.most_common(10))
        rlog.warn(f"{sum(unmapped.values())} events skipped — unrecognised Mount Wilson class: {top}")

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
