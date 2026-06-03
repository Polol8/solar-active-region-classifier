"""
Convert SDO/HMI FITS magnetograms to normalised PNG images suitable for YOLO.

What is a magnetogram?
  A line-of-sight (LOS) magnetogram measures the component of the solar
  magnetic field pointing toward (or away from) the observer.  HMI records
  one full-disk LOS magnetogram every 720 seconds (~12 minutes) in the
  'hmi.M_720s' JSOC series.  Pixel values are in Gauss (G): positive values
  mean field pointing toward Earth, negative means pointing away.

Why do we need a WCS sidecar?
  The YOLO label pipeline (labels.py) needs to convert active-region
  coordinates from the HEK catalogue (heliocentric arcseconds, HPC frame)
  into pixel coordinates in the PNG image.  That requires the World Coordinate
  System (WCS) parameters stored in the FITS header: reference pixel (CRPIX),
  reference coordinate (CRVAL), and plate scale (CDELT, arcsec/pixel).  We
  save these as a companion .json sidecar so that labels.py does not need to
  re-open every FITS file.

Why does sunpy.map import lazily?
  sunpy.net and sunpy.map make outbound network connections during module-level
  import (version checks, catalog downloads).  Importing them at the top of
  the file would silently freeze the terminal for 30–120 seconds before any
  user-visible output appears.  Moving the import inside process_directory()
  lets us show a spinner ('Loading sunpy.map…') so the user can see that
  something is happening.

Pipeline per FITS file:
  1. Load as a SunPy Map (reads FITS header + data, validates WCS).
  2. Cast to float32 and replace NaN (off-disk pixels) with 0.
  3. Clip to [−B_CLIP, +B_CLIP] Gauss.
  4. Linearly rescale to uint8 [0, 255].
  5. Resize to IMAGE_SIZE × IMAGE_SIZE pixels using Lanczos resampling.
  6. Save the PNG and the companion WCS sidecar JSON.

Usage:
    python -m src.preprocess --input data/raw --output data/images
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src import log as rlog

# Output resolution.  1024 × 1024 preserves enough detail to distinguish
# small Alpha-class regions (~50 px across) while keeping GPU memory usage
# manageable during training.
IMAGE_SIZE = 1024   # pixels

# Magnetogram clip range before normalisation.  Values outside [−B_CLIP, +B_CLIP]
# are saturated.  Rationale:
#   • Quiet-sun background noise:  typically ±50 G → maps to near-midpoint grey.
#   • Typical active-region fields: ±200–800 G → good contrast in [0, 255].
#   • Large delta-spot umbrae:      can reach ±2000–3000 G → saturate to
#     pure white/black, which is acceptable because YOLO only needs to localise
#     the region, not measure the exact field strength.
# A symmetric clip produces a neutral grey (≈127) for field-free quiet sun
# and uses the full contrast range for the active-region signal.
B_CLIP = 1000.0     # Gauss


def fits_to_png(fits_path: Path, output_dir: Path, size: int = IMAGE_SIZE, b_clip: float = B_CLIP):
    """Convert a single HMI FITS file to a normalised PNG + WCS sidecar JSON.

    Args:
        fits_path:  Path to the source FITS file.
        output_dir: Directory where the PNG and JSON sidecar will be written.
        size:       Side length of the output square PNG (pixels).
        b_clip:     Symmetric clip range in Gauss before [0, 255] mapping.

    Returns:
        (png_path, None)  on success.
        (None, error_str) on failure — the caller logs the error.
    """
    # sunpy.map is imported lazily (see module docstring).  The import is
    # cached by Python after the first call, so subsequent files incur no cost.
    import sunpy.map
    try:
        smap = sunpy.map.Map(str(fits_path))
    except Exception as exc:
        return None, str(exc)

    data = smap.data.astype(np.float32)

    # Off-disk pixels (outside the solar limb) are stored as NaN in HMI files.
    # Replace them with 0 so they map to neutral grey after normalisation
    # rather than causing undefined behaviour in clip/scale arithmetic.
    data = np.nan_to_num(data, nan=0.0)

    # Clip, then map linearly from [−b_clip, +b_clip] → [0.0, 255.0]:
    #   pixel = (value + b_clip) / (2 × b_clip) × 255
    # Negative field (e.g. −1000 G) → 0 (black)
    # Zero field                    → 127.5 (neutral grey)
    # Positive field (e.g. +1000 G) → 255 (white)
    data = np.clip(data, -b_clip, b_clip)
    data = (data + b_clip) / (2 * b_clip) * 255.0
    data = data.astype(np.uint8)

    # Lanczos resampling is the highest-quality PIL downsampling filter,
    # minimising aliasing artefacts when shrinking from 4096 → 1024 px.
    img = Image.fromarray(data, mode="L")   # "L" = 8-bit greyscale
    img = img.resize((size, size), Image.LANCZOS)

    stem     = fits_path.stem
    png_path = output_dir / f"{stem}.png"
    img.save(str(png_path))

    orig_h, orig_w = smap.data.shape

    # Sidecar JSON stores the WCS parameters needed by labels.py to convert
    # HPC arcsecond coordinates → pixel coordinates in this resized PNG.
    # scale_x/scale_y encode the ratio between the resized and original
    # dimensions so downstream code can work in PNG pixel space.
    sidecar = {
        "fits_file":   fits_path.name,
        "date_obs":    str(smap.date),
        "orig_width":  orig_w,
        "orig_height": orig_h,
        "target_size": size,
        "scale_x":     size / orig_w,   # multiply native px coords by this
        "scale_y":     size / orig_h,   # to get PNG pixel coords
        # WCS reference values (FITS 1-based convention; labels.py subtracts 1)
        "crpix1": float(smap.meta.get("crpix1", orig_w / 2)),  # ref pixel, X
        "crpix2": float(smap.meta.get("crpix2", orig_h / 2)),  # ref pixel, Y
        "cdelt1": float(smap.meta.get("cdelt1", 0.5)),          # arcsec/pixel, X
        "cdelt2": float(smap.meta.get("cdelt2", 0.5)),          # arcsec/pixel, Y
        "crval1": float(smap.meta.get("crval1", 0.0)),          # ref coord arcsec, X
        "crval2": float(smap.meta.get("crval2", 0.0)),          # ref coord arcsec, Y
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))

    return png_path, None


def process_directory(input_dir: str, output_dir: str, size: int = IMAGE_SIZE, b_clip: float = B_CLIP):
    """Convert all FITS files in input_dir to normalised PNGs in output_dir."""
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(in_path.glob("*.fits")) + sorted(in_path.glob("*.fit"))

    rlog.kv_table([
        ("Input",  f"{in_path}/  ({len(fits_files)} FITS files)"),
        ("Output", f"{out_path}/"),
        ("Clip",   f"±{b_clip:.0f} G  →  [0, 255]"),
        ("Size",   f"{size}×{size} px"),
    ])

    # Warm up the lazy sunpy.map import with a visible spinner.  The import
    # itself can take 10–60 s on a cold start because sunpy loads astronomical
    # catalogs and performs version/network checks at import time.
    with rlog.console.status("[cyan]Loading sunpy.map…[/cyan]", spinner="dots"):
        import sunpy.map  # noqa: F401  (import for side-effect: warm up the cache)

    ok, failed = 0, []
    with rlog.make_progress("Converting") as progress:
        task = progress.add_task("", total=len(fits_files))
        for f in fits_files:
            result, err = fits_to_png(f, out_path, size=size, b_clip=b_clip)
            if result:
                ok += 1
            else:
                failed.append((f.name, err))
            progress.advance(task)

    if failed:
        rlog.warn(f"{ok} converted  ·  [bold red]{len(failed)} failed[/bold red]")
        for name, reason in failed:
            rlog.error(f"  {name}: {reason}")
    else:
        rlog.success(f"{ok} files converted to [cyan]{out_path}/[/cyan]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Convert HMI FITS to normalised PNG")
    p.add_argument("--input",  default="data/raw",    help="Directory with FITS files")
    p.add_argument("--output", default="data/images", help="Output directory for PNGs")
    p.add_argument("--size",   type=int,   default=IMAGE_SIZE)
    p.add_argument("--bclip",  type=float, default=B_CLIP)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_directory(args.input, args.output, size=args.size, b_clip=args.bclip)
