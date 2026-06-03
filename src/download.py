"""
Download SDO/HMI line-of-sight magnetograms from JSOC via SunPy Fido.

sunpy.net is imported lazily (inside download()) because it makes network
calls at import time that block the terminal silently.  A spinner is shown
while the import loads and while JSOC processes the export request.

Usage:
    python -m src.download --start 2014-01-01 --end 2014-03-31 \
                           --email you@example.com --cadence 6 --output data/raw
"""

import argparse
from pathlib import Path

from src import log as rlog

HMI_SERIES = "hmi.M_720s"


def download(start: str, end: str, email: str, output_dir: str, cadence_hours: float = 6.0):
    """
    Download HMI magnetograms and return list of downloaded file paths.

    Args:
        start:          Start date, e.g. '2014-01-01'.
        end:            End date, e.g. '2014-03-31'.
        email:          Email registered with JSOC (required for data export).
        output_dir:     Local directory to save FITS files.
        cadence_hours:  Sampling interval in hours (default 6 h → 4 images/day).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rlog.kv_table([
        ("Time range", f"{start}  to  {end}"),
        ("Cadence",    f"{cadence_hours} h"),
        ("Series",     HMI_SERIES),
        ("Output",     str(output_path)),
    ])

    # sunpy.net connects to servers during import — show a spinner so the
    # terminal doesn't appear frozen while the module loads.
    with rlog.console.status("[cyan]Loading sunpy.net…[/cyan]", spinner="dots"):
        import astropy.units as u
        from sunpy.net import Fido, attrs as a

    with rlog.console.status("[cyan]Querying JSOC…[/cyan]", spinner="dots"):
        results = Fido.search(
            a.Time(start, end),
            a.jsoc.Series(HMI_SERIES),
            a.jsoc.Notify(email),
            a.Sample(cadence_hours * u.hour),
        )

    n_found = len(results)
    if n_found == 0:
        rlog.warn("No files found. Check time range or JSOC availability.")
        return []

    rlog.info(f"[cyan]{n_found}[/cyan] export(s) queued — downloading (Fido progress below)")
    files = Fido.fetch(results, path=str(output_path / "{file}"))

    rlog.success(f"{len(files)} files saved to [cyan]{output_path}/[/cyan]")
    return list(files)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Download SDO/HMI magnetograms from JSOC")
    p.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    p.add_argument("--email",   required=True, help="Email registered with JSOC")
    p.add_argument("--cadence", type=float, default=6.0)
    p.add_argument("--output",  default="data/raw")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download(args.start, args.end, args.email, args.output, args.cadence)
