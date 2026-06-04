"""
Download SDO/HMI line-of-sight magnetograms from JSOC.

JSOC export workflow  (three HTTP steps):
  1. GET jsoc_fetch?op=exp_request  →  receive a requestid
  2. Poll jsoc_fetch?op=exp_status  →  wait for status = 0 (ready)
  3. Stream-download each file URL  →  save to disk

JSOC query format:
  series[T_START/DURATION@CADENCE]{SEGMENT}
  e.g.  hmi.M_720s[2014.01.07_00:00:00_TAI/1d@360m]{magnetogram}

Usage:
    python -m src.download --start 2014-01-07 --end 2014-01-07 \\
                           --email you@example.com --output data/raw
"""

import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from src import log as rlog

JSOC_URL    = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch"
HMI_SERIES  = "hmi.M_720s"
HMI_SEGMENT = "magnetogram"

EXPORT_TIMEOUT_SEC = 600   # max wait for JSOC to prepare export
POLL_INTERVAL_SEC  = 15    # seconds between status polls
HTTP_TIMEOUT_SEC   = 45    # per-request timeout


def _build_query(start: str, end: str, cadence_hours: float) -> str:
    """Return a JSOC record-set query string for the given time range."""
    start_dt    = datetime.strptime(start, "%Y-%m-%d")
    end_dt      = datetime.strptime(end,   "%Y-%m-%d")
    days        = (end_dt - start_dt).days + 1
    t_start     = start_dt.strftime("%Y.%m.%d_00:00:00_TAI")
    cadence_min = int(cadence_hours * 60)
    return f"{HMI_SERIES}[{t_start}/{days}d@{cadence_min}m]{{{HMI_SEGMENT}}}"


async def _run_download(start: str, end: str, email: str,
                        output_path: Path, cadence_hours: float) -> list[str]:
    """Async implementation — called by download() via asyncio.run()."""

    connector = aiohttp.TCPConnector()
    timeout   = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        # ── Step 1: submit export request ────────────────────────────────────
        rlog.info("Submitting export request to JSOC…")
        async with session.get(JSOC_URL, params={
            "op":       "exp_request",
            "ds":       _build_query(start, end, cadence_hours),
            "method":   "url",
            "protocol": "fits",
            "notify":   email,
            "format":   "json",
        }) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        status = int(data.get("status", -1))
        req_id = data.get("requestid", "")

        if status < 0 or not req_id:
            rlog.error(f"JSOC rejected the export (status {status}): {data.get('message','')}")
            rlog.info(f"  Ensure {email!r} is registered at jsoc.stanford.edu")
            return []

        rlog.info(f"Request ID: [cyan]{req_id}[/cyan]")

        # ── Step 2: poll until ready ──────────────────────────────────────────
        if status != 0:
            waited = 0
            with rlog.make_progress("Waiting for JSOC export") as progress:
                task = progress.add_task("", total=None)

                while True:
                    await asyncio.sleep(POLL_INTERVAL_SEC)
                    waited += POLL_INTERVAL_SEC
                    progress.advance(task)

                    async with session.get(JSOC_URL, params={
                        "op":        "exp_status",
                        "requestid": req_id,
                        "format":    "json",
                    }) as r:
                        r.raise_for_status()
                        poll = await r.json(content_type=None)

                    poll_status = int(poll.get("status", -1))
                    if poll_status == 0:
                        data = poll
                        break
                    elif poll_status < 0:
                        rlog.error(f"JSOC export failed (status {poll_status})")
                        return []
                    elif waited >= EXPORT_TIMEOUT_SEC:
                        rlog.warn(f"Timed out after {EXPORT_TIMEOUT_SEC}s.")
                        return []

        # ── Step 3: download files ────────────────────────────────────────────
        records = data.get("data", [])
        if not records:
            rlog.warn("Export succeeded but returned 0 files.")
            return []

        # dir is a server-relative path; filenames are relative to it.
        # Some exports include a full "url" per record; others only have "filename".
        raw_dir = data.get("dir", "").rstrip("/")
        base = f"http://jsoc.stanford.edu{raw_dir}" if raw_dir.startswith("/") else raw_dir
        urls = [
            r.get("url") or f"{base}/{r['filename']}"
            for r in records
            if r.get("url") or r.get("filename")
        ]
        rlog.info(f"[cyan]{len(urls)}[/cyan] files ready — downloading…")

        downloaded = []
        with rlog.make_progress("Downloading") as progress:
            task = progress.add_task("", total=len(urls))
            for url in urls:
                filename = output_path / url.split("/")[-1]
                try:
                    async with session.get(url) as r:
                        r.raise_for_status()
                        with open(filename, "wb") as f:
                            async for chunk in r.content.iter_chunked(1 << 20):
                                f.write(chunk)
                    downloaded.append(str(filename))
                except Exception as exc:
                    rlog.warn(f"  Failed: {url.split('/')[-1]} — {exc}")
                progress.advance(task)

    return downloaded


def download(start: str, end: str, email: str, output_dir: str, cadence_hours: float = 6.0):
    """Download HMI magnetograms and return a list of local file paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rlog.kv_table([
        ("Time range", f"{start}  to  {end}"),
        ("Cadence",    f"{cadence_hours} h  ({int(cadence_hours * 60)} min)"),
        ("Query",      _build_query(start, end, cadence_hours)),
        ("Output",     str(output_path)),
    ])

    files = asyncio.run(_run_download(start, end, email, output_path, cadence_hours))
    if files:
        rlog.success(f"{len(files)} files saved to [cyan]{output_path}/[/cyan]")
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Download SDO/HMI magnetograms from JSOC")
    p.add_argument("--start",   required=True)
    p.add_argument("--end",     required=True)
    p.add_argument("--email",   required=True)
    p.add_argument("--cadence", type=float, default=6.0)
    p.add_argument("--output",  default="data/raw")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download(args.start, args.end, args.email, args.output, args.cadence)
