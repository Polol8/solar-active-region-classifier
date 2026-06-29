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
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

from src import log as rlog

JSOC_URL    = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch"
HMI_SERIES  = "hmi.M_720s"
HMI_SEGMENT = "magnetogram"

EXPORT_TIMEOUT_SEC  = 600   # max wait for JSOC to prepare each chunk export
POLL_INTERVAL_SEC   = 15    # seconds between status polls
CONNECT_TIMEOUT_SEC = 30    # TCP connection timeout
READ_TIMEOUT_SEC    = 120   # per-request read timeout
MAX_RETRIES         = 3     # retries for transient network failures
CHUNK_DAYS          = 90    # split large date ranges into ~3-month batches


def _date_chunks(start: str, end: str):
    """Yield (t_start_tai, days) pairs covering [start, end] in CHUNK_DAYS steps."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    while s <= e:
        chunk_end = min(s + timedelta(days=CHUNK_DAYS - 1), e)
        days = (chunk_end - s).days + 1
        yield s.strftime("%Y.%m.%d_00:00:00_TAI"), days
        s = chunk_end + timedelta(days=1)


def _build_query(t_start_tai: str, days: int, cadence_min: int) -> str:
    return f"{HMI_SERIES}[{t_start_tai}/{days}d@{cadence_min}m]{{{HMI_SEGMENT}}}"


def _full_range_query(start: str, end: str, cadence_hours: float) -> str:
    """Full-range query string used only for display."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    days = (e - s).days + 1
    t_start = s.strftime("%Y.%m.%d_00:00:00_TAI")
    return f"{HMI_SERIES}[{t_start}/{days}d@{int(cadence_hours * 60)}m]{{{HMI_SEGMENT}}}"


async def _download_chunk(
    session: aiohttp.ClientSession,
    ds_query: str,
    email: str,
    output_path: Path,
) -> list[str]:
    """Run the 3-step JSOC workflow for one query chunk. Returns saved file paths."""

    # ── Step 1: submit export request ────────────────────────────────────────
    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(JSOC_URL, params={
                "op":       "exp_request",
                "ds":       ds_query,
                "method":   "url",
                "protocol": "fits",
                "notify":   email,
                "format":   "json",
            }) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            break
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                rlog.error(f"  JSOC request failed after {MAX_RETRIES} attempts: {exc}")
                return []
            wait = 10 * attempt
            rlog.warn(f"  Attempt {attempt} failed ({exc}); retrying in {wait}s…")
            await asyncio.sleep(wait)

    status = int(data.get("status", -1))
    req_id = data.get("requestid", "")

    if status < 0 or not req_id:
        rlog.error(f"  JSOC rejected export (status {status}): {data.get('message', '')}")
        rlog.info(f"  Ensure your email is registered at jsoc.stanford.edu")
        return []

    # ── Step 2: poll until ready ──────────────────────────────────────────────
    if status != 0:
        waited = 0
        with rlog.make_progress(f"  Waiting for {req_id}") as progress:
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
                    rlog.error(f"  JSOC export failed (status {poll_status})")
                    return []
                elif waited >= EXPORT_TIMEOUT_SEC:
                    rlog.warn(f"  Timed out after {EXPORT_TIMEOUT_SEC}s.")
                    return []

    # ── Step 3: download files ────────────────────────────────────────────────
    records = data.get("data", [])
    if not records:
        rlog.warn("  Export succeeded but returned 0 files.")
        return []

    raw_dir = data.get("dir", "").rstrip("/")
    base = f"http://jsoc.stanford.edu{raw_dir}" if raw_dir.startswith("/") else raw_dir
    urls = [
        r.get("url") or f"{base}/{r['filename']}"
        for r in records
        if r.get("url") or r.get("filename")
    ]
    rlog.info(f"  [cyan]{len(urls)}[/cyan] files ready — downloading…")

    downloaded = []
    with rlog.make_progress("  Downloading") as progress:
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


async def _run_download(start: str, end: str, email: str,
                        output_path: Path, cadence_hours: float) -> list[str]:
    """Async implementation — called by download() via asyncio.run()."""

    cadence_min = int(cadence_hours * 60)
    chunks = list(_date_chunks(start, end))

    connector = aiohttp.TCPConnector()
    timeout   = aiohttp.ClientTimeout(
        connect=CONNECT_TIMEOUT_SEC,
        sock_read=READ_TIMEOUT_SEC,
    )

    all_files: list[str] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, (t_start, days) in enumerate(chunks, 1):
            ds = _build_query(t_start, days, cadence_min)
            rlog.info(f"Chunk [cyan]{i}/{len(chunks)}[/cyan]: {ds}")
            files = await _download_chunk(session, ds, email, output_path)
            all_files.extend(files)

    return all_files


def download(start: str, end: str, email: str, output_dir: str, cadence_hours: float = 6.0):
    """Download HMI magnetograms and return a list of local file paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks = list(_date_chunks(start, end))
    rlog.kv_table([
        ("Time range", f"{start}  to  {end}"),
        ("Cadence",    f"{cadence_hours} h  ({int(cadence_hours * 60)} min)"),
        ("Query",      _full_range_query(start, end, cadence_hours)),
        ("Chunks",     str(len(chunks))),
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
