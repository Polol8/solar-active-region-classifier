"""
Terminal output utilities — all user-facing messages go through this module.

Every pipeline stage calls stage(), then uses info/success/warn and the
progress-bar factory. Python's logging module is kept at WARNING+ so that
third-party library chatter (sunpy, astropy, etc.) stays off the terminal.
"""

import logging
import time

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Silence noisy INFO-level output from generic network/utility libraries.
# astropy and sunpy are intentionally excluded: calling getLogger("astropy")
# before astropy itself is imported pre-creates a plain Logger and prevents
# astropy from registering its custom AstropyLogger class (which would cause
# an AttributeError on _set_defaults at import time).
for _lib in ("drms", "zeep", "urllib3", "requests", "parfive", "paramiko",
             "matplotlib"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

# legacy_windows=False forces ANSI rendering on Windows 11 (Windows Terminal
# supports ANSI natively; the legacy Win32 API path only handles cp1252 and
# would crash on characters like →, ✓, ⚠ that we use for stage output).
console = Console(highlight=False, legacy_windows=False)

_TOTAL_STAGES = 6


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def banner():
    """Application header — call once at the start of main.py."""
    console.print()
    console.print(
        Panel.fit(
            "[bold white]Solar Active Region Classifier[/bold white]\n"
            "[dim]SDO/HMI  ·  YOLOv11m  ·  Mount Wilson classification[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def stage(n: int, title: str, total: int = _TOTAL_STAGES):
    """Horizontal rule marking the start of a pipeline stage."""
    console.print()
    console.rule(f"[bold cyan][{n}/{total}] {title}[/bold cyan]", style="dim cyan")


# ---------------------------------------------------------------------------
# Single-line messages
# ---------------------------------------------------------------------------

def info(msg: str):
    console.print(f"  {msg}")


def success(msg: str):
    console.print(f"  [bold green]✓[/bold green]  {msg}")


def warn(msg: str):
    console.print(f"  [bold yellow]⚠[/bold yellow]  {msg}")


def error(msg: str):
    console.print(f"  [bold red]✗[/bold red]  {msg}")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def make_progress(description: str) -> Progress:
    """Return a Rich Progress bar configured for pipeline use (use as context manager)."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"  [cyan]{description}[/cyan]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def kv_table(rows: list[tuple[str, str]]):
    """Print a compact key / value list (no borders, left-aligned keys)."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim",   min_width=12, no_wrap=True)
    table.add_column(style="white", no_wrap=False)
    for key, val in rows:
        table.add_row(key, val)
    console.print(table)


def class_table(counts: dict[int, int], names: list[str]):
    """Print a class-distribution table after label generation."""
    total = sum(counts.values())
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Class",  style="cyan",  min_width=20)
    table.add_column("Boxes",  justify="right")
    table.add_column("%",      justify="right", style="dim")
    for idx, name in enumerate(names):
        n = counts.get(idx, 0)
        pct = f"{100 * n / total:.1f}" if total else "—"
        table.add_row(name, str(n), pct)
    console.print()
    console.print("  [bold]Class distribution[/bold]")
    console.print(table)


def metrics_table(rows: list[tuple[str, str]]):
    """Print an evaluation metrics table."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan", min_width=16)
    table.add_column("Value",  justify="right")
    for key, val in rows:
        table.add_row(key, val)
    console.print()
    console.print("  [bold]Evaluation results[/bold]")
    console.print(table)


# ---------------------------------------------------------------------------
# Pipeline-level progress (overall across all stages)
# ---------------------------------------------------------------------------

class PipelineTracker:
    """Tracks stage timing and renders a live overview panel.

    Usage in main.py:
        tracker = PipelineTracker(["Download", "Preprocess", ...])
        tracker.start(0)            # mark stage 0 as running
        ...run stage...
        tracker.done(0)             # mark done, print updated panel
        tracker.start(1)
        ...
    """

    _STATUS_ICON = {
        "done":    "[bold green]✓[/bold green]",
        "running": "[bold yellow]▶[/bold yellow]",
        "skipped": "[dim]⊘[/dim]",
        "pending": "[dim]·[/dim]",
    }

    def __init__(self, stage_names: list[str]):
        self._names   = stage_names
        self._total   = len(stage_names)
        self._status  = ["pending"] * self._total
        self._start_t = [0.0]       * self._total
        self._elapsed = [0.0]       * self._total

    def skip(self, idx: int):
        self._status[idx] = "skipped"

    def start(self, idx: int):
        self._status[idx]  = "running"
        self._start_t[idx] = time.monotonic()

    def done(self, idx: int):
        self._elapsed[idx] = time.monotonic() - self._start_t[idx]
        self._status[idx]  = "done"
        self._render()

    def _render(self):
        n_done = sum(1 for s in self._status if s in ("done", "skipped"))

        # Stage rows
        rows = Table(show_header=False, box=None, padding=(0, 1))
        rows.add_column(min_width=6,  style="dim")
        rows.add_column(min_width=2)
        rows.add_column(min_width=20)
        rows.add_column(min_width=8, style="dim", justify="right")

        for i, (name, st) in enumerate(zip(self._names, self._status)):
            icon = self._STATUS_ICON[st]
            nm   = f"[bold]{name}[/bold]" if st == "running" else (
                   f"[white]{name}[/white]"  if st == "done"    else
                   f"[dim]{name}[/dim]")
            elapsed = _fmt_elapsed(self._elapsed[i]) if st == "done" else ""
            rows.add_row(f"[{i+1}/{self._total}]", icon, nm, elapsed)

        # Overall bar
        filled = round(38 * n_done / self._total) if self._total else 0
        bar    = Text()
        bar.append("  ")
        bar.append("█" * filled,             style="bold cyan")
        bar.append("░" * (38 - filled),      style="dim")
        bar.append(f"  {n_done}/{self._total}",  style="bold")
        pct = round(100 * n_done / self._total) if self._total else 0
        bar.append(f"  {pct}%",              style="dim")

        console.print()
        console.print(Panel(Group(rows, bar), border_style="dim cyan",
                            padding=(0, 1), expand=False))


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"
