"""Background system-resource sampler for RunPod training pods.

Logs CPU%, RAM used / total, and load average as JSON lines at a fixed
interval. Designed to run as a sibling background process to the training
script; it shuts down cleanly on SIGTERM/SIGINT so the onstart trap can
upload the final ``system.log`` snapshot.

Why a Python-side sampler in addition to ``nvidia-smi`` ?
  * nvidia-smi covers GPU only; CPU/RAM blind spots have caused multiple
    silent OOM kills in earlier RunPod attempts.
  * psutil is portable across container images (no nvidia driver assumed).

CLI usage (typically wired into ``onstart.sh.tmpl``):

    python -m gpu.runpod.execution.system_monitor \
        --interval 10 \
        --output /workspace/reinforce/data/output/.../system.log
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import TextIO

import psutil
import typer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    timestamp: str
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    load_avg_1m: float | None


def _take_sample() -> Sample:
    mem = psutil.virtual_memory()
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        load_1m = None
    return Sample(
        timestamp=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_used_mb=round(mem.used / (1024 * 1024), 1),
        ram_total_mb=round(mem.total / (1024 * 1024), 1),
        ram_percent=mem.percent,
        load_avg_1m=load_1m,
    )


def _emit(sample: Sample, fp: TextIO) -> None:
    payload = {
        "timestamp": sample.timestamp,
        "cpu_percent": sample.cpu_percent,
        "ram_used_mb": sample.ram_used_mb,
        "ram_total_mb": sample.ram_total_mb,
        "ram_percent": sample.ram_percent,
        "load_avg_1m": sample.load_avg_1m,
    }
    fp.write(json.dumps(payload) + "\n")
    fp.flush()


_STOP = False


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    global _STOP  # noqa: PLW0603
    _STOP = True
    logger.info("system_monitor received signal %s, shutting down", signum)


def run(interval_seconds: float, output: Path | None) -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if output is None:
        fp: TextIO = sys.stdout
        close_fp = False
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fp = output.open("a", encoding="utf-8")
        close_fp = True

    # Prime psutil.cpu_percent so the first reading is non-zero.
    psutil.cpu_percent(interval=None)
    try:
        while not _STOP:
            _emit(_take_sample(), fp)
            # Sleep in short increments so SIGTERM is honored quickly.
            slept = 0.0
            while not _STOP and slept < interval_seconds:
                time.sleep(min(0.5, interval_seconds - slept))
                slept += 0.5
    finally:
        if close_fp:
            fp.close()


app = typer.Typer(
    add_completion=False,
    help="Background CPU/RAM sampler for RunPod training pods.",
)

_INTERVAL_OPTION = typer.Option(
    10.0,
    "--interval",
    help="Seconds between samples.",
)
_OUTPUT_OPTION = typer.Option(
    None,
    "--output",
    help="Append JSON lines to this path (default: stdout).",
)


@app.command()
def main(
    interval: float = _INTERVAL_OPTION,
    output: Path | None = _OUTPUT_OPTION,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if interval <= 0:
        raise typer.BadParameter("--interval must be > 0")
    run(interval_seconds=interval, output=output)


if __name__ == "__main__":
    app()
