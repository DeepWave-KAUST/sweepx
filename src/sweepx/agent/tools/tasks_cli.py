"""``sweep_tasks`` — run any ``sweep-tasks`` subcommand as a subprocess.

Covers the commands that have no dedicated tool (data preparation:
``build-index``, ``build-plan``, ``extract-coeff``, ``analyze-wavelet``, ...).
Bounded by a timeout; long runs belong in ``submit_task``.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from typing import Any

from pydantic import BaseModel, Field

from sweepx.agent.tools import register

_TAIL_CHARS = 6000


def _tail(text: str) -> str:
    return text if len(text) <= _TAIL_CHARS else "…" + text[-_TAIL_CHARS:]


class SweepTasksParams(BaseModel):
    argv: list[str] = Field(
        ..., min_length=1,
        description="Arguments after `sweep-tasks`, e.g. [\"build-plan\", \"--index\", \"idx.npz\", ...]. "
                    "[\"--help\"] lists the commands; [\"<command>\", \"--help\"] shows a command's options.",
    )
    timeout_s: float = Field(600.0, gt=0, description="Kill the command (and its children) after this many seconds.")


@register(
    name="sweep_tasks",
    description=(
        "Run a `sweep-tasks` CLI command in a subprocess and return its exit code and output tails. "
        "Use it for commands without a dedicated tool: init, build-index, build-plan, extract-coeff, "
        "extract-coeff-segy, analyze-wavelet, estimate-wavelet, convert-farfield, plot-wavelet-steps, "
        "filter-image, migrate-water, tasks list/status/logs. Check a command's options with --help first. "
        "Runs on this machine and blocks until done; submit long runs with submit_task instead."
    ),
    params_model=SweepTasksParams,
)
def sweep_tasks(args: SweepTasksParams) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "sweepx.tasks", *args.argv]
    # Own session so a timeout also kills torchrun workers; stdin closed so a
    # child never reads the MCP stream.
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    result: dict[str, Any] = {"command": "sweep-tasks " + shlex.join(args.argv)}
    try:
        out, err = proc.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
        result["error"] = f"timed out after {args.timeout_s:g} s and was killed; submit long runs with submit_task"
    result.update(returncode=proc.returncode, stdout=_tail(out), stderr=_tail(err))
    if "error" not in result and proc.returncode != 0:
        result["error"] = f"sweep-tasks exited with code {proc.returncode}"
    return result
