"""A run's stdout / stderr, also written to its run dir.

Where a job's log lands is the launcher's choice (``sbatch --output``, a
studio job dir, a terminal); once the run dir is archived away from there the
two part. :func:`tee_run_logs` keeps a copy in ``<run>/logs/``: rank 0 writes
``stdout.log`` / ``stderr.log``, rank N ``stdout.rank<N>.log`` / ``stderr.rank<N>.log``.
Only what goes through ``sys.stdout`` / ``sys.stderr`` is copied (output a C
extension writes to the file descriptors directly is not). A resumed run
appends, each launch after a ``# ---- <time> host= pid= slurm_job=`` line.
``SWEEP_RUN_LOG=0`` turns it off.
"""

from __future__ import annotations

import io
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _Tee(io.TextIOBase):
    """Writes go to the original stream and, best-effort, to a file."""

    def __init__(self, stream, path: Path, header: str = ""):
        super().__init__()
        self._stream = stream
        self._fh = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        if header:
            self._fh.write(header)

    def write(self, s):  # noqa: D401
        n = self._stream.write(s)
        try:
            self._fh.write(s)
        except Exception:  # noqa: BLE001  (the run never fails for its log copy)
            pass
        return n

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._stream, "errors", "strict")

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close_copy(self):
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def tee_run_logs(logs_dir: Path, rank: int = 0):
    """Copy ``sys.stdout`` / ``sys.stderr`` into ``logs_dir`` while the block runs."""
    if os.environ.get("SWEEP_RUN_LOG", "1") == "0":
        yield
        return
    suffix = "" if rank == 0 else f".rank{rank}"
    # a resumed run appends to the same files: each launch starts with this line
    header = (f"# ---- {datetime.now().astimezone().isoformat(timespec='seconds')} "
              f"host={socket.gethostname()} pid={os.getpid()} "
              f"slurm_job={os.environ.get('SLURM_JOB_ID', '-')}\n")
    old_out, old_err = sys.stdout, sys.stderr
    tees = []
    try:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        out = _Tee(old_out, Path(logs_dir) / f"stdout{suffix}.log", header)
        tees.append(out)
        err = _Tee(old_err, Path(logs_dir) / f"stderr{suffix}.log", header)
        tees.append(err)
        sys.stdout, sys.stderr = out, err
    except Exception as e:  # noqa: BLE001
        sys.stdout, sys.stderr = old_out, old_err
        print(f"[run-log] not copying the logs to {logs_dir}: {e!r}")
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout, sys.stderr = old_out, old_err
        for t in tees:
            t.close_copy()
