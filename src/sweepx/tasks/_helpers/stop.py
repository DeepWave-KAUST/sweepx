"""Graceful SIGINT/SIGTERM stopper (checkpoint-then-exit). Verbatim from runner.py."""
from typing import Any


class _GracefulStopper:
    """Defer SIGINT / SIGTERM to the next training-iteration boundary.

    The training loops save a checkpoint at every epoch (see
    ``_save_checkpoint`` callsites in ``_run_fwi`` / ``_run_lsrtm``), so by
    the time the loop next checks :meth:`should_stop` there is already a
    fresh ``checkpoint.pt`` on disk capturing the just-completed epoch.
    The loop then breaks out cleanly, runs the usual final-outputs path,
    and exits with status="success" — but with ``interrupted=True`` /
    ``interrupted_at_epoch=N`` in the summary so callers can tell the
    difference.

    Distributed: every rank installs the handler (torchrun delivers
    SIGINT/SIGTERM to every worker process when the user hits ctrl-c on
    the launcher), but the canonical "should we stop" flag is rank 0's
    value, broadcast at each iter boundary. That keeps the ranks in
    lockstep even if individual workers race on signal delivery.
    """

    def __init__(self) -> None:
        self.requested = False
        self._signum: int | None = None
        self._installed = False
        self._prev: dict[int, Any] = {}

    def install(self, *, label: str = "runner") -> None:
        import signal

        if self._installed:
            return

        def _handler(signum, frame):  # noqa: ARG001
            # First signal: arm the stop flag, let the iter complete.
            # Second signal: restore the previous handler so a follow-up
            # ctrl-c terminates the process the usual way (escape hatch
            # when an iter is genuinely stuck).
            if self.requested:
                self._restore_handlers()
                print(f"\n[{label}] second signal {signum} — restoring default "
                      f"handler; next signal will abort.", flush=True)
                return
            self.requested = True
            self._signum = signum
            print(f"\n[{label}] signal {signum} received; will checkpoint and stop "
                  f"after the current iteration. Hit ctrl-c again to abort "
                  f"immediately.", flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError):
                # ValueError: not on the main thread. OSError: platform
                # doesn't support this signal. Either way, just don't
                # catch this one — the runner will still work, ctrl-c
                # just won't be graceful.
                pass
        self._installed = True

    def _restore_handlers(self) -> None:
        import signal

        for sig, prev in list(self._prev.items()):
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev.clear()

    def uninstall(self) -> None:
        self._restore_handlers()
        self._installed = False

    def should_stop(self, dist_info=None) -> bool:
        if dist_info is None or not getattr(dist_info, "is_distributed", False):
            return self.requested
        from sweep_tasks.runtime import distributed as _dist

        flag = self.requested if dist_info.is_root else None
        flag = _dist.broadcast_object(flag, dist_info, src=0)
        return bool(flag)
