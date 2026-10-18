"""Runtime orchestration helpers for sweep-tasks.

Absorbed from the former standalone ``sweep-runner`` package. Dependency-light
(torch/numpy only) utilities used by :mod:`sweepx.tasks.runner`:

- :mod:`~sweepx.tasks.runtime.scheduler`   -- LR schedule builders
- :mod:`~sweepx.tasks.runtime.checkpoint`  -- atomic optimiser/model checkpoint I/O
- :mod:`~sweepx.tasks.runtime.distributed` -- torchrun / DDP initialisation
- :mod:`~sweepx.tasks.runtime.config`      -- nested-config override helpers

Kept as a cohesive subpackage (each module standalone, no cross-imports) so it
stays easy to reuse or re-extract.
"""

from . import config, scheduler, checkpoint, distributed  # noqa: F401
