"""Entrypoint for ``python -m sweep_tasks``.

Forwards to :func:`sweep_tasks.cli.main` so ``torchrun -m sweep_tasks ...``
works the same way as the installed ``sweep-tasks`` console script.
"""

from __future__ import annotations

import sys

from sweep_tasks.cli import main


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main() or 0)
