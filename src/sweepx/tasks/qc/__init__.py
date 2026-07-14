"""QC products for sweep-tasks.

``fwi``       — QC written during an FWI / LSRTM run (vp snapshots, gradients,
                shot gathers, loss curves, well logs, supershot panels).
``data_prep`` — quick-look PNGs for the ``build-index`` / ``build-plan`` steps.

The FWI-run QC functions are re-exported at the package level, so
``from sweep_tasks.qc import save_vp_png`` keeps working.
"""

from . import data_prep, fwi  # noqa: F401
from .fwi import *  # noqa: F401,F403
