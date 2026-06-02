#!/usr/bin/env bash
# Dump the embedded Marmousi velocity-model presets from
# `sweep.datasets` into the npy files that marmousi_forward.yaml /
# marmousi_fwi.yaml expect.
#
# Usage:
#   export MARMOUSI_HOME=$HOME/marmousi    # work directory
#   bash examples/tasks/marmousi_prepare.sh
#
# Generated files (under $MARMOUSI_HOME):
#   vp_true.npy         — Marmousi true vp (shape 281×1361, range 1028–4700 m/s)
#   vp_smooth.npy       — Marmousi smoothed vp (easy FWI starting model)
#   vp_linear.npy       — sweep.datasets's built-in 1-D linear-gradient
#                          (vmin 1500, vmax 3812 — vmax is well below
#                          vp_true's deep section, starves FWI at depth)
#   vp_linear_steep.npy — 1500 m/s water layer for the top 37 rows
#                          (matches vp_true exactly through ~450 m depth)
#                          + linear ramp 1500→4000 m/s below. Deep ceiling
#                          (4000) is just above vp_true's bottom-row mean
#                          (~3926 m/s), leaving headroom for FWI to fill
#                          in the deep high-velocity layers.
#                          This is the default init in the bundled
#                          marmousi_fwi*.yaml examples.

set -euo pipefail

: "${MARMOUSI_HOME:?error: set MARMOUSI_HOME to your work directory first, e.g. export MARMOUSI_HOME=\$HOME/marmousi}"
mkdir -p "$MARMOUSI_HOME"

python3 - "$MARMOUSI_HOME" <<'PY'
import sys
from pathlib import Path
import numpy as np
from sweep.datasets import load_marmousi

out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
for name in ("vp_true", "vp_smooth", "vp_linear"):
    arr = load_marmousi(name)
    out = out_dir / f"{name}.npy"
    np.save(out, arr)
    print(f"wrote {out}  shape={arr.shape}  vmin={arr.min():.1f}  vmax={arr.max():.1f}")

# Custom: steeper 1-D linear gradient (1500 -> 4500 m/s). The default
# vp_linear from sweep.datasets caps at 3812 which is well below
# vp_true's deep section (mean of bottom 50 rows ~3926, peak 4700) and
# starves FWI at depth. The steep version still has zero lateral
# structure (fair cycle-skip stress test) but the deep ceiling matches.
import numpy as _np
true_arr = load_marmousi("vp_true")
nz, nx = true_arr.shape
_N_WATER = 37    # vp_true is constant 1500 m/s for rows 0..36 (depth 0..450 m)
steep = _np.empty((nz, nx), dtype=_np.float32)
steep[:_N_WATER] = 1500.0
ramp = _np.linspace(1500.0, 4000.0, nz - _N_WATER, dtype=_np.float32)
steep[_N_WATER:] = ramp[:, None]
out = out_dir / "vp_linear_steep.npy"
_np.save(out, steep)
print(f"wrote {out}  shape={steep.shape}  vmin={steep.min():.1f}  vmax={steep.max():.1f}")
PY

echo
echo "Done. Next:"
echo "  sweep-tasks run examples/tasks/marmousi_forward.yaml"
echo "  sweep-tasks run examples/tasks/marmousi_fwi.yaml"
echo "(both reference \$MARMOUSI_HOME/vp_*.npy via env-var expansion)"
