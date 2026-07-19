"""``check_parameters`` — a CFL stability + grid-dispersion sanity check.

The single most common way a forward / FWI run goes wrong is a bad dt/dh/fm
combination: too-large dt blows up (CFL), too-coarse dh disperses the wavefield.
This tool turns those rules into a red/green report with concrete fixes, and
recommends safe values — so the LLM (and the user) can sanity-check before
spending minutes on a run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sweep_agent.tools import register

# Energy in a Ricker wavelet extends to ~2.5×fm; sample the SHORTEST wavelength.
_FMAX_FACTOR = 2.5
# Points-per-wavelength thresholds (at fmax) for a high-order FD scheme.
_PPW_GOOD = 5.0
_PPW_MARGINAL = 3.5
# Courant number thresholds (vmax·dt·√ndim / dh).
_CFL_GOOD = 0.5
_CFL_MAX = 0.9


class CheckParametersParams(BaseModel):
    dh: float = Field(..., gt=0, description="Grid spacing (m).")
    dt: float = Field(..., gt=0, description="Time step (s).")
    fm: float = Field(..., gt=0, description="Ricker peak frequency (Hz).")
    model_path: str | None = Field(None, description="Velocity model .npy — used to read v_min / v_max and the dimensionality.")
    vp_min: float | None = Field(None, gt=0, description="Min velocity (m/s); needed if model_path is not given.")
    vp_max: float | None = Field(None, gt=0, description="Max velocity (m/s); needed if model_path is not given.")
    ndim: int = Field(2, ge=2, le=3, description="2 or 3 spatial dimensions (inferred from model_path if given).")


@register(
    name="check_parameters",
    description=(
        "Sanity-check a forward/FWI parameter set for STABILITY (CFL) and grid DISPERSION "
        "(points-per-wavelength), and recommend safe values. Give dh, dt, fm plus either a model_path "
        "or vp_min/vp_max. Returns the Courant number, points-per-wavelength, pass/fail flags, plain "
        "warnings, and recommended dt/dh. Use when the user asks if their parameters are OK / stable / "
        "会不会发散 / 频散, or before launching a costly run."
    ),
    params_model=CheckParametersParams,
)
def check_parameters(args: CheckParametersParams) -> dict[str, Any]:
    vmin, vmax, ndim = args.vp_min, args.vp_max, args.ndim
    if args.model_path:
        p = Path(args.model_path).expanduser()
        if not p.is_file():
            return {"error": f"model file not found: {args.model_path}"}
        arr = np.load(p, mmap_mode="r")
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        ndim = 3 if arr.ndim == 3 else 2
    if vmin is None or vmax is None:
        return {"error": "need vp_min and vp_max (or a model_path to read them from)."}

    sq = math.sqrt(ndim)
    courant = vmax * args.dt * sq / args.dh
    fmax = _FMAX_FACTOR * args.fm
    ppw_peak = vmin / (args.fm * args.dh)      # at the dominant frequency
    ppw_max = vmin / (fmax * args.dh)          # at the (stricter) shortest wavelength

    cfl_ok = courant <= _CFL_MAX
    disp_ok = ppw_max >= _PPW_MARGINAL

    warnings: list[str] = []
    if courant > 1.0:
        warnings.append(f"UNSTABLE: Courant {courant:.2f} > 1 — the run will blow up. Reduce dt.")
    elif courant > _CFL_MAX:
        warnings.append(f"marginal stability: Courant {courant:.2f} (keep ≤ {_CFL_GOOD}). Reduce dt.")
    if ppw_max < _PPW_MARGINAL:
        warnings.append(
            f"strong grid dispersion: only {ppw_max:.1f} pts/shortest-wavelength "
            f"(want ≥ {_PPW_GOOD}). Reduce dh or lower fm."
        )
    elif ppw_max < _PPW_GOOD:
        warnings.append(f"mild dispersion: {ppw_max:.1f} pts/shortest-wavelength (ideal ≥ {_PPW_GOOD}).")

    rec_dt = _CFL_GOOD * args.dh / (vmax * sq)              # dt for a comfortable Courant
    rec_dh = vmin / (_PPW_GOOD * fmax)                       # dh for ≥5 pts at fmax
    return {
        "courant_number": round(courant, 3),
        "cfl_ok": cfl_ok,
        "points_per_wavelength_peak": round(ppw_peak, 2),
        "points_per_wavelength_fmax": round(ppw_max, 2),
        "dispersion_ok": disp_ok,
        "vp_min": vmin, "vp_max": vmax, "ndim": ndim,
        "ok": cfl_ok and disp_ok,
        "warnings": warnings or ["parameters look good (stable + well-sampled)."],
        "recommended_dt_max": round(rec_dt, 6),
        "recommended_dh_max": round(rec_dh, 3),
    }
