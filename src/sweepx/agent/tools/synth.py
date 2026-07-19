"""``make_synthetic_model`` — generate a velocity model .npy from scratch.

Lets a user without any data immediately experiment: a two-layer reflector, a
depth gradient, a layered stack, a background+anomaly, or a faulted model. The
file it writes plugs straight into the forward / wavefield / FWI tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from sweep_agent.tools import register


class MakeSyntheticModelParams(BaseModel):
    kind: Literal["two_layer", "gradient", "smooth", "layered", "anomaly", "fault"] = Field(
        "two_layer",
        description=(
            "Model type: 'two_layer' (flat reflector at mid-depth), 'gradient' / 'smooth' (vp "
            "increases smoothly & linearly with depth — the canonical smooth FWI starting model), "
            "'layered' (several flat layers), 'anomaly' (background + a rectangular high/low-velocity "
            "box), 'fault' (two-layer with a vertical offset)."
        ),
    )
    nz: int = Field(120, ge=8, description="Depth samples (rows).")
    nx: int = Field(160, ge=8, description="Lateral samples (cols).")
    vp_min: float = Field(1800.0, gt=0, description="Low velocity (m/s) — top / background.")
    vp_max: float = Field(3200.0, gt=0, description="High velocity (m/s) — bottom / anomaly.")
    n_layers: int = Field(4, ge=2, description="Number of layers when kind='layered'.")
    out_path: str = Field("/tmp/synthetic_vp.npy", description="Where to save the .npy model (.npy appended if missing).")
    elastic: bool = Field(
        False,
        description=(
            "Generate a full ELASTIC medium, not just vp: writes <stem>_vp.npy, <stem>_vs.npy, "
            "<stem>_rho.npy with physically sensible values (vs ≈ vp/1.73, rho via Gardner) and returns "
            "all three paths. Use when the user wants an elastic model (for Elastic / elastic wavefield)."
        ),
    )


def _two_layer(nz, nx, lo, hi):
    m = np.full((nz, nx), lo, np.float32)
    m[nz // 2:, :] = hi
    return m


def _gradient(nz, nx, lo, hi):
    col = np.linspace(lo, hi, nz, dtype=np.float32)
    return np.repeat(col[:, None], nx, axis=1)


def _layered(nz, nx, lo, hi, n):
    m = np.empty((nz, nx), np.float32)
    vals = np.linspace(lo, hi, n, dtype=np.float32)
    edges = np.linspace(0, nz, n + 1).astype(int)
    for i in range(n):
        m[edges[i]:edges[i + 1], :] = vals[i]
    return m


def _anomaly(nz, nx, bg, anom):
    m = np.full((nz, nx), bg, np.float32)
    z0, z1 = nz // 3, 2 * nz // 3
    x0, x1 = nx // 3, 2 * nx // 3
    m[z0:z1, x0:x1] = anom
    return m


def _fault(nz, nx, lo, hi):
    m = np.full((nz, nx), lo, np.float32)
    half = nx // 2
    m[nz // 2:, :half] = hi
    m[nz // 2 + nz // 6:, half:] = hi  # right block dropped down by nz/6
    return m


@register(
    name="make_synthetic_model",
    description=(
        "Generate a synthetic velocity model and save it to a .npy file — for users who have no data "
        "and want to try forward modelling / FWI. kind ∈ {two_layer, gradient, smooth, layered, anomaly, fault} "
        "('smooth' = a smooth depth-gradient, the usual FWI starting model). "
        "For an ELASTIC medium pass elastic=true to also get vs + rho (returns vp_path/vs_path/rho_path). "
        "Returns the saved path(s), shape and vp range; feed straight into build_forward_spec / run_fwi / "
        "animate_wavefield / plot_velocity_model. Use when the user asks to create/make a model / 生成/造一个模型."
    ),
    params_model=MakeSyntheticModelParams,
)
def make_synthetic_model(args: MakeSyntheticModelParams) -> dict[str, Any]:
    lo, hi = float(args.vp_min), float(args.vp_max)
    if args.kind == "two_layer":
        m = _two_layer(args.nz, args.nx, lo, hi)
    elif args.kind in ("gradient", "smooth"):  # 'smooth' = the FWI smooth-start alias
        m = _gradient(args.nz, args.nx, lo, hi)
    elif args.kind == "layered":
        m = _layered(args.nz, args.nx, lo, hi, args.n_layers)
    elif args.kind == "anomaly":
        m = _anomaly(args.nz, args.nx, lo, hi)
    elif args.kind == "fault":
        m = _fault(args.nz, args.nx, lo, hi)
    else:  # pragma: no cover - guarded by Literal
        return {"error": f"unknown kind '{args.kind}'"}

    out = Path(args.out_path).expanduser()
    if out.suffix != ".npy":            # np.save would append .npy — keep the returned path correct
        out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.elastic:
        # vp from the chosen structure; vs ≈ vp/√3 (Poisson 0.25); rho via Gardner.
        vp = m.astype(np.float32)
        vs = (vp / 1.73).astype(np.float32)
        rho = (310.0 * np.power(vp, 0.25)).astype(np.float32)  # kg/m³, vp in m/s
        stem = str(out.with_suffix(""))
        paths = {}
        for name, arr in (("vp", vp), ("vs", vs), ("rho", rho)):
            p = f"{stem}_{name}.npy"
            np.save(p, arr)
            paths[f"{name}_path"] = p
        return {
            **paths, "kind": args.kind, "shape": list(vp.shape), "elastic": True,
            "summary": (
                f"elastic {args.kind} set saved: vp {vp.min():.0f}-{vp.max():.0f}, "
                f"vs {vs.min():.0f}-{vs.max():.0f} m/s, rho {rho.min():.0f}-{rho.max():.0f} kg/m³. "
                f"Pass vp_path + extra_models={{'vs':..,'rho':..}} to the Elastic tools."
            ),
        }

    np.save(out, m.astype(np.float32))
    return {
        "model_path": str(out),
        "kind": args.kind,
        "shape": list(m.shape),
        "vp_min": float(m.min()),
        "vp_max": float(m.max()),
        "summary": f"{args.kind} model {m.shape}, vp {m.min():.0f}–{m.max():.0f} m/s saved to {out}",
    }
