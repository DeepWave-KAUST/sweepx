"""Benchmark velocity models — list and fetch the models `sweep.datasets` ships or
can download (Marmousi, Overthrust, BP, Hess-VTI, …). No ``sweep_tasks`` needed:
this is pure ``sweep`` (the core solver package), so it works on a base install.

The Marmousi and Overthrust 2-D demos are *embedded* in the wheel — they load
instantly with no download, which makes them the easy way to try forward
modelling on a real model instead of a synthetic one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sweep_agent.tools import register


def _require_datasets():
    """Return ``(module, None)`` or ``(None, error_dict)`` — a missing layer is
    reported as data, never raised, so the agent stays up on a base install."""
    try:
        import sweep.datasets as D  # noqa: F401
    except Exception as exc:  # pragma: no cover - only when sweep is absent
        return None, {"error": f"sweep.datasets is not importable: {exc}. "
                               "Install sweep-solver (`pip install sweep-solver`)."}
    import sweep.datasets as D
    return D, None


class ListBenchmarkModelsParams(BaseModel):
    pass


@register(
    name="list_benchmark_models",
    description=(
        "List the benchmark velocity models from `sweep.datasets` — Marmousi, Overthrust, BP, "
        "Hess-VTI, etc. Each row is marked EMBEDDED (bundled, loads instantly) or DOWNLOAD (fetched "
        "on first use). Pure core-solver, no sweep_tasks. Use when the user asks what real / benchmark "
        "models are available, before get_benchmark_model."
    ),
    params_model=ListBenchmarkModelsParams,
)
def list_benchmark_models(args: ListBenchmarkModelsParams) -> dict[str, Any]:
    D, err = _require_datasets()
    if err is not None:
        return err
    models = []
    for name, variant in D.available():
        row = {"name": name, "variant": variant}
        try:
            e = D.info(name, variant)
            row.update({"kind": getattr(e, "kind", "?"),
                        "default": bool(getattr(e, "default", False)),
                        "description": getattr(e, "description", "")})
        except Exception:
            pass
        models.append(row)
    return {
        "count": len(models),
        "models": models,
        "hint": ("EMBEDDED models load instantly; DOWNLOAD models need get_benchmark_model with "
                 "allow_download=true. Fetch one with get_benchmark_model, then run_forward_sweep."),
    }


class GetBenchmarkModelParams(BaseModel):
    name: str = Field(..., description="Benchmark name, e.g. 'marmousi' or 'overthrust'. See list_benchmark_models.")
    variant: str | None = Field(None, description="Variant such as '2d-demo'; omit for the model's default variant.")
    out_dir: str = Field("/tmp", description="Directory to save the model .npy.")
    allow_download: bool = Field(False, description="Allow fetching DOWNLOAD-kind models (may be large / need network). Embedded demos never need this.")


@register(
    name="get_benchmark_model",
    description=(
        "Fetch a benchmark velocity model from `sweep.datasets` (Marmousi, Overthrust, …) and save it "
        "as a .npy ready for run_forward_sweep — pure core-solver, no sweep_tasks. Returns the saved vp "
        "path, its shape, and the model's native grid spacing dh (pass that dh to run_forward_sweep). "
        "Embedded demos (e.g. marmousi/overthrust '2d-demo') load instantly; DOWNLOAD models need "
        "allow_download=true. Use when the user wants to model on a real / named benchmark."
    ),
    params_model=GetBenchmarkModelParams,
)
def get_benchmark_model(args: GetBenchmarkModelParams) -> dict[str, Any]:
    D, err = _require_datasets()
    if err is not None:
        return err
    try:
        entry = D.info(args.name, args.variant)
    except Exception as exc:
        return {"error": f"unknown benchmark '{args.name}'"
                         + (f":{args.variant}" if args.variant else "")
                         + f" ({exc}). See list_benchmark_models."}
    kind = getattr(entry, "kind", "?")
    variant = args.variant or getattr(entry, "variant", None)
    if kind == "download" and not args.allow_download:
        return {"note": f"'{args.name}:{variant}' is a DOWNLOAD model (may be large / need network). "
                        f"Re-call with allow_download=true to fetch it.",
                "license": getattr(entry, "license", None),
                "citation": getattr(entry, "citation", None)}
    try:
        data = D.load(args.name, args.variant)
    except Exception as exc:
        return {"error": f"failed to load '{args.name}:{variant}': {exc}"}

    if isinstance(data, dict):
        vp = data.get("vp")
        dh = data.get("dh")
        presets = data.get("presets")
        citation = data.get("citation")
    else:
        vp, dh, presets, citation = data, None, None, None
    vp = None if vp is None else np.asarray(vp)
    if vp is None or vp.ndim != 2:
        return {"error": f"expected a 2-D vp array from '{args.name}'; got {getattr(vp, 'shape', type(vp).__name__)}."}
    dh_scalar = float(dh[0]) if isinstance(dh, (tuple, list)) and dh else (float(dh) if isinstance(dh, (int, float)) else None)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    vp_path = out_dir / f"{args.name}_{(variant or 'default').replace(':', '_')}_vp.npy"
    np.save(vp_path, vp.astype(np.float32))

    dh_txt = f"{dh_scalar} m" if dh_scalar is not None else "unknown (set dh yourself)"
    return {
        "model_path": str(vp_path),
        "shape": list(vp.shape),
        "dh": dh_scalar,
        "vp_min": float(vp.min()),
        "vp_max": float(vp.max()),
        "kind": kind,
        "presets": presets,
        "citation": citation,
        "summary": (f"{args.name}:{variant} vp {list(vp.shape)}, native dh={dh_txt}, "
                    f"vp {vp.min():.0f}–{vp.max():.0f} m/s → {vp_path}. "
                    f"Run it with run_forward_sweep(vp_path=..., dh={dh_scalar})."),
    }
