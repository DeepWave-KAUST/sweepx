"""Shared helpers for sweep-agent visualization demos.

These demos drive the *same tools the LLM uses* (build_spec / run_task) and then
read the resulting artifacts to plot. Keeping the plotting here lets each demo
script stay short and focused on the scientific story.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sweep_agent.tools import registry


def percentile_clip(data: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Signed percentile color limits on the raw array (no abs). Matches the
    project convention for wavefield / gradient displays."""
    vmin, vmax = np.percentile(data, [lo, hi])
    return float(vmin), float(vmax)


def load_snapshots(task_dir: Path | str, abcn: int, *, free_surface: bool = False,
                   shot: int = 0, field: int = 0) -> np.ndarray:
    """Load output/snapshots.npy and crop the PML → (n_snap, nz, nx).

    snapshots.npy layout is (n_snap, n_field, nshots, nchannel, nz_pad, nx_pad);
    we take one field + one shot, then strip the absorbing border. With a free
    surface the top row is physical (not padded), so it is not cropped.
    """
    snap = np.load(Path(task_dir) / "output" / "snapshots.npy")
    arr = snap[:, field, shot, 0]  # (n_snap, nz_pad, nx_pad)
    nzp, nxp = arr.shape[-2:]
    top = 0 if free_surface else abcn
    return arr[:, top:nzp - abcn, abcn:nxp - abcn]


def run_wavefield(
    equation: str,
    vp_path: str,
    shape: tuple[int, int],
    snapshot_times: list[int],
    *,
    extra_models: dict[str, str] | None = None,
    output_dir: Path | str,
    task_id: str,
    dh: float = 10.0,
    dt: float = 1.0e-3,
    nt: int = 400,
    fm: float = 12.0,
    abcn: int = 20,
    spatial_order: int = 4,
    source_xz: tuple[int, int] | None = None,
    free_surface: bool = False,
    topography: str | None = None,
) -> Path:
    """Build (generic build_spec) + run a wavefield task with an explicit
    center source. Auto-selects source_type/receiver_type/pml_type for the
    equation via the introspection helpers. Returns the task_dir."""
    from sweep_agent.tools.introspect import equation_default_fields, equation_default_pml

    nz, nx = shape[-2], shape[-1]
    sx, sz = source_xz or (nx // 2, nz // 2)
    sf, rf = equation_default_fields(equation)
    pml = equation_default_pml(equation)

    models: list[dict[str, Any]] = [{"name": "vp", "path": vp_path}]
    for k, v in (extra_models or {}).items():
        models.append({"name": k, "path": v})

    physics: dict[str, Any] = {
        "equation": equation, "spatial_order": spatial_order, "abcn": abcn,
        "free_surface": free_surface,
    }
    if sf:
        physics["source_type"] = sf
    if rf:
        physics["receiver_type"] = rf
    if pml:
        physics["pml_type"] = pml
    if topography is not None:
        physics["topography"] = topography

    spec = {
        "grid": {"dh": dh},
        "time": {"dt": dt, "nt": nt},
        "wavelet": {"kind": "ricker", "fm": fm, "delay": 1.0 / fm},
        "geometry": {
            "kind": "explicit",
            "sources": [[sx, sz]],
            "receivers": [[x, 2] for x in range(2, nx - 2, 3)],
        },
        "physics": physics,
        "backend": {"impl": "eager", "eager_options": {"use_compile": False}},
        "models": models,
        "snapshot_times": list(snapshot_times),
        "plot": False,
        "device": "cpu",
    }
    b = registry.get("build_spec").invoke({
        "task_type": "wavefield", "spec": spec,
        "output_dir": str(output_dir), "task_id": task_id,
    })
    if not b.ok or "error" in b.value:
        raise RuntimeError(b.value.get("error") if b.ok else b.error)
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 400})
    if r.value.get("state") != "success":
        raise RuntimeError(f"run failed: {r.value.get('error')}")
    return Path(r.value["task_dir"])
