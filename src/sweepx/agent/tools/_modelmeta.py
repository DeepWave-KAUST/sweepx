"""Grid spacing recorded next to a model file, and the check that uses it.

A ``.npy`` velocity model carries no grid spacing, so every modelling tool asks
for ``dh`` — and a model that just loaded Marmousi (dh 12.5 m) will happily pass
``dh=20`` and get a run on a stretched grid that reports success. The tools that
write models (``get_benchmark_model``, ``make_synthetic_model``) therefore save
``<stem>.meta.json`` beside ``<stem>.npy``; :func:`fill_dh` reads it to supply a
missing ``dh`` and to reject one that disagrees.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Argument names that hold a model file whose grid the run uses.
MODEL_PATH_ARGS = ("vp_path", "init_model_path", "synthetic_true_vp_path", "model_path")

DH_OMIT_NOTE = (
    " Omit it when the model came from get_benchmark_model or make_synthetic_model: "
    "the dh recorded with the model file is used, and a different value is rejected."
)


def meta_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(p.stem + ".meta.json") if p.suffix == ".npy" else Path(str(p) + ".meta.json")


def write_model_meta(model_path: str | Path, *, dh: float, shape, source: str) -> Path:
    out = meta_path(model_path)
    out.write_text(json.dumps({"dh": float(dh), "shape": [int(n) for n in shape], "source": source}, indent=1))
    return out


def read_model_meta(model_path: str | Path) -> dict[str, Any] | None:
    """The recorded metadata, or None when there is none or it is stale (the
    model's shape no longer matches, i.e. the .npy was overwritten)."""
    mp = meta_path(model_path)
    if not mp.is_file() or not Path(model_path).is_file():
        return None
    try:
        meta = json.loads(mp.read_text())
        import numpy as np

        shape = list(np.load(model_path, mmap_mode="r").shape)
    except Exception:
        return None
    if meta.get("shape") != shape or not isinstance(meta.get("dh"), (int, float)):
        return None
    return meta


def fill_dh(arg_names, raw_args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Supply or check ``dh`` from the metadata of the model files in ``raw_args``.

    Returns ``(args, error)``. ``args`` has ``dh`` filled in when it was omitted
    and the models record one. ``error`` is set when two models record different
    spacings, or the given ``dh`` disagrees with the recorded one.
    """
    if "dh" not in arg_names:
        return raw_args, None
    recorded: dict[str, float] = {}
    for key in MODEL_PATH_ARGS:
        path = raw_args.get(key)
        if key in arg_names and isinstance(path, str) and path:
            meta = read_model_meta(path)
            if meta is not None:
                recorded[path] = float(meta["dh"])
    if not recorded:
        return raw_args, None
    values = sorted(set(recorded.values()))
    if len(values) > 1:
        listing = ", ".join(f"{p}: {v:g} m" for p, v in recorded.items())
        return raw_args, f"the model files record different grid spacings ({listing}); they cannot share one grid."
    ref = values[0]
    given = raw_args.get("dh")
    if given is None:
        return {**raw_args, "dh": ref}, None
    try:
        given_f = float(given)
    except (TypeError, ValueError):
        return raw_args, None  # let the params model report the bad type
    if not math.isclose(given_f, ref, rel_tol=1e-6):
        path = next(iter(recorded))
        return raw_args, (
            f"dh={given_f:g} m disagrees with the {ref:g} m recorded for {path} "
            f"({meta_path(path).name}). Omit dh, or pass dh={ref:g}."
        )
    return raw_args, None
