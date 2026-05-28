"""``build_fwi_spec`` — flat shortcut for the common FWI (inversion) case.

Analogous to build_forward_spec but for full-waveform inversion: give an initial
velocity model, an observed-data source, and the optimization knobs. The full
long tail (multiscale stages, NN reparam, data/model plans, SEG-Y index, source
encoding...) stays available through the generic build_spec + describe_task_schema.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sweep_agent.tools import register

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text).strip("-_") or "fwi"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class BuildFwiParams(BaseModel):
    """Flat parameter surface for a 2-D / 3-D FWI run."""

    # --- models -----------------------------------------------------------
    init_model_path: str = Field(..., description="Path to the INITIAL velocity model (.npy) the inversion starts from.")
    equation: str = Field("Acoustic", description="Wave equation class name; call list_equations for the full set.")
    extra_init_models: dict[str, str] | None = Field(
        None,
        description="For multi-parameter equations, the OTHER initial models (e.g. {'epsilon':'/p/e.npy','delta':'/p/d.npy'}). Acoustic needs none.",
    )

    # --- observed data (exactly one) -------------------------------------
    synthetic_true_vp_path: str | None = Field(
        None,
        description="RECOMMENDED for tests/demos: path to a TRUE vp .npy. sweep forward-models it to make obs, then inverts init toward it (shapes always match).",
    )
    obs_npy_path: str | None = Field(
        None,
        description="Pre-saved observed data .npy of shape (nshots, nreceivers, nt). NOTE: this is NOT the same layout as a forward record.npy (nshots,nt,nrec,1).",
    )
    obs_segy_path: str | None = Field(None, description="Single observed-data SEG-Y file.")

    # --- grid / time / wavelet -------------------------------------------
    dh: float = Field(..., gt=0, description="Grid spacing (m).")
    dt: float = Field(..., gt=0, description="Time step (s).")
    nt: int = Field(..., ge=1, description="Number of time samples.")
    fm: float = Field(8.0, gt=0, description="Ricker centre frequency (Hz).")
    wavelet_delay: float | None = Field(None, ge=0, description="Ricker delay (s); default 1/fm.")

    # --- geometry (line) --------------------------------------------------
    source_step: int = Field(50, ge=1)
    source_depth: int = Field(1, ge=0)
    receiver_step: int = Field(1, ge=1)
    receiver_depth: int = Field(18, ge=0)

    # --- inversion --------------------------------------------------------
    optimizer: Literal["adam", "sgd", "lbfgs"] = Field("adam", description="Optimizer.")
    lr: float = Field(10.0, gt=0, description="Learning rate. For vp with Adam O(10) m/s/step is typical; LBFGS uses ~1.")
    loss: Literal["mse", "l1", "huber", "trace_cosine"] = Field("mse", description="Misfit functional.")
    epochs: int = Field(30, ge=1, description="Number of inversion iterations.")
    batchsize: int = Field(1, ge=1, description="Shots per batch.")
    vp_min: float | None = Field(None, description="Lower clamp on vp (m/s) after each step, e.g. 1500.")
    vp_max: float | None = Field(None, description="Upper clamp on vp (m/s), e.g. 4700.")

    # --- physics / backend ------------------------------------------------
    spatial_order: int = Field(8)
    abcn: int = Field(20, ge=0)
    free_surface: bool = Field(False)
    backend_impl: Literal["eager", "c"] = Field("eager")
    use_compile: bool = Field(False, description="torch.compile the eager step (needs sweep.torch + inductor). Default off.")

    # --- io ---------------------------------------------------------------
    output_dir: str = Field("./sweep_runs")
    task_id: str | None = Field(None)
    device: str = Field("auto")
    seed: int = Field(0)
    show_every: int = Field(10)
    save_yaml: bool = Field(True)
    extra: dict[str, Any] | None = Field(None, description="Deep-merged overrides for FWISpec fields not exposed here (stages, data_plan, reparam, scheduler, ...).")


def _resolve_init_models(p: BuildFwiParams) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate equation + assemble init_model / init_models block."""
    extra = p.extra_init_models or {}
    try:
        from sweep_agent.tools.introspect import all_equations
        eqs = all_equations()
    except Exception:
        eqs = None

    if eqs is None:
        if extra:
            models = [{"name": "vp", "path": p.init_model_path}]
            models += [{"name": k, "path": v} for k, v in extra.items()]
            return {"init_models": models}, None
        return {"init_model": {"name": "vp", "path": p.init_model_path}}, None

    if p.equation not in eqs:
        import difflib
        ql = p.equation.lower()
        substr = [e for e in eqs if ql in e.lower()]
        fuzzy = difflib.get_close_matches(p.equation, list(eqs), n=5, cutoff=0.4)
        sug = list(dict.fromkeys(substr + fuzzy))[:6]
        return {}, {"error": f"unknown equation '{p.equation}'. closest: {sug}. call list_equations."}

    required = eqs[p.equation]
    provided = {"vp", *extra.keys()}
    missing = [m for m in required if m not in provided]
    unneeded = [m for m in extra if m not in required]
    if missing:
        return {}, {"error": f"equation '{p.equation}' requires init models {required}; missing {missing}. provide via extra_init_models."}
    if unneeded:
        return {}, {"error": f"equation '{p.equation}' only needs {required}; got unexpected extra_init_models {unneeded}."}

    if len(required) == 1:
        return {"init_model": {"name": "vp", "path": p.init_model_path}}, None
    sources = {"vp": p.init_model_path, **extra}
    return {"init_models": [{"name": m, "path": sources[m]} for m in required]}, None


def _resolve_obs(p: BuildFwiParams) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    chosen = [
        name for name, val in (
            ("synthetic_true_vp_path", p.synthetic_true_vp_path),
            ("obs_npy_path", p.obs_npy_path),
            ("obs_segy_path", p.obs_segy_path),
        ) if val
    ]
    if len(chosen) != 1:
        return {}, "", {
            "error": "provide exactly one obs source: synthetic_true_vp_path (demos) / "
            f"obs_npy_path / obs_segy_path. got {chosen}."
        }
    if p.synthetic_true_vp_path:
        return {"synthetic_from": {"name": "vp", "path": p.synthetic_true_vp_path}}, chosen[0], None
    if p.obs_npy_path:
        return {"npy_path": p.obs_npy_path}, chosen[0], None
    return {"segy": {"path": p.obs_segy_path}}, chosen[0], None


@register(
    name="build_fwi_spec",
    description=(
        "Construct a validated FWI (full-waveform inversion) spec — the flat shortcut for the "
        "common inversion case (analogous to build_forward_spec). Give init_model_path, exactly one "
        "obs source (synthetic_true_vp_path for demos/tests, or obs_npy_path / obs_segy_path for real "
        "data), and optimizer/lr/loss/epochs (+ optional vp_min/vp_max bounds). Validates the equation "
        "and required models. For multiscale stages / NN reparam / data plans / other advanced knobs, "
        "use describe_task_schema('fwi') + build_spec instead. Call inspect_file on inputs first."
    ),
    params_model=BuildFwiParams,
)
def build_fwi_spec(args: BuildFwiParams) -> dict[str, Any]:
    try:
        from sweep_tasks.schemas import FWISpec
        from sweep_tasks import dump_task
    except ImportError as exc:
        return {"error": f"sweep_tasks is not importable: {exc}"}

    init_block, err = _resolve_init_models(args)
    if err is not None:
        return err
    obs, obs_kind, err = _resolve_obs(args)
    if err is not None:
        return err

    delay = args.wavelet_delay if args.wavelet_delay is not None else 1.0 / args.fm
    opt = {
        "adam": {"kind": "adam", "lr": args.lr},
        "sgd": {"kind": "sgd", "lr": args.lr},
        "lbfgs": {"kind": "lbfgs", "lr": args.lr},
    }[args.optimizer]

    spec: dict[str, Any] = {
        "task_type": "fwi",
        "output_dir": args.output_dir,
        "seed": args.seed,
        "device": args.device,
        "grid": {"dh": args.dh},
        "time": {"dt": args.dt, "nt": args.nt},
        "wavelet": {"kind": "ricker", "fm": args.fm, "delay": delay},
        "geometry": {
            "kind": "line",
            "sources": {"step": args.source_step, "depth": args.source_depth},
            "receivers": {"step": args.receiver_step, "depth": args.receiver_depth},
        },
        "physics": {
            "equation": args.equation,
            "spatial_order": args.spatial_order,
            "abcn": args.abcn,
            "free_surface": args.free_surface,
        },
        "backend": {"impl": args.backend_impl},
        "obs": obs,
        "optimizer": opt,
        "loss": {"kind": args.loss},
        "epochs": args.epochs,
        "batchsize": args.batchsize,
        "show_every": args.show_every,
        **init_block,
    }
    if args.backend_impl == "eager":
        spec["backend"]["eager_options"] = {"use_compile": args.use_compile}
    if args.vp_min is not None or args.vp_max is not None:
        b: dict[str, Any] = {"enabled": True}
        if args.vp_min is not None:
            b["min"] = args.vp_min
        if args.vp_max is not None:
            b["max"] = args.vp_max
        spec["model_bounds"] = {"vp": b}
    if args.task_id is not None:
        spec["task_id"] = args.task_id
    if args.extra:
        spec = _deep_merge(spec, args.extra)

    try:
        spec_obj = FWISpec.model_validate(spec)
    except Exception as exc:
        return {"error": f"FWISpec validation failed: {exc}", "spec_attempted": spec}

    eff_id = args.task_id or _slug(Path(args.init_model_path).stem) + "-fwi"
    yaml_path: str | None = None
    if args.save_yaml:
        root = Path(args.output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        yaml_path = str((root / f"{_slug(eff_id)}.yaml").resolve())
        dump_task(spec_obj, yaml_path)

    return {
        "spec": spec_obj.model_dump(mode="json"),
        "yaml_path": yaml_path,
        "task_id_hint": eff_id,
        "summary": (
            f"FWISpec ready: {args.equation} FWI, optimizer={args.optimizer} lr={args.lr}, "
            f"loss={args.loss}, {args.epochs} epochs, obs via {obs_kind}. "
            f"Run with run_task(yaml_path=...)."
        ),
    }
