"""Read-only views of what the runner does, for tools that edit task YAML.

sweep-studio builds its forms, previews and pre-flight checks from these. Each
one calls the runner's own helper, or reads the same sweep attribute the
propagator reads, instead of re-deriving it, so a preview cannot drift from
what ``sweep-tasks run`` does with the same spec. Nothing here builds or runs a
solver.
"""

from __future__ import annotations

import dataclasses
import difflib
import inspect
from pathlib import Path
from typing import Any

import numpy as np

import sweep
import sweep.equations as eq_mod

from sweep_tasks._helpers.geometry import _build_geometry
from sweep_tasks._helpers.model import (
    _infer_shape,
    _model_array,
    _model_names_for_equation,
)
from sweep_tasks._helpers.plan_apply import _normalize_fwi_init_models
from sweep_tasks._helpers.wavelet_build import _build_wavelet
from sweep_tasks.schemas import (
    ForwardSpec,
    FWISpec,
    LSRTMSpec,
    ModelRef,
    PhysicsSpec,
    RTMSpec,
    WavefieldSpec,
)
from sweep_tasks.yaml_io import resolve_paths

__all__ = [
    "CFL_RAISE",
    "CFL_WARN",
    "build_geometry_arrays",
    "build_wavelet",
    "cfl_number",
    "describe_equation",
    "equation_catalog",
    "grid_shape",
    "load_model_array",
    "time_axis",
]


# ----- CFL ---------------------------------------------------------------

# The runner's Courant gate (``_helpers.solver_build._cfl_check``): above
# CFL_WARN it prints a warning, above CFL_RAISE it refuses to start, because
# the forward propagator then produces NaNs within the first steps.
CFL_WARN = 0.85
CFL_RAISE = 1.0


def cfl_number(vmax: float, dh: float, dt: float) -> float:
    """``vmax * dt / dh``, evaluated exactly as the runner's CFL check does."""
    return float(vmax) * float(dt) / float(dh)


# ----- equations ---------------------------------------------------------

# How the runner constructs every equation (``_build_solver``): only these
# three arguments, with ``spatial_order`` from the spec. The catalogue builds
# its probe instance the same way, at the schema's default order.
_PROBE_KWARGS = {
    "spatial_order": PhysicsSpec.model_fields["spatial_order"].default,
    "device": "cpu",
    "backend": "torch",
}

# Attributes an equation's eager step reads when it honours an irregular
# surface: the image-method surface rows and the APM air mask the propagator
# attaches from ``topography=``. sweep has no declared capability flag for
# this, and the propagator accepts ``topography=`` for every equation, so an
# equation whose step never reads either one ignores ``physics.topography``.
_TOPOGRAPHY_ATTRS = ("_topo_rows_runtime", "_apm_air_mask_runtime")


def equation_catalog() -> dict[str, dict]:
    """Every name sweep exports as an equation, mapped to its description.

    Keys are what ``physics.equation`` takes (aliases such as
    ``AcousticVTILiang`` included, flagged in ``note``), plus the facades
    (``is_facade``) that sweep exports but the runner does not accept. See
    :func:`describe_equation` for the row format. JSON-serialisable.
    """
    concrete = _concrete_equations()
    facades = _facade_equations()
    ctx = _Context(concrete, facades)
    rows = {name: _describe_concrete(name, cls, ctx) for name, cls in concrete.items()}
    rows.update({name: _describe_facade(name, obj, ctx) for name, obj in facades.items()})
    return dict(sorted(rows.items()))


def describe_equation(name: str) -> dict:
    """Describe one equation by the name ``physics.equation`` would carry.

    Returns ``{name, ndim, summary, is_facade, models, fields,
    default_source_fields, default_receiver_fields, pml: {default,
    supported}, free_surface: {supported, per_edge}, impl: {eager, c,
    c_available}, boundary_saving_c, topography, note}``. ``models`` is in
    the order the runner loads them (``MODEL_SPECS``). A part that cannot be
    read for this equation is left empty and the reason is put in ``note``
    instead of raising. Unknown names raise ``KeyError`` with close matches.
    """
    concrete = _concrete_equations()
    facades = _facade_equations()
    if name in concrete:
        return _describe_concrete(name, concrete[name], _Context(concrete, facades))
    if name in facades:
        return _describe_facade(name, facades[name], _Context(concrete, facades))
    known = sorted(set(concrete) | set(facades))
    close = difflib.get_close_matches(name, known, n=5, cutoff=0.6)
    hint = f"close matches: {close}" if close else f"known: {known}"
    raise KeyError(f"Unknown equation {name!r}; {hint}")


def _concrete_equations() -> dict[str, type]:
    # The runner resolves ``physics.equation`` through this same private sweep
    # function (``_helpers.model._get_equation_class``).
    return dict(eq_mod._equation_classes())


def _facade_equations() -> dict[str, type]:
    """Exported classes that dispatch to an equation without being one.

    ``DAS`` (an ``nn.Module`` wiring a DAS equation to a propagator) and
    ``AcousticAniso`` (a factory whose ``__new__`` returns another class) are
    what sweep's own docs point at, but the runner only accepts WaveEquation
    subclasses: they are listed to say so, not to be selected. The spec
    dataclasses exported next to them (FieldSpec, ModelSpec, CUDALayoutSpec)
    are not entry points.
    """
    out = {}
    for name, obj in vars(eq_mod).items():
        if (name.startswith("_") or not isinstance(obj, type)
                or issubclass(obj, eq_mod.WaveEquation)
                or dataclasses.is_dataclass(obj)
                or not obj.__module__.startswith(eq_mod.__name__ + ".")):
            continue
        out[name] = obj
    return out


class _Context:
    """Per-call lookups shared by every row: alias groups and the C toolchain."""

    def __init__(self, concrete: dict[str, type], facades: dict[str, type]):
        self.concrete = set(concrete.values())
        self.names_by_class: dict[type, list[str]] = {}
        for name, cls in sorted({**concrete, **facades}.items()):
            self.names_by_class.setdefault(cls, []).append(name)
        self.c_toolchain = bool(sweep.is_torch_binding_available())

    def canonical(self, cls: type) -> str:
        """The one exported name a class is described under; the rest are aliases.

        Identity, not ``__name__``: ``Elastic3D`` is a class whose ``__name__``
        is "Elastic", and that name belongs to the 2-D class.
        """
        names = self.names_by_class[cls]
        return cls.__name__ if cls.__name__ in names else names[0]


def _summary(obj: Any) -> str:
    doc = inspect.cleandoc(getattr(obj, "__doc__", None) or "")
    return " ".join(doc.split("\n\n", 1)[0].split())


def _empty_row(name: str, *, is_facade: bool) -> dict:
    return {
        "name": name,
        "ndim": None,
        "summary": "",
        "is_facade": is_facade,
        "models": [],
        "fields": [],
        "default_source_fields": [],
        "default_receiver_fields": [],
        "pml": {"default": None, "supported": []},
        "free_surface": {"supported": False, "per_edge": False},
        "impl": {"eager": False, "c": False, "c_available": False},
        "boundary_saving_c": False,
        "topography": False,
        "note": None,
    }


def _describe_concrete(name: str, cls: type, ctx: _Context) -> dict:
    row = _empty_row(name, is_facade=False)
    notes: list[str] = []

    def part(label, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- one equation must not hide the rest
            notes.append(f"{label} unavailable: {type(exc).__name__}: {exc}")

    canonical = ctx.canonical(cls)
    if canonical != name:
        notes.append(f"alias of {canonical} (same class)")
    row["summary"] = _summary(cls)
    row["impl"]["eager"] = True

    try:
        eq = cls(**_PROBE_KWARGS)
    except Exception as exc:  # noqa: BLE001
        eq = None
        notes.append(f"instance unavailable: {type(exc).__name__}: {exc}")

    def models():
        by_name = {s.name: s for s in (eq.model_specs if eq is not None else cls.MODEL_SPECS)}
        rows = []
        for model_name in _model_names_for_equation(cls):
            s = by_name.get(model_name)
            rows.append({
                "name": model_name,
                "aliases": list(s.aliases) if s else [],
                "unit": s.unit if s else "",
                "description": s.description if s else "",
                "required": bool(s.required) if s else True,
            })
        row["models"] = rows

    def fields():
        # The propagator validates source_type / receiver_type against the
        # instance's field_specs, which can differ from the class table.
        row["fields"] = [{
            "name": s.name,
            "aliases": list(s.aliases),
            "description": s.description,
            "supports_source": bool(s.supports_source),
            "supports_receiver": bool(s.supports_receiver),
            "internal": bool(s.internal),
        } for s in eq.field_specs]

    def dimensionality():
        # The runner never passes ``ndim`` / ``dim`` (see _PROBE_KWARGS), so
        # the dimensionality is whatever the constructor fixes: the
        # FirstOrderEquation ``ndim`` and SecondOrderEquation ``dim`` defaults
        # are 2 and each 3-D class pins 3 in its own __init__. Read it off the
        # instance; class names are not evidence.
        ndim = int(eq.ndim)
        if ndim not in (2, 3):
            raise ValueError(f"instance ndim={ndim}")
        row["ndim"] = ndim

    def defaults():
        d = eq.defaults()
        row["default_source_fields"] = list(d["default_source_fields"])
        row["default_receiver_fields"] = list(d["default_receiver_fields"])
        default_pml = d["default_pml_type"]
        # ``physics.pml_type`` unset means this default; sweep only knows an
        # alternative where the equation lists one (Acoustic1st: spml).
        supported = getattr(eq, "supported_pml", None)
        row["pml"] = {
            "default": default_pml,
            "supported": list(supported) if supported else [default_pml],
        }
        if callable(getattr(eq, "setup_pml", None)) and len(row["pml"]["supported"]) > 1:
            notes.append(f"fields are listed for pml_type={default_pml!r}; "
                         "another pml_type changes the wavefield set")

    def capabilities():
        # Undeclared flags read as the propagator's own getattr default, which
        # is what a sweep that predates the flag does with every equation.
        row["free_surface"] = {
            "supported": bool(getattr(cls, "supports_free_surface", True)),
            "per_edge": bool(getattr(cls, "supports_per_edge_free_surface", False)),
        }
        c = bool(cls.supports_torch_binding())
        row["impl"] = {"eager": True, "c": c, "c_available": c and ctx.c_toolchain}
        row["boundary_saving_c"] = c and bool(getattr(cls, "supports_boundary_saving_c", True))

    def topography():
        row["topography"] = (bool(getattr(eq, "is_curvilinear", False))
                             or _step_reads_surface(cls))

    part("models", models)
    if eq is not None:
        part("fields", fields)
        part("ndim", dimensionality)
        part("defaults", defaults)
        part("topography", topography)
    part("capabilities", capabilities)
    row["note"] = "; ".join(notes) or None
    return row


def _step_reads_surface(cls: type) -> bool:
    """Whether a method the class resolves to reads the attached surface."""
    for attr in dir(cls):
        obj = inspect.getattr_static(cls, attr)
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        elif isinstance(obj, property):
            obj = obj.fget
        if not inspect.isfunction(obj):
            continue
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        if any(tok in src for tok in _TOPOGRAPHY_ATTRS):
            return True
    return False


def _describe_facade(name: str, obj: type, ctx: _Context) -> dict:
    row = _empty_row(name, is_facade=True)
    row["summary"] = _summary(obj)
    notes = [f"facade, not a WaveEquation subclass: physics.equation does not "
             f"accept {name}"]
    canonical = ctx.canonical(obj)
    if canonical != name:
        notes.append(f"alias of {canonical} (same class)")
    try:
        targets = _facade_targets(obj, ctx)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"dispatch targets unavailable: {type(exc).__name__}: {exc}")
    else:
        if targets:
            notes.append(f"name the equation it dispatches to instead: {targets}")
    row["note"] = "; ".join(notes)
    return row


def _facade_targets(obj: type, ctx: _Context) -> list[str]:
    if callable(getattr(obj, "supported", None)) and callable(getattr(obj, "equation_class", None)):
        classes = [obj.equation_class(method=m, symmetry=s, ndim=n) for m, s, n in obj.supported()]
    else:
        # DAS-style module facade: the concrete equations defined beside it.
        classes = [c for c in ctx.concrete if c.__module__ == obj.__module__]
    return sorted({ctx.canonical(c) for c in classes if c in ctx.concrete})


# ----- model / geometry / wavelet loaders ---------------------------------

def load_model_array(model_ref: dict | ModelRef, base_dir: str | Path) -> np.ndarray:
    """The float32 array the runner loads for one ``ModelRef``.

    ``model_ref`` is treated as a ``models:`` entry of a task YAML that lives
    in ``base_dir``: :func:`sweep_tasks.yaml_io.resolve_paths` (``$VAR``
    expansion, relative ``path`` made absolute), ``ModelRef`` validation, then
    the runner's loader with its ``path`` / ``constant`` / ``dataset`` /
    ``linear_gradient`` sources and ``smooth_sigma_cells``. A ``ModelRef``
    instance goes through the same steps as its dict form.

    Shape is the model's axes, ``(nz, nx)`` or ``(nz, ny, nx)``. The result is
    the caller's own array: the runner hands out its per-process dataset
    cache entry, which is copied here so a caller cannot edit it.
    """
    raw = model_ref.model_dump(mode="json") if isinstance(model_ref, ModelRef) else model_ref
    if not isinstance(raw, dict):
        raise ValueError(f"model_ref must be a mapping or ModelRef, got {type(raw).__name__}.")
    ref = ModelRef.model_validate(resolve_paths(raw, base_dir))
    arr = _model_array(ref)
    if ref.dataset is not None and ref.smooth_sigma_cells is None:
        arr = arr.copy()
    return arr


_SOLVER_TASKS = (ForwardSpec, WavefieldSpec, FWISpec, RTMSpec, LSRTMSpec)


def _check_task(spec: Any) -> None:
    if not isinstance(spec, _SOLVER_TASKS):
        raise TypeError(
            f"expected a validated forward / wavefield / fwi / rtm / lsrtm spec, "
            f"got {type(spec).__name__} (validate with sweep_tasks.load_task_from_dict)."
        )


def _freqsel(spec: Any) -> bool:
    enc = getattr(spec, "source_encoding", None)
    return enc is not None and enc.enabled and enc.mode == "frequency_selection"


def _plan_streaming(spec: Any) -> bool:
    plan = getattr(getattr(spec, "obs", None), "plan", None)
    return plan is not None and plan.sampling is not None


def grid_shape(spec: Any) -> tuple[int, ...]:
    """Model-axes grid shape the runner sets up for this task, before any crop.

    Per task type, as the runner infers it: ``grid.shape`` when given, else
    the first model's (``models[0]`` / first init model / ``velocity_model``
    / ``background_model``). ``model_plan`` crops and per-stage ``dh_m``
    resampling come later and are not applied.
    """
    _check_task(spec)
    if isinstance(spec, (ForwardSpec, WavefieldSpec)):
        return _infer_shape(spec.models, spec.grid.shape)
    if spec.grid.shape is not None:
        return tuple(int(v) for v in spec.grid.shape)
    if isinstance(spec, FWISpec):
        ref = _normalize_fwi_init_models(spec)[0]
    elif isinstance(spec, RTMSpec):
        ref = spec.velocity_model
    else:
        ref = spec.background_model
    if ref.constant is not None:
        return tuple(int(v) for v in ref.shape)
    return tuple(_model_array(ref, mmap=True).shape)


def time_axis(spec: Any) -> tuple[float, int]:
    """``(dt, nt)`` of the task-level solver and wavelet.

    fwi (conventional path) and rtm re-grid to ``data_plan.dt_target_s`` when
    it differs from ``time.dt``, keeping ``dt * nt``; the plan-streaming fwi
    path and the other tasks run at ``time``. Per-stage ``dt_s`` / ``nt`` come
    later and are not applied. Frequency-selection fwi raises ``ValueError``:
    its record length comes from the comb, not from ``time.nt``.
    """
    _check_task(spec)
    if isinstance(spec, FWISpec) and _freqsel(spec):
        raise ValueError(
            "fwi with source_encoding.mode='frequency_selection' sets its time "
            "axis per stage from the comb (probe/steady/slack samples), not "
            "from a task-level wavelet axis."
        )
    dt, nt = float(spec.time.dt), int(spec.time.nt)
    synced = isinstance(spec, RTMSpec) or (isinstance(spec, FWISpec)
                                            and not _plan_streaming(spec))
    if synced:
        plan = spec.data_plan
        if plan is not None and plan.dt_target_s is not None:
            target = float(plan.dt_target_s)
            if abs(target - dt) > 1e-12:
                nt = int(round(dt * nt / target))
                dt = target
    return dt, nt


def build_geometry_arrays(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    """Sources and receivers exactly as the runner builds them from ``spec.geometry``.

    Returns ``sources`` int64 ``(nshots, ndim)`` and ``receivers`` int64
    ``(nshots, nrec, ndim)`` in GRID INDICES (times ``grid.dh`` for metres),
    in sweep's coordinate order ``(x, z)`` / ``(x, y, z)`` -- the reverse of
    the model axes ``(nz, nx)`` / ``(nz, ny, nx)``. Built on
    :func:`grid_shape`. SEG-Y / plan kinds snap at ``grid.dh`` on fwi and
    rtm; forward / wavefield / lsrtm pass no ``dh`` and so reject those kinds,
    exactly as the runner does.

    This is the acquisition before the runner edits it: ``data_plan``
    shot / receiver subsetting, ``model_plan`` cropping, per-stage dedupe and
    ``modeling_override.geometry`` (obs synthesis only) are not applied. fwi
    paths that take geometry from somewhere else (frequency-selection shards,
    ``obs.plan``) raise ``ValueError``.
    """
    _check_task(spec)
    if isinstance(spec, FWISpec):
        if _freqsel(spec):
            raise ValueError(
                "fwi with source_encoding.mode='frequency_selection' takes its "
                "geometry from the coefficient shards, not spec.geometry."
            )
        if spec.obs.plan is not None:
            raise ValueError(
                "fwi with obs.plan builds its geometry from the plan "
                "(materialised or streamed), not from spec.geometry."
            )
    shape = grid_shape(spec)
    if isinstance(spec, (FWISpec, RTMSpec)):
        return _build_geometry(spec.geometry, shape, dh=spec.grid.dh, segy_cache={})
    return _build_geometry(spec.geometry, shape)


def build_wavelet(spec: Any) -> np.ndarray:
    """The task-level source wavelet exactly as the runner builds it.

    1-D float32 of length ``nt``, sampled at ``dt``, both from
    :func:`time_axis`, already multiplied by ``wavelet.scale``. Not applied:
    per-stage ``stages[].wavelet`` / ``dt_s`` / ``nt``, ``bandpass`` /
    ``stages[].bandpass`` filtering, ``modeling_override.wavelet`` (obs
    synthesis only) and the SIREN-npz ``source_delay_s`` obs shift.
    """
    _check_task(spec)
    if isinstance(spec, FWISpec) and _freqsel(spec):
        raise ValueError(
            "fwi with source_encoding.mode='frequency_selection' is wavelet-free: "
            "the runner builds no source wavelet."
        )
    dt, nt = time_axis(spec)
    if isinstance(spec, (FWISpec, RTMSpec)):
        return _build_wavelet(spec.wavelet, spec.time, override_dt=dt, override_nt=nt)
    return _build_wavelet(spec.wavelet, spec.time)
