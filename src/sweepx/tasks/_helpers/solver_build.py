"""Solver construction + CFL check + modeling-input resolution. Verbatim from runner.py."""
from pathlib import Path

import numpy as np
import torch

from sweep_tasks.schemas import ModelRef, PhysicsSpec
from sweep_tasks._helpers.geometry import _build_geometry_2d
from sweep_tasks._helpers.model import (
    _get_equation_class,
    _load_model_tensor,
    _model_names_for_equation,
)
from sweep_tasks._helpers.wavelet import _build_wavelet

def _resolve_modeling_inputs(spec, base_wavelet, base_sources, base_receivers, shape):
    """If spec.modeling_override is set, build wavelet/geometry overrides used only
    for the obs-synthesis forward pass. Validates shot/receiver count compatibility.
    """

    mod_override = getattr(spec, "modeling_override", None)
    if mod_override is None:
        return base_wavelet, base_sources, base_receivers, False

    wavelet = base_wavelet
    sources = base_sources
    receivers = base_receivers
    if mod_override.wavelet is not None:
        wavelet = _build_wavelet(mod_override.wavelet, spec.time)
    if mod_override.geometry is not None:
        sources, receivers = _build_geometry_2d(mod_override.geometry, shape)
        if sources.shape[0] != base_sources.shape[0]:
            raise ValueError(
                f"modeling_override.geometry produced {sources.shape[0]} shots but "
                f"inversion expects {base_sources.shape[0]}. Shot count must match "
                "so obs[shot_idx] aligns with the inversion shot indexing."
            )
        if receivers.shape[1] != base_receivers.shape[1]:
            raise ValueError(
                f"modeling_override.geometry produced {receivers.shape[1]} receivers per "
                f"shot but inversion expects {base_receivers.shape[1]}. Receiver count "
                "must match so the (syn - obs) residual is well-defined."
            )
    return wavelet, sources, receivers, True


def _cfl_check(vmax_m_s: float, dh_m: float, dt_s: float, *, threshold: float = 0.85) -> None:
    """Warn / raise if the FD time step violates the Courant condition.

    For 8th-order acoustic FD in 2D the practical safe range is roughly
    ``vmax * dt / dh ≤ 0.5–0.6``. We warn at 0.85 and raise above 1.0
    because anything above 1.0 will produce NaNs on the first step.
    """
    cfl = float(vmax_m_s) * float(dt_s) / float(dh_m)
    if cfl > 1.0:
        raise ValueError(
            f"CFL violation: vmax({vmax_m_s:.0f}) * dt({dt_s}) / dh({dh_m}) = "
            f"{cfl:.3f} > 1.0 — the forward propagator will produce NaNs. "
            f"Lower `time.dt` (try {0.6 * dh_m / vmax_m_s:.4f}s) or coarsen "
            f"`grid.dh`."
        )
    if cfl > threshold:
        print(
            f"[cfl] WARNING: vmax * dt / dh = {cfl:.3f} > {threshold:.2f}. "
            f"Stable forward modeling is not guaranteed; consider dt <= "
            f"{0.6 * dh_m / vmax_m_s:.4f}s for vmax={vmax_m_s:.0f} m/s."
        )


def _build_solver(physics: PhysicsSpec, backend, shape: tuple[int, ...], dh: float, dt: float,
                  nt: int, dev: "torch.device") -> "Any":
    from sweep.propagator.torch import PropTorch

    equation_cls = _get_equation_class(physics.equation)
    equation = equation_cls(
        spatial_order=physics.spatial_order,
        device=dev,
        backend="torch",
    )

    prop_kwargs: dict[str, Any] = dict(
        shape=tuple(shape),
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        abcn=physics.abcn,
        free_surface=physics.free_surface,
        pml_type=physics.pml_type,
        source_type=list(physics.source_type),
        receiver_type=list(physics.receiver_type),
        use_ckpt=backend.use_ckpt,
    )
    if backend.use_ckpt:
        prop_kwargs["ckpt_chunks"] = backend.ckpt_chunks

    # Irregular free-surface topography → boundary-fitted curvilinear grid.
    # PropTorch builds the metric tensors from the 1-D surface-row array.
    if getattr(physics, "topography", None) is not None:
        prop_kwargs["topography"] = np.load(physics.topography)

    if backend.impl == "eager":
        eager = (backend.eager_options.to_dataclass()
                 if backend.eager_options is not None
                 else None)
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="torch",
            impl="eager",
            eager_options=eager,
        )

    cuda = (backend.cuda_options.to_dataclass()
            if backend.cuda_options is not None
            else None)
    return PropTorch(
        equation,
        **prop_kwargs,
        backend="torch",
        impl="c",
        cuda_options=cuda,
    )


def _solver_models_in_order(equation_cls: type, refs: list[ModelRef], base_dir: Path | None,
                            dev: "torch.device") -> list["torch.Tensor"]:
    """Return tensors in the order the equation declares its MODEL_SPECS."""

    required = _model_names_for_equation(equation_cls)
    by_name = {r.name: r for r in refs}
    missing = set(required) - set(by_name)
    if missing:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' requires models {required}; "
            f"missing: {sorted(missing)}"
        )
    extra = set(by_name) - set(required)
    if extra:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' does not use model(s) {sorted(extra)}."
        )
    return [_load_model_tensor(by_name[name], base_dir).to(dev) for name in required]


def _validate_single_model(equation_cls: type, ref: ModelRef) -> None:
    required = _model_names_for_equation(equation_cls)
    if len(required) != 1:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' expects models {required}, "
            "but this task type only supports single-model equations (e.g. Acoustic)."
        )
    if ref.name != required[0]:
        raise ValueError(
            f"ModelRef name '{ref.name}' does not match equation's required model '{required[0]}'."
        )


# ---------- FWI / LSRTM training helpers ---------------------------------
