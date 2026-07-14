"""Equation-class resolution + model-tensor loading + shape/Gardner helpers.
Verbatim from runner.py. Base layer: solver_build / plan_apply import from here."""
from pathlib import Path

import numpy as np
import torch

import sweep.equations as eq_mod
from sweep_tasks.schemas import ModelRef

def _get_equation_class(name: str) -> type:
    classes = eq_mod._equation_classes()
    if name not in classes:
        raise ValueError(
            f"Unknown equation '{name}'. Available: {sorted(classes)}"
        )
    return classes[name]


def _model_names_for_equation(equation_cls: type) -> list[str]:
    specs = getattr(equation_cls, "MODEL_SPECS", None)
    if specs:
        return [s.name for s in specs]
    return _read_class_property(equation_cls, "models") or []


# --- AcousticVRZ option-A coupling: z (impedance) = Gardner(vp); network stays vp-only ---
_VRZ_EQUATIONS = ("AcousticVRZ", "AcousticVRZ3D")


def _gardner_z(vp, water_thr=1505.0, coeff=0.31, exp=0.25):
    """Acoustic impedance z (MRayl) coupled to vp via a water-aware Gardner law.

    z = rho[g/cm3] * vp[km/s]; water (vp <= water_thr) uses rho = 1.0, sediment uses
    Gardner rho = coeff * vp**exp.  Differentiable in ``vp`` so dL/dvp carries the
    density-coupling term — the reparam network predicts ONLY vp (no 2-parameter
    scale mismatch); z is recomputed from the current rendered vp every solver call.
    """
    import torch
    rho = torch.where(vp <= water_thr, torch.ones_like(vp),
                      coeff * vp.clamp_min(1.0) ** exp)
    return rho * (vp / 1000.0)


def _solver_models(leaf, spec):
    """Model list for a solver forward. Non-VRZ equations: ``[vp]`` (unchanged).
    AcousticVRZ/3D (option A): ``[vp, Gardner-z(vp)]``, z coupled to the current vp."""
    if getattr(getattr(spec, "physics", None), "equation", None) in _VRZ_EQUATIONS:
        wthr = 1505.0
        wv = (getattr(spec.reparam, "water_vp_m_s", None)
              if getattr(spec, "reparam", None) else None)
        if wv:
            wthr = float(wv) + 5.0
        return [leaf, _gardner_z(leaf, water_thr=wthr)]
    return [leaf]


def _wavefield_names_for_equation(equation_cls: type) -> list[str]:
    specs = getattr(equation_cls, "FIELD_SPECS", None)
    if specs:
        return [s.name for s in specs]
    return _read_class_property(equation_cls, "wavefields") or []


def _read_class_property(cls: type, name: str):
    import inspect as _inspect

    descriptor = _inspect.getattr_static(cls, name, None)
    if isinstance(descriptor, property):
        try:
            return list(descriptor.fget(cls))
        except Exception:
            return None
    if descriptor is not None and not callable(descriptor):
        try:
            return list(descriptor)
        except Exception:
            return None
    return None


def _load_model_tensor(ref: ModelRef, base_dir: Path | None = None) -> "torch.Tensor":
    import torch

    if ref.constant is not None:
        return torch.full(tuple(ref.shape), float(ref.constant), dtype=torch.float32)
    path = ref.path
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr)


def _infer_shape(models: list[ModelRef], grid_shape: tuple | None) -> tuple[int, ...]:
    if grid_shape is not None:
        return tuple(int(v) for v in grid_shape)
    first = models[0]
    if first.constant is not None:
        return tuple(int(v) for v in first.shape)
    arr = np.load(first.path, mmap_mode="r")
    return tuple(arr.shape)
