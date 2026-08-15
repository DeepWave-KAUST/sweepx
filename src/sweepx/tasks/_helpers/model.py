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
    """Model list for a solver forward.

    * ``leaf`` is a LIST (multi-parameter reparam, ``reparam.free_params`` set):
      the caller already rendered every freed parameter as its own channel/leaf
      in solver-model order (VRZ option C: ``[vp, z]``) — pass it straight
      through.
    * Non-VRZ equations: ``[vp]`` (unchanged).
    * AcousticVRZ/3D (option A, single-channel vp): ``[vp, Gardner-z(vp)]``, z
      coupled to the current vp.
    """
    if isinstance(leaf, (list, tuple)):
        return list(leaf)
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


# Loader results are cached per (dataset, preset, downsample): a single run
# resolves the same ModelRef several times (shape inference, vmax probe, the
# actual load), and a full-size benchmark costs a decode — or a download — each
# time. Keyed by the resolved kwargs, so two refs sharing a dataset share the
# decode.
_DATASET_CACHE: dict = {}


def _dataset_payload(ref: ModelRef) -> dict:
    """Resolve ``ref.dataset`` to a :mod:`sweep.datasets` payload dict (cached).

    Goes through ``info()`` + ``entry.loader()`` rather than ``load()``: the
    embedded demo loaders select their preset with a ``name=`` kwarg, which
    collides with ``load(name, variant, **kwargs)``'s own first parameter. The
    license/citation announcement that ``load`` performs is reproduced here so
    the terms still print once per process.
    """
    from sweep.datasets import info as _ds_info

    ds_name, _, variant = str(ref.dataset).partition(":")
    kwargs: dict = {}
    if ref.preset is not None:
        kwargs["name"] = ref.preset
    if ref.downsample is not None:
        kwargs["downsample"] = (int(ref.downsample)
                                if isinstance(ref.downsample, int)
                                else tuple(int(v) for v in ref.downsample))
    key = (ds_name, variant or None,
           tuple(sorted(kwargs.items(), key=lambda kv: kv[0])))
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    entry = _ds_info(ds_name, variant or None)
    try:                                     # license notice is best-effort
        from sweep.datasets.registry import _announce
        _announce(entry)
    except Exception:                        # noqa: BLE001
        pass
    try:
        payload = dict(entry.loader(**kwargs))
    except TypeError as err:
        raise ValueError(
            f"ModelRef '{ref.name}': dataset '{ref.dataset}' does not accept "
            f"{sorted(kwargs)} ({err}). Embedded demo entries take `preset`; "
            f"full-size entries take `downsample`."
        ) from err
    for k, v in (("name", entry.name), ("variant", entry.variant),
                 ("citation", entry.citation), ("license", entry.license)):
        payload.setdefault(k, v)
    _DATASET_CACHE[key] = payload
    return payload


def _model_array(ref: ModelRef, base_dir: Path | None = None, *,
                 mmap: bool = False) -> "np.ndarray":
    """Resolve any ModelRef source to a float32 numpy array.

    Single entry point for ``path`` / ``constant`` / ``dataset`` so a new
    source works on EVERY task path rather than the handful that happen to
    call the right helper. ``mmap`` is a hint honoured only for the ``path``
    source with no smoothing (the other sources build arrays in memory
    anyway); callers that only need ``.shape`` pass it to avoid a full read.
    """
    if ref.constant is not None:
        arr = np.full(tuple(ref.shape), float(ref.constant), dtype=np.float32)
    elif ref.linear_gradient is not None:
        lg = ref.linear_gradient
        shape = tuple(int(v) for v in ref.shape)
        nz = shape[0]
        n_water = min(int(lg.water_rows), nz)
        col = np.empty(nz, dtype=np.float32)
        col[:n_water] = float(lg.water_vp)
        if nz > n_water:
            # Ramp spans the rows BELOW the water layer, so water_rows does not
            # eat into the velocity range the gradient has to cover.
            col[n_water:] = np.linspace(float(lg.vmin), float(lg.vmax),
                                        nz - n_water, dtype=np.float32)
        # laterally constant: broadcast the column over every remaining axis
        arr = np.ascontiguousarray(
            np.broadcast_to(col.reshape((nz,) + (1,) * (len(shape) - 1)), shape),
            dtype=np.float32)
    elif ref.dataset is not None:
        payload = _dataset_payload(ref)
        fields = sorted(k for k, v in payload.items() if hasattr(v, "shape"))
        # Field pick, most explicit first. The sole-array fallback matters for
        # the embedded demo entries: they return the SELECTED preset under the
        # key "vp" whatever the preset was, so ``preset: vs_true`` on a ModelRef
        # named "vs" would otherwise look for a "vs" key that never exists.
        if ref.dataset_field is not None:
            field = ref.dataset_field
        elif ref.name in fields:
            field = ref.name
        elif len(fields) == 1:
            field = fields[0]
        else:
            raise ValueError(
                f"ModelRef '{ref.name}': dataset '{ref.dataset}' provides "
                f"{fields} and none matches the model name. Set `dataset_field`."
            )
        if field not in payload:
            raise ValueError(
                f"ModelRef '{ref.name}': dataset '{ref.dataset}' has no field "
                f"'{field}'. Available: {fields}."
            )
        arr = np.asarray(payload[field], dtype=np.float32)
    else:
        path = ref.path
        if not path.is_absolute() and base_dir is not None:
            path = (base_dir / path).resolve()
        want_mmap = mmap and ref.smooth_sigma_cells is None
        arr = np.load(path, mmap_mode="r" if want_mmap else None)
        if not want_mmap:
            arr = arr.astype(np.float32)

    if ref.smooth_sigma_cells is not None:
        from scipy.ndimage import gaussian_filter
        arr = gaussian_filter(np.ascontiguousarray(arr, dtype=np.float32),
                              sigma=float(ref.smooth_sigma_cells))
    return arr


def _load_model_tensor(ref: ModelRef, base_dir: Path | None = None) -> "torch.Tensor":
    import torch

    arr = np.ascontiguousarray(_model_array(ref, base_dir), dtype=np.float32)
    return torch.from_numpy(arr)


def _infer_shape(models: list[ModelRef], grid_shape: tuple | None) -> tuple[int, ...]:
    if grid_shape is not None:
        return tuple(int(v) for v in grid_shape)
    first = models[0]
    if first.constant is not None:
        return tuple(int(v) for v in first.shape)
    return tuple(_model_array(first, mmap=True).shape)
