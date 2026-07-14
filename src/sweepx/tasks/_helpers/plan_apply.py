"""Apply model/data plan overrides to the FWI setup + build inv tensors. Verbatim from runner.py."""
import numpy as np
import torch

from sweep_tasks.schemas import ModelRef
from sweep_tasks._helpers.model import _load_model_tensor, _model_names_for_equation

def _apply_model_plan_to_fwi(
    spec, init_models_loaded: list[tuple[str, "np.ndarray"]],
    sources_idx, receivers_idx, dh: float,
):
    """Apply ``spec.model_plan`` (Gap 2): crop each loaded vp array to the
    window + shift / drop sources / receivers so their grid indices line up
    with the cropped model.

    Returns:
        cropped_vp_by_name: dict[name -> cropped ndarray]
        sources_idx_new, receivers_idx_new (or unchanged if no plan / no drop)
        keep_shots_mask (None when drop_outside_sources is False)
    """
    if spec.model_plan is None:
        return ({name: vp for name, vp in init_models_loaded},
                sources_idx, receivers_idx, None)

    plan = spec.model_plan
    from sweep_io.plan import ModelPlan, apply_model_plan
    from sweep_io.geometry import PhysicalGeometry

    # Lift current grid-index geometry into meters using `dh`.
    pg = PhysicalGeometry(
        sources_xyz_m=sources_idx.astype("float64") * float(dh),
        receivers_xyz_m=receivers_idx.astype("float64") * float(dh),
        dt=float(spec.time.dt), nt=int(spec.time.nt),
    )

    dp = ModelPlan(
        x_window_m=plan.x_window_m,
        y_window_m=plan.y_window_m,
        z_window_m=plan.z_window_m,
        drop_outside_sources=plan.drop_outside_sources,
        drop_outside_receivers=plan.drop_outside_receivers,
    )

    cropped_by_name: dict = {}
    new_pg = pg
    keep_mask = None
    # Apply the same crop to every model tensor. Geometry is rebased only on
    # the FIRST pass (subsequent calls would double-shift).
    first = True
    for name, vp in init_models_loaded:
        if vp.ndim == 2:
            vp_dh = (float(dh), float(dh))
        elif vp.ndim == 3:
            vp_dh = (float(dh), float(dh), float(dh))
        else:
            raise ValueError(f"ModelPlan: unsupported vp ndim {vp.ndim} for {name!r}")
        vp_out, geom_out, src_keep = apply_model_plan(
            dp, vp.astype("float32", copy=False), dh=vp_dh,
            geom=new_pg if first else None,
        )
        cropped_by_name[name] = vp_out
        if first and geom_out is not None:
            new_pg = geom_out
            keep_mask = src_keep
        first = False

    # Re-snap to grid indices at the SAME dh, no dedupe (positions are exact).
    gg, _ = new_pg.to_grid(
        dh=(float(dh),) * pg.ndim, dedupe=False,
    )
    return cropped_by_name, gg.sources.astype(np.int64), gg.receivers.astype(np.int64), keep_mask


def _apply_data_plan_to_fwi(
    spec, sources_idx, receivers_idx, obs, dh: float, dt: float, nt: int, dev
):
    """Apply ``spec.data_plan``.

    Subsets shots / receivers / time *after* obs has been generated. Lifts
    the grid-index geometry into ``sweep_io.PhysicalGeometry`` so the
    offset filter works in real meters, then drops back to grid indices.

    Receiver-axis subsetting requires a uniform mask across shots (typical
    for streamers). A non-uniform mask (offset filter where each shot picks
    a different receiver set) raises a clear error.
    """
    if spec.data_plan is None:
        return sources_idx, receivers_idx, obs

    import torch
    from sweep_io.geometry import PhysicalGeometry
    from sweep_io.plan import DataPlan, apply_data_plan

    dh_f = float(dh)
    sources_xyz = sources_idx.astype("float64") * dh_f
    receivers_xyz = receivers_idx.astype("float64") * dh_f

    pg = PhysicalGeometry(
        sources_xyz_m=sources_xyz,
        receivers_xyz_m=receivers_xyz,
        dt=dt,
        nt=nt,
    )

    plan_kwargs = spec.data_plan.model_dump(exclude_none=True)
    plan = DataPlan(**plan_kwargs)

    # Both backends now share the canonical ``(nshots, nt, nrec[, nfield])``
    # layout (the c-backend was aligned in geophyai commit 21041c5; the
    # eager backend was always canonical). Time is axis -3 / -2 depending
    # on whether the trailing channel axis is present.
    obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
    time_axis = -3 if obs_np.ndim >= 4 else -2
    receiver_axis = -2 if obs_np.ndim >= 4 else -1

    pg_planned, obs_planned, rcv_mask = apply_data_plan(
        plan, pg, obs_np, time_axis=time_axis, receiver_axis=receiver_axis,
    )

    # Receiver-axis subsetting: only handle uniform-across-shots masks.
    if not np.all(rcv_mask == rcv_mask[0:1]):
        raise NotImplementedError(
            "data_plan: non-uniform per-shot receiver mask (typical when an "
            "offset filter coexists with non-streamer geometry) is not yet "
            "supported. Drop the offset filter, or apply it upstream."
        )
    uniform_keep = rcv_mask[0]
    if not uniform_keep.all():
        slicer = [slice(None)] * obs_planned.ndim
        slicer[receiver_axis] = np.flatnonzero(uniform_keep)
        obs_planned = obs_planned[tuple(slicer)]
        # Also drop the same receivers from geometry.
        pg_planned = PhysicalGeometry(
            sources_xyz_m=pg_planned.sources_xyz_m,
            receivers_xyz_m=pg_planned.receivers_xyz_m[:, uniform_keep, :],
            dt=pg_planned.dt,
            nt=pg_planned.nt,
            meta=pg_planned.meta,
        )

    # Back to grid indices.
    gg, _ = pg_planned.to_grid(dh=(dh_f,) * pg.ndim, dedupe=False)
    sources_new = gg.sources
    receivers_new = gg.receivers

    # If time was resampled, push that into spec.time.nt and the obs.
    obs_t = torch.as_tensor(obs_planned, device=dev) if isinstance(obs, torch.Tensor) else obs_planned
    return sources_new, receivers_new, obs_t


def _normalize_fwi_init_models(spec) -> list:
    """Return the list of ModelRef objects for FWI, accepting either init_model or init_models."""

    if spec.init_models is not None:
        return list(spec.init_models)
    return [spec.init_model]


def _build_inv_tensors(init_models: list, dev, equation_cls,
                       *, overrides: dict[str, "np.ndarray"] | None = None):
    """Load each ModelRef as a requires_grad=True tensor and return both
    ordered list (per equation MODEL_SPECS) and name-keyed dict.

    ``overrides`` (used by Gap 2 model_plan) maps model name -> already-loaded
    numpy array; when present, the ModelRef's path / constant fields are
    ignored for that name.
    """

    required = _model_names_for_equation(equation_cls)
    by_name = {m.name: m for m in init_models}
    missing = set(required) - set(by_name)
    if missing:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' requires models {required}; "
            f"init_model(s) missing {sorted(missing)}."
        )
    extra = set(by_name) - set(required)
    if extra:
        raise ValueError(
            f"Equation '{equation_cls.__name__}' does not use model(s) {sorted(extra)}."
        )
    in_order = []
    by_tensor: dict[str, "torch.Tensor"] = {}
    for name in required:
        if overrides is not None and name in overrides:
            import torch
            arr = np.ascontiguousarray(overrides[name].astype(np.float32, copy=False))
            t = torch.from_numpy(arr).to(dev).requires_grad_(True)
        else:
            t = _load_model_tensor(by_name[name]).to(dev).requires_grad_(True)
        in_order.append(t)
        by_tensor[name] = t
    return in_order, by_tensor, required
