"""Multiscale-stage prep + tensor/obs resampling + syn bandpass. Verbatim from runner.py."""
import os
import numpy as np
import torch

from sweepx.tasks.preproc.filter import bandpass as _bandpass_cpu
from sweepx.tasks.preproc.filter import bandpass_torch as _bandpass_torch_fft
from sweepx.tasks._helpers.optimizer import (
    _apply_stage_lr_scale,
    _build_optimizer,
    _build_reparam_optimizer,
    _remember_initial_lrs,
)
from sweepx.tasks._helpers.solver_build import _build_solver
from sweepx.tasks._helpers.stage_grid import (
    _resample_to_grid,
    _set_physical_inr_frame,
    _snap_covered,
    _stage_grid_shape,
)
from sweepx.tasks._helpers.water import rebake_water_, stamp_water_
from sweepx.tasks._helpers.util import _is_cuda_dev
from sweepx.tasks._helpers.wavelet_build import _build_wavelet

def _normalise_stage_list(spec) -> list:
    """Return the effective stage list (single-stage fallback when spec.stages is None)."""

    if spec.stages:
        return list(spec.stages)
    from sweepx.tasks.schemas import StageSpec
    return [StageSpec(epochs=spec.epochs, wavelet=None, lr_scale=1.0)]


def _next_stage_grid(stage, dh: float, dt: float, nt: int) -> tuple[float, float, int]:
    """The ``(dh, dt, nt)`` a stage runs on, given those of the stage before it.

    A stage inherits every value it does not set; a new ``dt_s`` without
    ``nt`` keeps the record length ``nt * dt``.
    """
    new_dh = float(stage.dh_m) if stage.dh_m is not None else dh
    new_dt = float(stage.dt_s) if stage.dt_s is not None else dt
    if stage.nt is not None:
        new_nt = int(stage.nt)
    elif abs(new_dt - dt) > 1e-12:
        new_nt = int(round(nt * dt / new_dt))
    else:
        new_nt = nt
    return new_dh, new_dt, new_nt


def _stage_grid_chain(stages, dh: float, dt: float, nt: int) -> list[dict]:
    """Per stage, the grid it runs on and what its entry rebuilds.

    ``dh, dt, nt``: the setup grid, which stage 0 starts from. Each entry
    holds the stage's ``dh``, ``dt``, ``nt`` and whether entering it changes
    the space grid (``grid_changed``: model resample, geometry re-snap) or
    the time axis (``time_changed``), as :func:`_prepare_stage` decides.
    """
    chain = []
    for stage in stages:
        new_dh, new_dt, new_nt = _next_stage_grid(stage, dh, dt, nt)
        chain.append({
            "dh": new_dh, "dt": new_dt, "nt": new_nt,
            "grid_changed": abs(new_dh - dh) > 1e-12,
            "time_changed": abs(new_dt - dt) > 1e-12 or new_nt != nt,
        })
        dh, dt, nt = new_dh, new_dt, new_nt
    return chain


def _stage_checkpoint_fields(state: dict, stage_idx: int) -> dict:
    """Checkpoint entries that put a resumed run back on its stage.

    Beyond models / optimizer / losses / RNG a resume needs the stage and its
    grid (the models live on it), and under reparam the network itself: its
    weights are the model, ``models['vp']`` is only a render of them. The
    buffers that stage entries write into the network (base velocity, water
    mask) and the optimizer's parameter groups (``hash.c2f.growing`` adds one
    per grown level) are saved too, so nothing has to be re-derived.
    """
    out = {
        "stage": int(stage_idx),
        "grid": {"dh": float(state["dh"]), "dt": float(state["dt"]),
                 "nt": int(state["nt"]),
                 "shape": [int(s) for s in state["shape"]]},
    }
    net = state.get("reparam_net")
    if net is not None:
        names = {id(p): n for n, p in net.named_parameters()}
        water = getattr(net, "water_mask", None)
        region = state.get("water_region_mask")
        out["reparam_net"] = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
        out["reparam_param_groups"] = [[names[id(p)] for p in g["params"]]
                                       for g in state["optimizer"].param_groups]
        out["reparam_buffers"] = {
            "base_velocity": net.base_velocity.detach().cpu().clone(),
            "water_mask": None if water is None else water.detach().cpu().clone(),
        }
        out["water_region_mask"] = (None if region is None
                                    else region.detach().cpu().clone())
    return out


def _restore_stage_state(*, spec, ckpt: dict, state: dict, stages: list,
                         required_names: list[str], dev):
    """Put ``state`` where the checkpointed epoch left it.

    Returns the checkpointed optimizer's lrs, one per param group, for
    ``_bind_stage_scheduler`` when the run continues inside that stage.

    The checkpoint was written after epoch ``ckpt['epoch']``, inside the stage
    holding that epoch, so its models are on that stage's grid. ``state`` is
    the setup state on the native grid. This moves it to the stage's grid
    (``dh``/``dt``/``nt``/shape from the stage chain) and loads the models,
    the reparam network and the optimizer there; the stage loop then either
    continues that stage (``_prepare_stage(resumed=True)``) or enters the next
    one exactly as an uninterrupted run would. The geometry and solver are
    marked stale when an earlier entry rebuilt them, so ``_prepare_stage``
    rebuilds them for the stage grid.

    Checkpoints written before the stage entries existed hold only models,
    optimizer, losses and RNG; on the grid path they load (the stage is the
    one holding the epoch). Under reparam they do not: they lack the network.
    """
    epoch = int(ckpt["epoch"])
    chain = _stage_grid_chain(stages, state["dh"], state["dt"], state["nt"])
    ends = np.cumsum([int(s.epochs) for s in stages])
    s = min(int(np.searchsorted(ends, epoch, side="right")), len(stages) - 1)
    grid = chain[s]
    native_shape = tuple(state["pristine_shape"])
    native_dh = float(state["pristine_dh"])
    shape = _stage_grid_shape(native_shape, native_dh, grid["dh"])

    models = ckpt["models"]
    missing = [n for n in state["inv_by_name"] if n not in models]
    if missing:
        raise ValueError(f"resume: the checkpoint has no {missing} model(s); "
                         f"it holds {sorted(models)}")
    wrong = {n: tuple(models[n].shape) for n in state["inv_by_name"]
             if tuple(models[n].shape) != shape}
    if wrong:
        written = ckpt.get("grid")
        at = (f" (written in stage {ckpt.get('stage')} on a {written['dh']} m "
              f"{tuple(written['shape'])} grid)" if written else "")
        raise ValueError(
            f"resume: the checkpoint of epoch {epoch}{at} holds models of shape "
            f"{wrong}, but epoch {epoch} falls in stage {s} of this config, "
            f"which runs on a {grid['dh']} m {shape} grid. The stages changed "
            "since the checkpoint was written; restore them, or start the run "
            "without resume.")

    state["dh"], state["dt"], state["nt"] = grid["dh"], grid["dt"], grid["nt"]
    state["shape"] = shape
    if any(c["grid_changed"] for c in chain[:s + 1]):
        state["_resume_geometry_stale"] = True
    if any(c["grid_changed"] or c["time_changed"] for c in chain[:s + 1]):
        state["_resume_solver_stale"] = True

    def _tensor(name, *, grad):
        old = state["inv_by_name"][name]
        t = models[name].detach().to(device=dev, dtype=old.dtype).clone()
        return t.requires_grad_(True) if grad else t

    net = state.get("reparam_net")
    if net is None:
        by_name = {n: _tensor(n, grad=True) for n in state["inv_by_name"]}
        optimizer = _build_optimizer(spec.optimizer, by_name, required_names)
        state["initial_lrs"] = _remember_initial_lrs(optimizer)
    else:
        if "reparam_net" not in ckpt:
            raise ValueError(
                f"resume: the checkpoint of epoch {epoch} holds no reparam "
                "network, only its rendered vp (checkpoints written before "
                "sweep-tasks saved it), so the run cannot continue from it: "
                "the network would restart from its initial weights. Start "
                "the run again without resume.")
        net.load_state_dict(ckpt["reparam_net"])  # grows a GrowingHashGrid as saved
        bufs = ckpt["reparam_buffers"]
        net.water_mask = None
        net.update_base_velocity(bufs["base_velocity"].to(dev))
        if bufs["water_mask"] is not None:
            net.water_mask = bufs["water_mask"].to(dev)
        _set_physical_inr_frame(net, shape=shape, dh=grid["dh"],
                                frame_shape=native_shape, frame_dh=native_dh)
        if ckpt.get("water_region_mask") is not None:
            state["water_region_mask"] = ckpt["water_region_mask"].to(dev)
        params = dict(net.named_parameters())
        groups = [[params[n] for n in g] for g in ckpt["reparam_param_groups"]]
        optimizer = _build_reparam_optimizer(spec.optimizer, groups[0],
                                             float(spec.reparam.lr))
        # The base lrs of the optimizer's last build: a stage's
        # optimizer_reset, else setup's (reparam.init_optimizer_from included).
        if any(bool(st.optimizer_reset) for st in stages[:s + 1]):
            state["initial_lrs"] = _remember_initial_lrs(optimizer)
        for g in groups[1:]:
            optimizer.add_param_group({"params": g})
        # vp is the checkpointed render; the other models are not trained here.
        by_name = {n: (_tensor(n, grad=False) if n == "vp"
                       else _tensor(n, grad=state["inv_by_name"][n].requires_grad))
                   for n in state["inv_by_name"]}
    optimizer.load_state_dict(ckpt["optimizer"])
    state["optimizer"] = optimizer
    state["inv_by_name"] = by_name
    state["inv_in_order"] = [by_name[n] for n in required_names]
    return [g["lr"] for g in optimizer.param_groups]


# ---------------------------------------------------------------------------
# Per-stage rebuilds (Gaps 4 + 5)
# ---------------------------------------------------------------------------


def _prepare_stage(
    *, spec, stage, state: dict, equation_cls, required_names: list[str],
    dev, dist_info, stage_idx: int, resumed: bool = False,
) -> None:
    """Mutate ``state`` so the runner can run this stage end-to-end.

    Detects which of ``(dh, dt, nt, batch_size, lr_scale)`` changed against the
    previous stage state. Triggers any combination of:

    - vp resample onto the stage grid (``_helpers.stage_grid``: same physical
      samples as the geometry, ``i * dh`` from the native origin) + fresh
      leaf tensors
    - solver rebuild at the new ``(shape, dh, dt, nt)``
    - geometry re-snap from the cached pristine PhysicalGeometry
    - optimizer re-init (Adam state is shape-bound) + initial_lrs cache

    Two things are rebuilt UNCONDITIONALLY on every stage entry, both from
    their pristine source rather than from the previous stage's result: the
    obs (time-resample + bandpass from the cached pristine obs) and the
    wavelet (``_build_wavelet`` from spec). That is what makes per-stage
    bandpasses non-composing, as the schema documents.

    ``resumed``: the run stopped inside this stage, and ``state`` already
    holds the checkpointed models, network and optimizer on this stage's grid
    (``_restore_stage_state``). Everything the stage runs with is rebuilt as
    at entry, but what only entering a stage does is not redone: no
    ``optimizer_reset`` (it would drop the restored Adam moments) and no
    water reset (``water.mode: reset_each_stage``).
    """
    import torch

    new_dh, new_dt, new_nt = _next_stage_grid(stage, state["dh"], state["dt"],
                                              state["nt"])

    grid_changed = abs(new_dh - state["dh"]) > 1e-12
    time_changed = (abs(new_dt - state["dt"]) > 1e-12) or (new_nt != state["nt"])
    if resumed and (grid_changed or time_changed):
        raise AssertionError(
            f"stage {stage_idx}: resumed inside the stage, but the restored state "
            f"is on dh={state['dh']} dt={state['dt']} nt={state['nt']}, not the "
            f"stage's dh={new_dh} dt={new_dt} nt={new_nt}")
    # Set by _restore_stage_state: an earlier stage entry rebuilt these, so the
    # setup ones in ``state`` are not the ones this stage would have.
    geometry_stale = state.pop("_resume_geometry_stale", False)
    solver_stale = state.pop("_resume_solver_stale", False)
    if resumed and dist_info.is_root:
        print(f"[stage {stage_idx}] resumed inside the stage: optimizer_reset "
              f"and water reset are not repeated")

    if grid_changed and dist_info.is_root:
        print(f"[stage {stage_idx}] dh: {state['dh']} -> {new_dh}")
    if time_changed and dist_info.is_root:
        print(f"[stage {stage_idx}] dt: {state['dt']} -> {new_dt}, nt: {state['nt']} -> {new_nt}")

    # The stage grid samples the native model at i * new_dh from the native
    # origin and spans the whole native extent, so geometry snapped with
    # round(x / new_dh) and the resampled model agree (``_helpers.stage_grid``).
    native_shape = tuple(state.get("pristine_shape", state["shape"]))
    native_dh = float(state["pristine_dh"])
    stage_shape = _stage_grid_shape(native_shape, native_dh, new_dh)
    on_native_grid = stage_shape == native_shape and abs(new_dh - native_dh) <= 1e-12

    # ---- Geometry re-snap from pristine physical positions -----------------
    # We re-snap whenever:
    #   - the stage changes dh (grid_changed)
    #   - state hasn't been initialised yet
    #   - dedupe is enabled and we haven't applied it at this dh yet (the
    #     initial geometry parse always keeps all receivers; per-stage
    #     dedupe is the first time it gets applied at the stage's dh)
    need_geom_resnap = (
        grid_changed
        or geometry_stale
        or "sources" not in state
        # Re-snap every stage when dedupe is on. The source-dedup step below
        # destructively shrinks per_shot_keep_idx (fewer rows than pristine),
        # while the obs is ALWAYS re-derived from pristine; on a FIXED grid
        # (no grid_changed) the old guard skipped the re-snap, so the next
        # stage masked pristine with a stale, too-short per_shot_keep_idx and
        # raised IndexError. Re-snapping is idempotent at an unchanged dh, and
        # multiscale already re-snaps on every grid change, so this only affects
        # the fixed-grid frequency-continuation path (which it fixes).
        or state.get("dedupe_grid_snap", False)
    )
    if need_geom_resnap:
        from sweepx.io.geometry import PhysicalGeometry
        pg: PhysicalGeometry = state["pristine_physical_geom"]
        dedupe_flag = state.get("dedupe_grid_snap", True)
        dedup_method = state.get("dedup_method", "nearest")
        gg, mask = pg.to_grid(
            dh=(float(new_dh),) * pg.ndim,
            dedupe=dedupe_flag,
            dedup_method=dedup_method,
        )
        if not on_native_grid:
            # A finer stage can round a position the native grid holds (within
            # half a native cell of its edge) one cell past the stage edge,
            # where the solver would record the absorbing layer. Put those on
            # the edge node: no further than the native snap already moved them.
            gg_n, _ = pg.to_grid(dh=(native_dh,) * pg.ndim, dedupe=False)
            gg.sources, n_src_mv = _snap_covered(
                gg.sources, gg_n.sources, stage_shape, native_shape)
            gg.receivers, n_rec_mv = _snap_covered(
                gg.receivers, gg_n.receivers, stage_shape, native_shape)
            if (n_src_mv or n_rec_mv) and dist_info.is_root:
                print(f"[stage {stage_idx}] moved {n_src_mv} source(s) and "
                      f"{n_rec_mv} receiver(s) within half a native cell of the "
                      f"model edge onto the stage grid's edge node")
        uniform = bool(np.all(mask == mask[0:1]))
        if not dedupe_flag or uniform:
            # Fast path: keep all (dedupe=false) or uniform mask across shots.
            keep_idx = np.flatnonzero(mask[0])
            state["sources"] = gg.sources.astype(np.int64)
            state["receivers"] = gg.receivers[:, keep_idx, :].astype(np.int64)
            state["receiver_keep_idx"] = keep_idx
            state["per_shot_keep_idx"] = None
            state["nshots"] = int(state["sources"].shape[0])
        else:
            # Non-uniform mask path: streamer-style acquisition where the
            # source position shifts per shot, so each shot's dedupe pattern
            # differs. Build per-shot ``keep_idx`` arrays, then TRUNCATE to
            # the minimum keep count across shots so the resulting tensors
            # stay rectangular (no ragged). For each shot, ``keep_idx`` is
            # already in receiver-index order from ``flatnonzero``; truncate
            # via ``[:n_common]`` preserves spatial order.
            per_shot_keep: list[np.ndarray] = [np.flatnonzero(mask[s]) for s in range(mask.shape[0])]
            counts = np.asarray([k.size for k in per_shot_keep])
            n_common = int(counts.min())
            if n_common <= 0:
                raise ValueError(
                    f"[stage {stage_idx}] dedupe removed all receivers in at least one shot"
                )
            truncated = np.stack(
                [k[:n_common].astype(np.int64) for k in per_shot_keep], axis=0
            )  # (nshots, n_common)
            # Build deduped receivers: gg.receivers is already the deduped
            # output (with dropped traces zeroed in the dropped slots but
            # mask tells us which to keep). However ``flatnonzero(mask[s])``
            # returns indices into the ORIGINAL receiver axis — we use those
            # against ``gg.receivers`` (which has the same axis length, with
            # mask[s][i] = True iff receiver i survived for shot s).
            # Truncated keep gives a rectangular (nshots, n_common) index.
            rec_out = np.zeros((mask.shape[0], n_common, gg.receivers.shape[-1]), dtype=np.int64)
            for s in range(mask.shape[0]):
                rec_out[s] = gg.receivers[s, truncated[s], :].astype(np.int64)
            state["sources"] = gg.sources.astype(np.int64)
            state["receivers"] = rec_out
            state["per_shot_keep_idx"] = truncated.astype(np.int64)
            # Maintain receiver_keep_idx for backward-compat (used to slice
            # pristine obs when uniform); keep first-shot keep as a
            # representative (obs masking path below uses per_shot_keep_idx
            # when available).
            state["receiver_keep_idx"] = truncated[0].copy()
            state["nshots"] = int(state["sources"].shape[0])
            if dist_info.is_root:
                drop = int(mask.shape[1] - n_common)
                print(f"[stage {stage_idx}] dedupe={dedup_method!r}: "
                      f"per-shot rec count {counts.min()}–{counts.max()}, "
                      f"truncated to common {n_common} (dropped {drop} per shot)")
        # Record the dh at which the current geometry was snapped so the
        # check above doesn't re-run on every stage entry.
        state["_geom_applied_at_dh"] = float(new_dh)

    # ---- Stage grid shape (spans the native extent at new_dh) --------------
    old_dh = float(state["dh"])
    if grid_changed or "shape" not in state:
        if dist_info.is_root and grid_changed:
            print(f"[stage {stage_idx}] vp shape: {state['shape']} -> {stage_shape} "
                  f"(native {native_shape} at {native_dh} m)")
        state["shape"] = stage_shape
    # The `water:` layer on THIS stage's grid: the same physical layer on every
    # grid (``_helpers.water``), rebuilt from its definition, never resampled.
    water = state.get("water")
    if water is not None:
        state["water_mask"] = torch.from_numpy(
            water.mask_on_grid(state["shape"], new_dh)).to(dev)
    else:
        state["water_mask"] = None

    # ---- vp resample ----------------------------------------------------
    # Two paths, both sampling at the stage grid's physical positions:
    #  - Reparam off: resample each CURRENT leaf tensor from the previous
    #    stage's grid (old_dh) and make a fresh `requires_grad=True` leaf
    #    (Adam state is shape-bound, so we rebuild the optimizer below).
    #  - Reparam on: keep the same VelocityINR. Update its `base_velocity`
    #    buffer from the pristine init, resampled to the new grid, and put its
    #    coordinates in the native frame — and DO NOT rebuild the optimizer
    #    (its state on the network's params is independent of grid shape and
    #    is the entire point of the network-as-vp parameterization).
    if grid_changed:
        reparam_net = state.get("reparam_net")
        if reparam_net is not None:
            pristine_base = state["pristine_base_vp"]
            new_base = _resample_to_grid(pristine_base, native_dh,
                                         state["shape"], new_dh)
            # Hand the network this stage's water mask, or update_base_velocity
            # drops the old-shape pin and the water inverts freely from here
            # on. In reset_each_stage mode there is no render pin: the block
            # below re-bakes the base in the water instead.
            water_mask = (state["water_mask"]
                          if water is not None and water.pin else None)
            reparam_net.update_base_velocity(new_base, water_mask=water_mask)
            # update_base_velocity rebuilt the coordinates as i/n of the new
            # shape; without this the learned field would stretch with the grid.
            _set_physical_inr_frame(reparam_net, shape=state["shape"], dh=new_dh,
                                    frame_shape=native_shape, frame_dh=native_dh)
            with torch.no_grad():
                rendered = reparam_net().detach()
            state["inv_by_name"]["vp"] = rendered
            state["inv_in_order"] = [
                rendered if n == "vp" else state["inv_by_name"][n]
                for n in required_names
            ]
        else:
            new_inv: list = []
            new_by_name: dict = {}
            for name in required_names:
                old_t = state["inv_by_name"][name]
                new_t = _resample_to_grid(old_t, old_dh, state["shape"],
                                          new_dh).requires_grad_(True)
                new_inv.append(new_t)
                new_by_name[name] = new_t
            state["inv_in_order"] = new_inv
            state["inv_by_name"] = new_by_name

    # ---- Water on a grid model: stamped at every stage entry ---------------
    # Both modes start the stage at water.velocity; `pin` then holds it (the
    # train step zeroes the gradient there and re-stamps after each step),
    # `reset_each_stage` lets it invert until the next stage entry. A stage
    # resumed inside keeps its checkpointed model (no reset).
    if water is not None and state.get("reparam_net") is None and not resumed:
        stamp_water_(state["inv_by_name"]["vp"], state["water_mask"],
                     water.velocity)

    # ---- Water reset-each-stage on a reparam network ------------------------
    # water.mode = reset_each_stage: snap the water back to water.velocity at
    # the START of every stage, then let it invert FREELY within the stage. The
    # INR is a GLOBAL additive net, so resetting the base alone is not enough —
    # the net's learned water delta persists across stages. So (a) clear the
    # render-time pin and (b) re-bake the base in the water so the current
    # render there equals the water velocity (``base[water] += v - render``);
    # as the net evolves the water drifts, and the next stage snaps it back.
    # Runs EVERY stage (incl. the fixed-grid path, where grid_changed is False
    # and the block above is skipped). Deterministic → DDP-consistent.
    _net = state.get("reparam_net")
    if _net is not None and water is not None and not water.pin and not resumed:
        _n_wet = rebake_water_(_net, state["water_mask"], water.velocity)
        with torch.no_grad():
            _render_post = _net().detach()
        state["inv_by_name"]["vp"] = _render_post
        state["inv_in_order"] = [
            _render_post if n == "vp" else state["inv_by_name"][n]
            for n in required_names
        ]
        if getattr(dist_info, "is_root", True):
            print(f"[stage {stage_idx}] water reset-each-stage: snapped "
                  f"{_n_wet} water cells to {water.velocity:.0f} m/s, "
                  f"free within stage", flush=True)

    # ---- Rebuild solver ----------------------------------------------------
    if grid_changed or time_changed or solver_stale or "solver" not in state:
        state["solver"] = _build_solver(
            spec.physics, spec.backend, state["shape"], new_dh, new_dt, new_nt, dev,
        )
        # Drop cached local-window solvers — they were built for the
        # previous stage's (dh, dt, nt) and are no longer valid.
        if "local_solver_cache" in state:
            state["local_solver_cache"].clear()

    # ---- Rebuild wavelet ---------------------------------------------------
    # Rebuild the wavelet from spec at EVERY stage entry. ``_build_wavelet`` is
    # a deterministic function of (spec, dt, nt) and costs one array build (plus
    # at most one small npz read), so rebuilding unconditionally is cheap — and
    # it is the only way to guarantee the schema's promise that stages do NOT
    # compose their filters.
    #
    # A conditional rebuild is not enough. ``target='wavelet'`` writes the
    # FILTERED wavelet back into this same cache slot, and nothing else ever
    # resets it. So a later stage that keeps the same dt/nt, overrides no
    # wavelet, and uses the default ``target='syn'`` would silently inherit the
    # earlier stage's narrow-band source while its obs is re-derived from
    # pristine at the new band — leaving syn and obs on different bands.
    wav_spec = stage.wavelet if stage.wavelet is not None else spec.wavelet
    state["wavelet"] = _build_wavelet(
        wav_spec, spec.time, override_dt=new_dt, override_nt=new_nt,
    )

    # ---- Optional wavelet bandpass (target='wavelet') ----------------------
    # When ``stage.bandpass.target == 'wavelet'``, we band-pass the source
    # wavelet ONCE at stage entry. Syn forward is then naturally band-limited
    # without per-iteration filtering in the autograd path. fwi_workflow-dev
    # filters syn (target='syn'); both yield equivalent gradients for a
    # linear wave equation.
    # The wavelet above is always freshly built from spec, so this filter is
    # applied to the pristine broadband wavelet every time — never composed.
    bp_spec = stage.bandpass
    if bp_spec is not None and getattr(bp_spec, "target", "syn") == "wavelet":
        # Bandpass the wavelet via scipy (non-autograd: wavelet is constant
        # input to the solver). One-shot per stage, so cheap. Handle both
        # numpy and torch wavelet objects (build_wavelet returns numpy for
        # ricker, torch for from_npy).
        wav_t = state["wavelet"]
        is_torch = hasattr(wav_t, "detach")
        if is_torch:
            wav_np = wav_t.detach().cpu().numpy()
            wav_dtype = wav_np.dtype
            wav_device = wav_t.device
        else:
            wav_np = np.asarray(wav_t)
            wav_dtype = wav_np.dtype
            wav_device = None
        wav_filt = _bandpass_cpu(
            wav_np, lo=bp_spec.lo_hz, hi=bp_spec.hi_hz,
            dt=new_dt, order=bp_spec.order,
            axis=-1, padtype=bp_spec.padtype,
        ).astype(wav_dtype, copy=False)
        wav_filt = np.ascontiguousarray(wav_filt)
        if is_torch:
            state["wavelet"] = torch.from_numpy(wav_filt).to(wav_device)
        else:
            state["wavelet"] = wav_filt
        if dist_info.is_root:
            peak = float(np.abs(wav_filt).max())
            print(f"[stage {stage_idx}] wavelet bandpass "
                  f"{bp_spec.lo_hz}-{bp_spec.hi_hz} Hz applied (peak now {peak:.3f}) "
                  f"-> syn will be naturally band-limited; syn filter SKIPPED")

    # ---- Obs rebuild from pristine -----------------------------------------
    # Apply: receiver-mask -> time resample -> trim/pad -> optional bandpass.
    # Pristine obs is the snapshot at the END of step 5/5b (post-data_plan),
    # before any stage modification.
    pristine_obs = state["pristine_obs_np"]
    pristine_dt = state["pristine_dt"]
    receiver_axis_pristine = state["pristine_obs_receiver_axis"]
    time_axis_pristine = state["pristine_obs_time_axis"]

    # Derive this stage's obs from the cached pristine obs, in three steps:
    #   1) receiver mask (per-shot keep subset, or a uniform receiver_keep_idx)
    #   2) time-resample pristine_dt -> new_dt (no-op when dt is unchanged)
    #   3) trim/pad to new_nt
    # Deterministic + identical on every rank, so each rank just recomputes it
    # (the values match bit-for-bit). Returns a contiguous float32 copy that
    # downstream is free to upload to device / bandpass / restack.
    def _prep_obs_from_pristine():
        _obs = pristine_obs
        _keep_ps = state.get("per_shot_keep_idx")
        if _keep_ps is not None:
            # Per-shot dedupe: each shot keeps a different receiver subset.
            # ``np.take`` keeps the receiver axis in place (axis -2 of the
            # canonical 4-D ``(nshots, nt, nrec, nchan)``).
            if _obs.ndim != 4:
                raise NotImplementedError(
                    f"per-shot keep on obs.ndim={_obs.ndim} not implemented "
                    "(expected canonical 4-D (nshots, nt, nrec, nchan))"
                )
            _nshots, _nt_p, _nrec_p, _nchan = _obs.shape
            _new = np.empty((_nshots, _nt_p, _keep_ps.shape[1], _nchan), dtype=_obs.dtype)
            for s in range(_nshots):
                _new[s] = np.take(_obs[s], _keep_ps[s], axis=1)
            _obs = _new
        else:
            _keep = state.get("receiver_keep_idx")
            if _keep is not None and _keep.size != _obs.shape[receiver_axis_pristine]:
                _slicer = [slice(None)] * _obs.ndim
                _slicer[receiver_axis_pristine] = _keep
                _obs = _obs[tuple(_slicer)]
        _obs = _resample_obs_time(_obs, pristine_dt, new_dt, time_axis=time_axis_pristine)
        _obs = _trim_or_pad_time(_obs, new_nt, time_axis=time_axis_pristine)
        return np.ascontiguousarray(_obs.astype(np.float32, copy=False))

    obs_np = _prep_obs_from_pristine()

    # The previous stage's obs is dead from here on (this stage's is rebuilt
    # from pristine, and nothing reads state["obs"] before it is reassigned
    # below). Drop it before the new one is uploaded, so the GPU never holds
    # both, and hand its memory back so the headroom check below sees it.
    _prev_obs = state.pop("obs", None)
    if isinstance(_prev_obs, torch.Tensor) and _prev_obs.is_cuda:
        del _prev_obs
        torch.cuda.empty_cache()
    else:
        del _prev_obs

    # Upload obs to the solver device. The per-iter loop does
    # ``obs[chunk].to(dev)`` which becomes a free no-op once obs is on
    # dev, and saves the repeated H2D copy every iteration.
    obs_np = np.ascontiguousarray(obs_np.astype(np.float32, copy=False))
    obs_t = torch.from_numpy(obs_np)
    on_cuda = _is_cuda_dev(dev)
    if on_cuda:
        # Keep the full obs on GPU only when it fits with headroom. Large
        # field-data obs at fine stages (tens of GB) OOMs the device and can
        # exceed a V100's 32 GB; leave it on CPU and let the per-iter
        # ``obs[chunk].to(dev)`` stream each chunk on demand. The 6 GiB
        # headroom also covers the stage bandpass below, whose transient
        # memory _bandpass_obs_by_shot_chunks bounds by its chunk size.
        _free_b, _ = torch.cuda.mem_get_info(dev)
        if obs_t.nbytes < _free_b - (6 << 30) and os.environ.get("SWEEP_OBS_FORCE_CPU") != "1":
            obs_t = obs_t.to(dev, non_blocking=False)
        elif getattr(dist_info, "is_root", True):
            print(f"[stage {stage_idx}] obs {obs_t.nbytes / (1 << 30):.1f} GB "
                  f"> GPU headroom (free {_free_b / (1 << 30):.1f} GB) — kept on "
                  f"CPU, streaming per-chunk to device", flush=True)

    # 4) bandpass (per-stage; uses the new dt, so it's correctly normalised)
    #
    # GPU path (default when obs is on CUDA): reuse the canonical FFT
    # zero-phase Butterworth from sweepx.tasks.preproc — same implementation
    # used by ``_run_fwi_plan_streaming``'s per-iter encoded supershot path.
    # ~30× faster than scipy ``sosfiltfilt`` on Marmousi-scale obs
    # (1 GB float32), drops stage-entry time from ~10 s to <0.5 s.
    # CPU fallback: scipy ``sosfiltfilt`` with the configured padtype.
    if stage.bandpass is not None:
        if obs_t.is_cuda:   # obs may be CPU-resident for large field data (see upload above)
            obs_t = _bandpass_obs_by_shot_chunks(
                obs_t, lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                dt=new_dt, order=stage.bandpass.order,
                time_axis=time_axis_pristine,
            )
            _flavor = "GPU FFT (sweepx.tasks.preproc.bandpass_torch)"
        else:
            obs_np2 = _bandpass_cpu(
                obs_t.cpu().numpy(),
                lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                dt=new_dt, order=stage.bandpass.order,
                axis=time_axis_pristine,
                padtype=stage.bandpass.padtype,
            )
            obs_t = torch.from_numpy(np.ascontiguousarray(
                obs_np2.astype(np.float32, copy=False)
            ))
            _flavor = f"scipy sosfiltfilt (padtype={stage.bandpass.padtype or 'none'})"
        if dist_info.is_root:
            print(f"[stage {stage_idx}] bandpass {stage.bandpass.lo_hz}-{stage.bandpass.hi_hz} Hz "
                  f"order={stage.bandpass.order} ({_flavor})")
    state["obs"] = obs_t

    # ---- Source dedupe + obs stacking --------------------------------------
    # At coarse dh, the source spacing (typically 25 m for Viking-class
    # streamer surveys) is smaller than dh, so multiple shots round to the
    # same source grid cell. Group shots by source cell, stack (mean) their
    # obs traces. This:
    #   - reduces redundant forward modeling (1 syn per cell vs many)
    #   - cleans the gradient (1 stacked obs vs 1 syn per cell — no more
    #     "3 obs vs 1 syn" redundancy on the source side)
    #   - matches what a properly-deduped acquisition would naturally do
    # Per-stage (cheap: rerun from pristine each stage), so finer stages
    # automatically recover all shots if their source spacing >= dh.
    if state.get("dedupe_grid_snap", False):  # re-run every stage (matches the always-re-snapped geometry + re-derived obs)
        src_arr = state["sources"]
        keys = [(int(src_arr[s, 0]), int(src_arr[s, 1])) for s in range(src_arr.shape[0])]
        groups: dict[tuple[int, int], list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        if len(groups) < src_arr.shape[0]:
            sorted_keys = sorted(groups.keys())
            n_unique_src = len(sorted_keys)
            new_sources = np.zeros((n_unique_src, 2), dtype=np.int64)
            rec_curr = state["receivers"]
            new_receivers = np.zeros((n_unique_src, rec_curr.shape[1], 2), dtype=np.int64)
            obs_curr = state["obs"]  # tensor (n_pristine, n_common_rec, nt)
            new_obs_shape = (n_unique_src, *obs_curr.shape[1:])
            new_obs = torch.zeros(new_obs_shape, dtype=obs_curr.dtype)
            group_sizes = []
            for g, k in enumerate(sorted_keys):
                indices = np.asarray(groups[k], dtype=np.int64)
                group_sizes.append(int(indices.size))
                # Representative: first shot in the group (deterministic)
                new_sources[g] = src_arr[indices[0]]
                new_receivers[g] = rec_curr[indices[0]]
                # Stack obs: mean across the group's shots. obs_curr is
                # already receiver-deduped, so all shots in the group share
                # the same receiver layout (streamer geometry → constant
                # relative offsets) and can be averaged element-wise.
                if indices.size == 1:
                    new_obs[g] = obs_curr[int(indices[0])]
                else:
                    new_obs[g] = obs_curr[indices.tolist()].mean(dim=0)
            # Also dedupe per_shot_keep_idx if it's per-pristine-shot.
            per_shot_keep = state.get("per_shot_keep_idx")
            if per_shot_keep is not None:
                state["per_shot_keep_idx"] = np.stack(
                    [per_shot_keep[groups[k][0]] for k in sorted_keys]
                )
            state["sources"] = new_sources
            state["receivers"] = new_receivers
            state["obs"] = new_obs
            state["nshots"] = int(new_sources.shape[0])
            if dist_info.is_root:
                avg_grp = src_arr.shape[0] / n_unique_src
                print(f"[stage {stage_idx}] source dedupe: "
                      f"{src_arr.shape[0]} shots -> {n_unique_src} unique src cells "
                      f"(avg group size {avg_grp:.1f}, max {max(group_sizes)})")
        state["_source_dedup_at_dh"] = float(new_dh)

    # ---- Re-init optimizer (Adam state is shape-bound) ---------------------
    # Reparam mode: optimizer state is on network params (shape-invariant),
    # so DO NOT rebuild on grid change — preserving Adam moments across
    # stages is the main multi-scale benefit of network reparameterization.
    # However, an explicit ``stage.optimizer_reset=True`` forces a rebuild
    # even in reparam mode (e.g. when switching regimes drastically and
    # the old momentum has gone stale). Only on entering the stage: a run
    # resumed inside it continues with the checkpointed moments.
    force_reset = bool(getattr(stage, "optimizer_reset", False)) and not resumed
    if "optimizer" not in state:
        # Cold-init (e.g. resume-from path). Build appropriate optimizer.
        if state.get("reparam_net") is not None:
            state["optimizer"] = _build_reparam_optimizer(
                spec.optimizer, state["reparam_net"].parameters(),
                float(spec.reparam.lr),
            )
        else:
            state["optimizer"] = _build_optimizer(
                spec.optimizer, state["inv_by_name"], required_names,
            )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])
    elif force_reset:
        if state.get("reparam_net") is not None:
            state["optimizer"] = _build_reparam_optimizer(
                spec.optimizer, state["reparam_net"].parameters(),
                float(spec.reparam.lr),
            )
        else:
            state["optimizer"] = _build_optimizer(
                spec.optimizer, state["inv_by_name"], required_names,
            )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])
        if dist_info.is_root:
            print(f"[stage {stage_idx}] optimizer_reset=True — Adam moments dropped")
    elif grid_changed and state.get("reparam_net") is None:
        state["optimizer"] = _build_optimizer(
            spec.optimizer, state["inv_by_name"], required_names,
        )
        state["initial_lrs"] = _remember_initial_lrs(state["optimizer"])

    # ``lr_scale`` scales the optimizer's per-group initial lr; for the
    # reparam path the optimizer holds network params at ``spec.reparam.lr``,
    # so ``inr_lr_scale`` multiplies on top of that, on every group the
    # network has (grown hash levels included). Without reparam FWISpec
    # rejects a set ``inr_lr_scale``: there is no network to scale.
    if state.get("reparam_net") is not None:
        effective_scale = float(stage.lr_scale) * float(
            getattr(stage, "inr_lr_scale", 1.0)
        )
    else:
        effective_scale = float(stage.lr_scale)
    _apply_stage_lr_scale(state["optimizer"], state["initial_lrs"], effective_scale)

    # ---- Batch size override ----------------------------------------------
    state["batchsize"] = (
        int(stage.batch_size) if stage.batch_size is not None
        else int(spec.batchsize)
    )

    # ---- Commit state ------------------------------------------------------
    state["dh"] = new_dh
    state["dt"] = new_dt
    state["nt"] = new_nt
    if dist_info.is_root:
        print(f"[stage {stage_idx}] obs shape: {tuple(state['obs'].shape)}, "
              f"nshots={state['nshots']}, batchsize={state['batchsize']}")


# ---------- the TaskRunner ------------------------------------------------


def _resample_obs_time(obs_np: "np.ndarray", dt_old: float, dt_new: float,
                       *, time_axis: int = -1) -> "np.ndarray":
    """Resample obs along the time axis via sweepx.tasks.preproc.resample.resample_time."""
    if abs(dt_old - dt_new) < 1e-12:
        return obs_np
    from sweepx.tasks.preproc.resample import resample_time
    return resample_time(obs_np, dt_old, dt_new, axis=time_axis)


def _resample_obs_to_solver_dt(obs_t, dt_segy: float, dt_solver: float,
                               nt_solver: int):
    """Resample a torch obs tensor along the last axis to solver dt + length.

    Mirrors the legacy 3-D CRG FWI runner's per-iter obs prep: drop to
    numpy, run scipy.signal.resample_poly via sweepx.tasks.preproc, truncate /
    zero-pad to nt_solver, and ship back to the original device.
    """
    if abs(dt_segy - dt_solver) < 1.0e-12 and obs_t.shape[-1] == nt_solver:
        return obs_t
    import torch
    from sweepx.tasks.preproc.resample import resample_time

    arr = obs_t.detach().cpu().numpy()
    arr = resample_time(arr, dt_segy, dt_solver, axis=-1)
    cur_nt = arr.shape[-1]
    if cur_nt > nt_solver:
        arr = arr[..., :nt_solver]
    elif cur_nt < nt_solver:
        pad = [(0, 0)] * arr.ndim
        pad[-1] = (0, nt_solver - cur_nt)
        arr = np.pad(arr, pad)
    return torch.as_tensor(arr, dtype=obs_t.dtype, device=obs_t.device)


# Obs bytes one stage-entry bandpass call may span. The FFT bandpass
# (rfft -> x gain -> irfft) needs about 7x its input in transient GPU memory,
# so a 512 MiB chunk stays well inside the 6 GiB headroom the obs upload
# reserves, whatever the obs size. A fixed shot count did not: 64 shots of the
# Marmousi examples' obs are 3.2 GiB, ~23 GiB transient, an OOM on a 32 GB card.
_OBS_BANDPASS_CHUNK_BYTES = 512 << 20


def _bandpass_obs_by_shot_chunks(obs_t: "torch.Tensor", *, lo: float, hi: float,
                                 dt: float, order: int, time_axis: int,
                                 chunk_bytes: int = _OBS_BANDPASS_CHUNK_BYTES,
                                 ) -> "torch.Tensor":
    """Zero-phase FFT bandpass of obs along ``time_axis``, a few shots at a time.

    Each shot (dim 0) is filtered on its own time axis, so filtering groups of
    shots gives the same result as filtering all of them in one call; grouping
    only bounds the transient memory (and keeps each cuFFT plan under its
    2^31-element limit). A group holds as many shots as fit in ``chunk_bytes``
    (at least one). The filtered groups are written back into ``obs_t``, which
    is returned. With time on dim 0 there is no shot axis to split, and the
    whole array is filtered in one call.
    """
    kw = dict(lo=lo, hi=hi, dt=dt, order=order, axis=time_axis)
    n = int(obs_t.shape[0])
    with torch.no_grad():
        if time_axis % obs_t.ndim == 0 or n == 0:
            return _bandpass_torch_fft(obs_t, **kw)
        shot_bytes = max(1, obs_t[0].numel() * obs_t.element_size())
        step = max(1, int(chunk_bytes) // shot_bytes)
        for b0 in range(0, n, step):
            b1 = min(b0 + step, n)
            obs_t[b0:b1] = _bandpass_torch_fft(obs_t[b0:b1].contiguous(), **kw)
    return obs_t


def _bandpass_syn_torch(syn: "torch.Tensor", lo: float, hi: float, dt: float,
                        *, order: int) -> "torch.Tensor":
    """Differentiable bandpass on a synthetic torch tensor.

    Thin wrapper around :func:`sweepx.tasks.preproc.filter.bandpass_torch` —
    the same canonical zero-phase GPU FFT Butterworth used by the obs
    stage-entry filter and by the plan-streaming encoded FWI path. Going
    through one impl across syn, obs, and wavelet ensures the filter
    response is identical and avoids the latent NaN risk of
    ``torchaudio.functional.filtfilt`` without padding on narrow bands.

    Both sweep backends (eager + c, after geophyai 21041c5) emit syn in
    canonical 4-D ``(n, nt, nrec, nchan)`` with time at axis 1, so this
    helper only accepts 4-D and raises on anything else. The historical
    3-D fallback ``(nt, nrec, 1)`` (time at axis 0) is removed because
    (a) sweep no longer produces it, and (b) the layout it assumed
    disagreed with ``sweepx.loss.to_canonical``'s 3-D convention
    ``(nshots, nt, nrec)`` (time at axis 1), so any 3-D input would be
    silently filtered on the wrong axis. Manual 3-D inputs now fail
    loud here instead.
    """
    if syn.ndim != 4:
        raise ValueError(
            f"_bandpass_syn_torch: expected canonical 4-D syn "
            f"(n, nt, nrec, nchan); got {syn.ndim}-D shape {tuple(syn.shape)}. "
            "sweep backends always return 4-D — wrap or unsqueeze upstream."
        )
    return _bandpass_torch_fft(syn, lo=float(lo), hi=float(hi),
                               dt=float(dt), order=int(order), axis=1)


def _trim_or_pad_time(obs_np: "np.ndarray", target_nt: int, time_axis: int = -1) -> "np.ndarray":
    n = obs_np.shape[time_axis]
    if n == target_nt:
        return obs_np
    if n > target_nt:
        slicer = [slice(None)] * obs_np.ndim
        slicer[time_axis] = slice(0, target_nt)
        return obs_np[tuple(slicer)]
    pad_width = [(0, 0)] * obs_np.ndim
    pad_width[time_axis] = (0, target_nt - n)
    return np.pad(obs_np, pad_width)
