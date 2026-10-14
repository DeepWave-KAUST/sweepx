"""Frequency-coefficient (DTFT shard) FWI task runner (mixin).

Dispatched when ``source_encoding.mode == "frequency_selection"``. Obs is
a set of pre-extracted per-node DTFT coefficient shards, so this path
touches NO SEG-Y and needs NO wavelet: the GCN coherence misfit is
invariant to any per-node complex scale, which absorbs the source
spectrum, excitation delay and sensor coupling.

Note the word "encoding" means something different here than in
:mod:`sweep_tasks.tasks.fwi_plan_streaming`: that path encodes in TIME
with ±1 signs on a supershot, this one divides the FREQUENCY axis into a
comb. See ``_run_fwi``'s dispatch table.

Extracted from runner.py.
"""
from pathlib import Path
import os
from sweep_tasks.schemas import FWISpec
from sweep_tasks._helpers.dd import (
    _dd_backward_tile,
    _dd_backward_tile_multi,
    _dd_config,
    _dd_render_tile,
    _dd_render_tile_multi,
    _dd_tile_bounds,
    _dd_wrap,
)
from sweep_tasks._helpers.illumination import (
    _accumulate_illumination,
    _apply_illumination_precond,
)
from sweep_tasks._helpers.model import _solver_models
from sweep_tasks._helpers.plan_apply import _normalize_fwi_init_models
from sweep_tasks._helpers.reparam import (
    _advance_hash_schedule,
    _build_reparam_net,
    _has_hash_schedule,
)
from sweep_tasks._helpers.solver_build import _build_solver
from sweep_tasks._helpers.stage_grid import (
    _resample_to_grid,
    _seabed_to_grid,
    _set_physical_inr_frame,
    _stage_grid_shape,
)
from sweep_tasks._helpers.stages import _normalise_stage_list
from sweep_tasks._helpers.model import _model_array
from sweep_tasks._helpers.util import (
    _apply_seed,
    _resolve_device,
)


def backend_with_tail_steps(backend, fspec, dd_on: bool, dt: float | None = None):
    """Per-stage backend spec with truncated-backward boundary saving.

    ``fspec.bwd_tail_margin`` set -> return a deep copy of ``backend`` whose
    boundary config carries ``tail_steps = probe_samples + bwd_tail_margin``
    (per stage: every rung has its own probe window). Unset -> ``backend``
    unchanged, bit-identical behaviour.

    Guards raise HERE, with the YAML vocabulary, rather than letting sweep
    fail deep inside the propagator: truncation needs the c-backend
    boundary-saving adjoint (no ckpt, no eager). DD is supported — tail_steps
    rides the boundary config into every tile — with equation coverage
    guarded by the core (Acoustic 2D/3D).
    """
    margin = getattr(fspec, "bwd_tail_margin", None)
    margin_s = getattr(fspec, "bwd_tail_margin_s", None)
    if margin is None and margin_s is None:
        return backend
    if margin is None:
        # Seconds -> this stage's steps: the margin is a physical decay time,
        # so it must rescale with per-stage dt in a multi-rate cascade.
        if dt is None:
            raise ValueError("bwd_tail_margin_s needs the stage dt")
        import math
        margin = int(math.ceil(float(margin_s) / float(dt)))
    # DD passes straight through: every tile prop inherits tail_steps from
    # the boundary config (same route as storage/storage_dtype), which also
    # makes the reverse-loop stop step identical on every rank — the core
    # enforces its own capability guards (Acoustic 2D/3D only; a core
    # predating DD-tail support raises there, not here).
    if backend.impl != "c":
        raise ValueError(
            "frequency.bwd_tail_margin requires backend.impl='c' "
            f"(got '{backend.impl}'): the truncation lives in the c-backend "
            "boundary-saving adjoint")
    if backend.use_ckpt:
        raise ValueError(
            "frequency.bwd_tail_margin is incompatible with "
            "backend.use_ckpt=true — checkpointing replays forward segments "
            "and cannot skip the head of the record; use "
            "cuda_options.memory.strategy='boundary'")
    from sweep_tasks.schemas import (BoundaryOptionsModel, CUDAOptionsModel,
                                     MemoryOptionsModel)
    tail = int(fspec.probe_samples) + int(margin)
    out = backend.model_copy(deep=True)
    if out.cuda_options is None:
        out.cuda_options = CUDAOptionsModel(
            memory=MemoryOptionsModel(strategy="boundary",
                                      boundary=BoundaryOptionsModel(
                                          tail_steps=tail)))
        return out
    mem = out.cuda_options.memory
    if mem is None:
        out.cuda_options.memory = MemoryOptionsModel(
            strategy="boundary",
            boundary=BoundaryOptionsModel(tail_steps=tail))
        return out
    if mem.strategy != "boundary":
        raise ValueError(
            "frequency.bwd_tail_margin requires "
            "cuda_options.memory.strategy='boundary' "
            f"(got '{mem.strategy}')")
    if mem.boundary is None:
        mem.boundary = BoundaryOptionsModel(tail_steps=tail)
    else:
        mem.boundary.tail_steps = tail
    return out


def backend_with_boundary_storage(backend, storage: str):
    """Copy of ``backend`` with the boundary strips staged on ``storage``.

    Used for the out-of-memory fallback: the card is the right home while the
    strips fit, since they are written and read every step, but a stage that
    runs slower on the host beats one that dies.
    """
    out = backend.model_copy(deep=True)
    mem = getattr(getattr(out, "cuda_options", None), "memory", None)
    if mem is None or mem.boundary is None:
        raise ValueError(
            "boundary storage fallback needs cuda_options.memory.boundary "
            "(set memory.strategy='boundary')")
    mem.boundary.storage = storage
    return out


def backend_with_stage_boundary(backend, ov):
    """Copy of ``backend`` with a stage's ``boundary:`` block merged in.

    Only the fields the stage EXPLICITLY set are copied (``model_fields_set``),
    so an override of ``storage`` alone keeps the global dtype / interval /
    ring buffers instead of silently reverting them to schema defaults -- the
    same trap ``stage_freq_override_warnings`` exists to flag for ``frequency``.
    ``ov`` unset -> ``backend`` unchanged, bit-identical behaviour.
    """
    if ov is None:
        return backend
    fields = getattr(ov, "model_fields_set", None) or set()
    if not fields:
        return backend
    out = backend.model_copy(deep=True)
    mem = getattr(getattr(out, "cuda_options", None), "memory", None)
    if mem is None or mem.boundary is None:
        raise ValueError(
            "stage 'boundary:' override needs cuda_options.memory.boundary "
            "(set memory.strategy='boundary')")
    # An override that names EVERY field inherits nothing. Nobody writes that
    # by hand -- it is the signature of a config derived from a pre-2026-09
    # ``config_resolved.yaml``, which dumped overrides with all defaults
    # expanded. Say which values that silently replaces, because the damage is
    # invisible whenever the expanded default happens to be a legal value: an
    # inherited ``storage_dtype: int8`` came back as ``fp32`` and the boundary
    # buffer grew 4x.
    if fields >= set(type(ov).model_fields):
        stolen = [f"{k}: {getattr(mem.boundary, k)!r} -> {getattr(ov, k)!r}"
                  for k in sorted(fields)
                  if getattr(ov, k) != getattr(mem.boundary, k)]
        if stolen:
            print("[freqsel] WARNING stage 'boundary:' override names every "
                  "field, so it inherits nothing from the global block. If it "
                  "came from a config_resolved.yaml, these are schema defaults "
                  "overriding your settings: " + "; ".join(stolen), flush=True)
    for k in sorted(fields):
        setattr(mem.boundary, k, getattr(ov, k))
    # A stage that flips storage to the card inherits the global block's
    # staging knobs, and those are cpu/disk-only -- the core rejects the pair
    # rather than ignoring them. Say which knobs and how to clear them here,
    # where the YAML vocabulary still exists, instead of letting it surface as
    # a bare ValueError from deep inside BoundaryOptions.
    if mem.boundary.storage == "gpu":
        stale = [k for k in ("transfer_interval", "pinned_memory",
                             "ring_buffers", "disk_dir")
                 if getattr(mem.boundary, k, None) is not None]
        if stale:
            raise ValueError(
                f"stage boundary override sets storage='gpu' but inherits "
                f"{stale} from the global block; those are only valid for "
                f"storage='cpu'/'disk'. Set them to null in the SAME stage "
                f"'boundary:' block (an override only replaces the fields it "
                f"names).")
    return out


def stage_freq_override_warnings(fspec_global, fspec_stage, si) -> list[str]:
    """Warn when a stage's ``frequency:`` block silently drops a global setting.

    A stage's sub-spec REPLACES ``source_encoding.frequency`` wholesale — it is
    not merged — so any knob the global set and the stage omitted reverts to the
    schema default. That is invisible at runtime and changes what the run does:
    ``random_batch`` omitted turns random node batching back into deterministic
    pool rotation, and ``steady_samples`` omitted can drop a long ring-up to the
    2500-sample default, leaving a transient inside the analysis window.

    Every band needs its own block (different shard and comb), so this is the
    normal case, not a corner one — hence a warning rather than an error.
    """
    if fspec_global is None or fspec_stage is None or fspec_stage is fspec_global:
        return []
    set_global = getattr(fspec_global, "model_fields_set", set())
    set_stage = getattr(fspec_stage, "model_fields_set", set())
    out = []
    for name in sorted(set_global - set_stage):
        gv = getattr(fspec_global, name, None)
        sv = getattr(fspec_stage, name, None)
        if gv != sv:
            out.append(
                f"[freqsel] WARNING stage {si}: `frequency.{name}` is not "
                f"inherited — the global set {name}={gv!r} but this stage's "
                f"`frequency:` block replaces it wholesale, so {name}={sv!r} "
                f"(the schema default) is in effect. Repeat it in the stage "
                f"block if you meant {gv!r}.")
    return out


def _freqsel_adam(params, lr):
    """The optimizer this path runs: Adam at torch's default eps and betas.

    Only the lr comes from the config. FWISpec rejects another optimizer kind,
    eps or betas, and any scheduler, rather than let them be ignored.
    """
    import torch

    return torch.optim.Adam(params, lr=float(lr), eps=1e-8, betas=(0.9, 0.999))


def _release_card_between_stages(si: int, rank: int) -> None:
    """Hand the card back to the driver before the next stage builds its mesh.

    A finished stage's blocks go to PyTorch's caching allocator, not to the
    driver. The next stage constructs a fresh ``ModelParallel``, and
    ``ModelParallelMesh`` opens ``1 + py*px`` new process groups whose NCCL
    communicators ``cudaMalloc`` OUTSIDE that allocator. With the cache still
    owning the card, NCCL gets nothing and the run dies on the stage's first
    collective with a bare ``ncclUnhandledCudaError: Call to CUDA function
    failed / Cuda failure 2 'out of memory'`` -- no mention of which
    allocation failed, and long after the memory that matters was freed.

    Set ``SWEEP_FREQSEL_STAGE_RELEASE=0`` to skip the release; that is the
    negative arm of the reproduction, not something to run in production.
    """
    import gc

    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return
    release = os.environ.get("SWEEP_FREQSEL_STAGE_RELEASE", "1") != "0"
    free_before, total = torch.cuda.mem_get_info()
    if release:
        torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info()
    if rank == 0:
        print(f"[freqsel] stage {si} gap: gpu free "
              f"{free_before / 2 ** 30:.1f} -> {free_after / 2 ** 30:.1f} / "
              f"{total / 2 ** 30:.1f} GiB "
              f"(release={'on' if release else 'OFF'})", flush=True)


class FreqselRunnerMixin:
    @staticmethod
    def _restore_optimizer_state(spec, optimizer, rank):
        """Restore Adam's first/second moments.

        Restoring only the network weights is not equivalent to a continuous
        run, so a missing file RAISES rather than silently leaving a zero
        state: a silent fall back reads as "resumed, and the mismatch is
        something else". The lr the state was saved at comes back with it;
        the stage entry replaces it with this run's base lr x stage scale.
        """
        import torch
        from pathlib import Path
        rp = getattr(spec, "reparam", None)
        iof = getattr(rp, "init_optimizer_from", None) if rp is not None else None
        if not iof:
            return
        if str(iof) == "auto":
            _if = getattr(rp, "init_from", None)
            if not _if:
                raise ValueError(
                    "reparam.init_optimizer_from='auto' needs reparam.init_from "
                    "set as well -- it is derived from that path")
            op = Path(str(_if).replace("reparam_net_iter", "optim_iter"))
            if op == Path(str(_if)):
                op = Path(str(_if)).with_name("optim.pt")
        else:
            op = Path(str(iof))
        if not op.exists():
            raise FileNotFoundError(
                f"reparam.init_optimizer_from -> {op} does not exist. Without "
                "Adam's moments a resume is not equivalent to a continuous run "
                "(the first dozen steps take a completely different step "
                "scale), so this refuses to continue silently. Either provide "
                "the file or set init_optimizer_from to null.")
        sd = torch.load(str(op), map_location="cpu", weights_only=False)
        os_ = sd["optimizer"] if isinstance(sd, dict) and "optimizer" in sd else sd
        optimizer.load_state_dict(os_)
        st = os_.get("state", {})
        step = next((float(v["step"]) for v in st.values() if "step" in v), None)
        if rank == 0:
            print(f"[freqsel] init_optimizer_from: Adam state <- {op} "
                  f"(tensors={len(st)}, step={step}, meta="
                  f"{sd.get('meta') if isinstance(sd, dict) else None})", flush=True)

    def _freqsel_run_stage(
        self, spec, stage, si, fspec, dh, dt, dev,
        dd_on, dd_py, dd_px, rank, world, task_dir,
        use_reparam, net, optimizer, chunk_rows,
        vp0_native, native_dh, illum_spec, illum_on,
        losses, times, peaks, epoch_offset, total_epochs,
    ):
        """Run ONE frequency-selection stage on its own (dh, dt) grid.

        Builds this stage's comb / solver / targets / scheduler / loss, carries
        over the reparam network across stages (resampling its base to the new
        grid via ``update_base_velocity``; net params + Adam state kept) or the
        grid model (and its Adam, on the same grid without ``optimizer_reset``),
        runs ``stage.epochs`` iterations appending to ``losses/times/peaks``.
        Returns ``(net, optimizer, ny_, nx_, chk)``. Single-stage runs reduce
        to the original single-band behaviour bit-for-bit.
        """
        import time as _time

        import numpy as np
        import torch

        from sweep_tasks import freqsel as fsl

        comb = fsl.FrequencyComb(
            dt=float(dt), n_p=int(fspec.probe_samples),
            ks=np.arange(int(fspec.k_lo), int(fspec.k_hi) + 1))
        nt = int(fspec.steady_samples + fspec.slack_samples
                 + fspec.probe_samples)

        # ---- init model on THIS stage's grid (resample pristine to dh) ----
        # The stage grid samples the native model at i * dh from the native
        # origin and spans its whole extent -- the same frame the band's
        # shards are extracted on (round(x / dh)); see _helpers.stage_grid.
        native_grid = abs(float(dh) - float(native_dh)) < 1e-9
        if native_grid:
            vp0 = vp0_native
        else:
            new_shape = _stage_grid_shape(vp0_native.shape, native_dh, float(dh))
            vp0 = _resample_to_grid(vp0_native, native_dh, new_shape, float(dh))
        gshape = vp0.shape
        is2d = len(gshape) == 2
        if is2d:
            nz, nx_ = gshape
            ny_ = None
            nyp = None
            nxp = nx_                      # DD is 3-D only (guarded below)
            _pad = ((0, 0), (0, 0))
            _solver_shape = (nz, nxp)
        else:
            nz, ny_, nx_ = gshape
            nyp = -(-ny_ // dd_py) * dd_py if dd_on else ny_
            nxp = -(-nx_ // dd_px) * dd_px if dd_on else nx_
            _pad = ((0, 0), (0, nyp - ny_), (0, nxp - nx_))
            _solver_shape = (nz, nyp, nxp)
        # The grid the reparam network's coordinate frame is defined on: the
        # native grid as this runner pads it. Stage grids reuse that frame.
        if is2d:
            _frame_shape = tuple(vp0_native.shape)
        else:
            _nz0, _ny0, _nx0 = vp0_native.shape
            _frame_shape = (_nz0,
                            -(-_ny0 // dd_py) * dd_py if dd_on else _ny0,
                            -(-_nx0 // dd_px) * dd_px if dd_on else _nx0)
        vp0p = np.pad(vp0, _pad, mode="edge")
        frozen = vp0p == float(getattr(spec.reparam, "water_vp_m_s", 1500.0)
                               if spec.reparam else 1500.0)
        if not is2d:
            frozen[:, ny_:, :] = True
            frozen[:, :, nx_:] = True
        water_t = torch.tensor(frozen, device=dev)
        base_t = torch.tensor(vp0p, device=dev)

        # Multi-parameter reparam: prepare a base tensor per freed parameter
        # (vp is channel 0 = base_t; z / vs / … loaded from init_models by name),
        # each resampled to this stage's grid + padded like vp so the shared
        # MultiParamINR renders all channels on one grid.
        free_multi = bool(use_reparam and getattr(spec.reparam, "free_params", None))
        n_free = len(spec.reparam.free_params) if free_multi else 1
        param_bases = None
        if free_multi:
            def _prep_base(native_arr):
                if tuple(native_arr.shape) != tuple(vp0_native.shape):
                    raise ValueError(
                        f"reparam.free_params init model shape "
                        f"{tuple(native_arr.shape)} != vp init "
                        f"{tuple(vp0_native.shape)}: every channel must be on "
                        "the native grid (grid.dh)")
                if native_grid:
                    a0 = native_arr
                else:
                    a0 = _resample_to_grid(native_arr, native_dh, gshape, float(dh))
                a0p = np.pad(a0, _pad, mode="edge")
                return torch.tensor(a0p, device=dev)

            _refs = {m.name: m for m in _normalize_fwi_init_models(spec)}
            param_bases = []
            for _fp in spec.reparam.free_params:
                if _fp.name == "vp":
                    param_bases.append(base_t)
                    continue
                _ref = _refs.get(_fp.name)
                if _ref is None or getattr(_ref, "path", None) is None:
                    raise ValueError(
                        f"reparam.free_params '{_fp.name}' has no matching "
                        f"init_models entry with a path")
                param_bases.append(_prep_base(_model_array(_ref)))

        stage_backend = backend_with_tail_steps(spec.backend, fspec, dd_on,
                                                dt=float(dt))
        if rank == 0 and stage_backend is not spec.backend:
            _tl = stage_backend.cuda_options.memory.boundary.tail_steps
            print(f"[freqsel] s{si} truncated backward: boundary tail_steps="
                  f"{_tl} of nt={nt} (probe {fspec.probe_samples} + margin "
                  f"{_tl - fspec.probe_samples} steps; reverse depth "
                  f"{_tl - 1})", flush=True)
        def _make_solver(storage=None):
            """Build (and DD-wrap) the stage solver, optionally forcing storage.

            Precedence: the explicit argument (the out-of-memory fallback)
            beats SWEEP_FREQSEL_BOUNDARY_STORAGE (operator override of a stage
            that ran out of card) beats the stage's own ``boundary:`` block
            beats the top-level backend. The core fixes the boundary's home at
            construction, so this is the only door.
            """
            be = backend_with_stage_boundary(stage_backend,
                                             getattr(stage, "boundary", None))
            storage = storage or os.environ.get("SWEEP_FREQSEL_BOUNDARY_STORAGE") or None
            if storage is not None:
                be = backend_with_boundary_storage(be, storage)
            sv = _build_solver(spec.physics, be, _solver_shape, float(dh),
                               float(dt), nt, dev)
            if dd_on:
                from sweep.parallel import MeshTopology
                _mesh = MeshTopology(py=dd_py, px=dd_px, shot_groups=1,
                                     world_size=world, rank=rank)
                sv = _dd_wrap(sv, _mesh)
            return sv

        solver = _make_solver()
        # Boundary storage falls back GPU -> host on the first out-of-memory,
        # for the rest of the stage. The card is where this belongs when it
        # fits: the strips are written and read every step, and staging them
        # through the host costs a PCIe round trip each way. But the fine bands
        # do not fit -- a fine rung can exhaust an 80 GB card even split four ways, in
        # use -- and a stage that runs slower is worth more than one that dies.
        boundary_on_host = False

        illum_solver = getattr(solver, "prop", solver)
        if illum_on:
            try:
                illum_solver.compute_illumination = True
            except Exception:
                pass
            if rank == 0 and si == 0:
                print(f"[freqsel] illumination precond ON "
                      f"(exp={illum_spec.exponent}, eps={illum_spec.epsilon}, "
                      f"rel_eps={getattr(illum_spec, 'relative_epsilon', None)})",
                      flush=True)

        # ---- targets: field shards or one-shot synthesis ------------------
        if fspec.synthesize_from_true:
            if dd_on:
                raise NotImplementedError(
                    "synthesize_from_true is a single-device test path")
            vpt = np.load(fspec.true_model_path).astype(np.float32)
            if vpt.shape != tuple(vp0_native.shape):
                raise ValueError(
                    f"source_encoding.frequency.true_model_path: shape "
                    f"{vpt.shape} != init model {tuple(vp0_native.shape)}; the "
                    "true model must be on the native grid (grid.dh) so every "
                    "band samples the same physical model")
            if not native_grid:
                # Sample the true model on this stage's grid, exactly like the
                # init, so a multi-stage ladder synthesizes each band's obs on
                # the grid it inverts on.
                vpt = _resample_to_grid(vpt, native_dh, gshape, float(dh))
            # The synthetic acquisition is laid out in NATIVE cells and placed
            # on this stage's grid at round(x / dh), so every band records the
            # same physical nodes and receivers (on the native grid this is
            # the layout itself).
            n_nodes = int(fspec.synth_n_nodes)
            _nx0 = int(vp0_native.shape[-1])
            _r = float(native_dh) / float(dh)
            sx = np.rint(np.linspace(8, _nx0 - 9, n_nodes).astype(np.int64)
                         * _r).astype(np.int64)
            rx = np.unique(np.rint(np.arange(4, _nx0 - 4, int(fspec.synth_rec_stride))
                                   * _r).astype(np.int64))
            node_z = int(np.rint(int(fspec.synth_node_z) * _r))
            if is2d:
                nodes = np.stack([sx, np.full(n_nodes, node_z, np.int64)], -1)
                recs = np.stack([rx, np.zeros(len(rx), np.int64)], -1)
            else:
                _ym = int(np.rint((int(vp0_native.shape[1]) // 2) * _r))
                nodes = np.stack([sx, np.full(n_nodes, _ym, np.int64),
                                  np.full(n_nodes, node_z, np.int64)], -1)
                recs = np.stack([rx, np.full(len(rx), _ym, np.int64),
                                 np.zeros(len(rx), np.int64)], -1)
            shard = str(task_dir / f"freqsel_synth_obs_s{si}.npz")
            t = np.arange(nt, dtype=np.float64) * comb.dt
            f0 = 1.5 * float(np.mean(comb.freqs))
            a = np.pi * f0 * (t - 0.25)
            ricker = ((1 - 2 * a * a) * np.exp(-a * a)).astype(np.float32)
            vpt_t = torch.tensor(
                np.pad(vpt, _pad, mode="edge"),
                device=dev)
            # True obs uses the correct multi-parameter physics: VRZ -> [vp, z]
            # (Gardner-coupled truth for this synthetic test), acoustic -> [vp].
            _synth_models = _solver_models(vpt_t, spec)
            fsl.synthesize_shard(
                shard, solver, vpt_t, nodes, recs, comb, ricker, dev,
                verbose=rank == 0, models=_synth_models)
            shards_glob = shard
        else:
            shards_glob = fspec.coeff_shards
        targets = fsl.FreqSelTargets(shards_glob, comb, ny_ or 1,
                                     verbose=rank == 0)
        # node-clamp: lift freqsel node sources off the seabed sediment cell into
        # the water cell just above (per-node min), from the init vp's seabed.
        _lift = getattr(fspec, "lift_source_to_water_vp", None)
        if _lift is not None and spec.init_model is not None:
            # This stage's init model (the native one on the native grid): the
            # shard's node indices are on this grid, so the seabed must be too.
            _iv = vp0                                                # (nz, ny, nx)
            _nw = np.abs(_iv - float(_lift)) > 1.0
            _sb = np.where(_nw.any(0), _nw.argmax(0), _iv.shape[0])    # (ny, nx)
            _ng = targets.node_grid
            _yy = np.clip(_ng[:, 1].astype(int), 0, _sb.shape[0] - 1)
            _xx = np.clip(_ng[:, 0].astype(int), 0, _sb.shape[1] - 1)
            _sbn = _sb[_yy, _xx]
            _nl = int((_ng[:, 2] > _sbn - 1).sum())
            _ng[:, 2] = np.maximum(np.minimum(_ng[:, 2], _sbn - 1), 0)
            if rank == 0:
                print(f"[freqsel] lifted {_nl}/{len(_ng)} sources to water "
                      f"(vp!={_lift:g}); seabed_iz [{int(_sb.min())},{int(_sb.max())}]", flush=True)
        sched = fsl.PoolScheduler(targets.node_grid, int(fspec.n_pools),
                                  comb.n_bins, seed=int(spec.seed) + 17,
                                  random_batch=getattr(fspec, "random_batch", None))
        rec_table = targets.union_xyz[None]
        if rank == 0:
            print(f"[freqsel] stage {si}: {targets.n_nodes} nodes, "
                  f"{targets.n_items} items, {targets.n_union} union cells, "
                  f"{sched.selection}, comb {comb.freqs[0]:.4f}-"
                  f"{comb.freqs[-1]:.4f} Hz, nt={nt}, dh={dh}, dt={dt}",
                  flush=True)

        # ---- optional seabed-based water pin (geologically-correct) --------
        # The seabed_depth npz is a 2-D (ny, nx) bathymetry (meters from z=0)
        # on the model's lateral extent at ANY resolution; resample it to THIS
        # stage's (ny_, nx_) — same physical window — then broadcast
        # z*dh<seabed to a 3-D water mask. Preferred over the
        # vp==water_vp_m_s heuristic (which only catches voxels stamped to
        # exactly the water velocity — e.g. the lateral edge-fill — and misses
        # a water column filled with extrapolated rock).
        _seabed_override = None
        if (use_reparam and spec.reparam is not None
                and bool(getattr(spec.reparam, "mask_water_layer", False))
                and getattr(spec.reparam, "seabed_depth_path", None) is not None):
            from sweep_tasks.bathymetry import (
                load_seabed_depth_npz, water_mask_from_seabed_depth)
            _sd = load_seabed_depth_npz(spec.reparam.seabed_depth_path)
            if _sd.ndim != 2:
                raise ValueError(
                    f"reparam.seabed_depth_path: expected 2-D (ny, nx) "
                    f"seabed_depth for 3-D FWI; got shape {tuple(_sd.shape)}")
            # Registration guard: the npz must cover the SAME lateral extent as
            # the model (it is resampled to (ny_, nx_) below, so any resolution
            # is fine) — verify by aspect ratio so a wrong-frame file (e.g. the
            # full survey-frame bathymetry) is rejected instead of silently
            # stretched into the model window.
            _ar_model = float(vp0_native.shape[1]) / float(vp0_native.shape[2])
            _ar_sd = float(_sd.shape[0]) / float(_sd.shape[1])
            if abs(_ar_sd - _ar_model) / _ar_model > 0.03:
                raise ValueError(
                    f"reparam.seabed_depth_path shape {tuple(_sd.shape)} "
                    f"(ny/nx={_ar_sd:.3f}) does not match the model lateral "
                    f"aspect ratio {_ar_model:.3f} (native "
                    f"{vp0_native.shape[1]}x{vp0_native.shape[2]}); wrong-frame "
                    "bathymetry? regenerate on the model's lateral window.")
            _sd_r = _seabed_to_grid(_sd, vp0_native.shape[1:], native_dh,
                                    (int(ny_), int(nx_)), float(dh))
            _mask_np = water_mask_from_seabed_depth(
                _sd_r, nz=int(nz), dh_z_m=float(dh))
            if (nyp, nxp) != (ny_, nx_):
                _mask_np = np.pad(_mask_np, ((0, 0), (0, nyp - ny_),
                                             (0, nxp - nx_)), mode="edge")
            _seabed_override = torch.from_numpy(_mask_np).to(dev)
            if rank == 0:
                print(f"[freqsel] seabed water-pin from "
                      f"{spec.reparam.seabed_depth_path}: "
                      f"{int(_mask_np.sum()):,}/{_mask_np.size:,} voxels "
                      f"({100 * _mask_np.mean():.2f}%), seabed depth "
                      f"[{_sd_r.min():.0f}, {_sd_r.max():.0f}] m", flush=True)

        # ---- parameterisation: grid vp or reparam INR ---------------------
        vp = None
        if si == 0:
            if use_reparam:
                net = _build_reparam_net(spec.reparam, base_t,
                                         spec.model_bounds.get("vp"),
                                         water_mask_override=_seabed_override,
                                         param_bases=param_bases)
                net = net.to(dev)
                _set_physical_inr_frame(net, shape=_solver_shape, dh=float(dh),
                                        frame_shape=_frame_shape,
                                        frame_dh=native_dh)
                if dd_on:
                    import torch.distributed as dist
                    for p in net.parameters():
                        dist.broadcast(p.data, src=0)
                optimizer = _freqsel_adam(net.parameters(), spec.reparam.lr)
                self._restore_optimizer_state(spec, optimizer, rank)
            else:
                vp = base_t.clone().requires_grad_(True)
                optimizer = _freqsel_adam([vp], spec.optimizer.lr)
        else:
            if use_reparam:
                # Carry the network across the band: resample only its base to
                # the new grid, keep hash+SIREN params AND Adam state (the whole
                # point — the coarse structure learned in prior bands stays).
                # Rebuild the water-pin mask on the new grid (same base==water_vp
                # basis as the si=0 build) and hand it to update_base_velocity —
                # a grid (shape) change otherwise DROPS the stale render pin
                # (VelocityINR.update_base_velocity), so water would un-pin at
                # the 2-8 band. Mirrors the random/CRG path (passes new_mask).
                _wvp = float(getattr(spec.reparam, "water_vp_m_s", 1500.0))
                # Seabed pin (rebuilt on THIS stage's grid above) wins over the
                # vp==water_vp heuristic; keep the fallback for legacy configs.
                _wm = (_seabed_override if _seabed_override is not None
                       else ((base_t.detach() == _wvp)
                             if bool(getattr(spec.reparam,
                                             "mask_water_layer", False))
                             else None))
                if free_multi:
                    # Multi-parameter: carry the shared trunk, resample EVERY
                    # channel's base to the new grid (param_bases was rebuilt
                    # for this stage's dh above). vp (channel 0) drives _wm.
                    net.update_base_models(param_bases, water_mask=_wm)
                else:
                    net.update_base_velocity(base_t, water_mask=_wm)
                # The update rebuilt the coordinates as i/n of this grid; put
                # them back in the native frame so the carried field does not
                # stretch with the band grid.
                _set_physical_inr_frame(net, shape=_solver_shape, dh=float(dh),
                                        frame_shape=_frame_shape,
                                        frame_dh=native_dh)
                if bool(getattr(stage, "optimizer_reset", False)):
                    optimizer = _freqsel_adam(net.parameters(), spec.reparam.lr)
            else:
                # grid-vp mode: the MODEL always carries the previous band's
                # result forward. Cloning ``base_t`` here restarted every stage
                # from the pristine init model, which silently turns a frequency
                # ladder into "only the last rung counts". The reparam branch
                # above never had the bug: it carries ``net`` and only swaps
                # the base, which is why production iFWI runs never showed it.
                # The OPTIMIZER follows the conventional path's rule
                # (_helpers.stages._prepare_stage): on the same grid -- same dh
                # and same padded shape -- the stage keeps the same leaf and the
                # same Adam, moments and step count included, unless it sets
                # ``optimizer_reset``. A grid change rebuilds it: Adam's state
                # is shape-bound. Rebuilding on every entry ignored
                # ``optimizer_reset: false``.
                prev = getattr(self, "_freqsel_last_vp", None)
                if prev is None:
                    vp = base_t.clone().requires_grad_(True)
                    optimizer = _freqsel_adam([vp], spec.optimizer.lr)
                else:
                    prev_dh = float(self._freqsel_last_dh)
                    same_grid = (abs(prev_dh - float(dh)) < 1e-9
                                 and tuple(prev.shape) == tuple(base_t.shape))
                    if same_grid and not bool(getattr(stage, "optimizer_reset",
                                                      False)):
                        # the very leaf the carried Adam steps: its state is
                        # keyed on this tensor, so a copy would start from zero
                        vp = prev
                    else:
                        prev = prev.detach()
                        if same_grid:
                            _why = "optimizer_reset=True — Adam moments dropped"
                        else:
                            # crop the previous result to its physical grid
                            # (drop the DD tile pad), sample it at this grid's
                            # positions, pad like the base. The resample
                            # returns a plain tensor, so the clone below is a
                            # LEAF Adam can step.
                            _ps = self._freqsel_last_shape
                            prev = prev[tuple(slice(0, int(n)) for n in _ps)]
                            prev = _resample_to_grid(prev, prev_dh, gshape, float(dh))
                            prev = torch.from_numpy(np.pad(
                                prev.cpu().numpy(), _pad, mode="edge"))
                            _why = (f"dh {prev_dh:g} -> {float(dh):g} m — Adam "
                                    "moments dropped (its state is shape-bound)")
                        vp = prev.to(dev).clone().requires_grad_(True)
                        optimizer = _freqsel_adam([vp], spec.optimizer.lr)
                        if rank == 0:
                            print(f"[freqsel] stage {si}: {_why}", flush=True)
        # per-stage lr scale: this stage's lr is the base (reparam.lr, or
        # optimizer.lr for a grid model) x this stage's own scale, written on
        # EVERY entry. A carried optimizer (reparam, or a grid model on the
        # same grid) and a restored state come back with the lr they last ran
        # at, so skipping a 1.0 scale kept the lr of the previous stage or of
        # the checkpoint. Under reparam lr_scale does not apply (FWISpec
        # rejects a set one).
        if use_reparam:
            _base, _scale = spec.reparam.lr, float(stage.inr_lr_scale)
            _what = "reparam.lr x inr_lr_scale"
        else:
            _base, _scale = spec.optimizer.lr, float(stage.lr_scale)
            _what = "optimizer.lr x lr_scale"
        _lr = float(_base) * _scale
        for g in optimizer.param_groups:
            g["lr"] = _lr
        if rank == 0:
            print(f"[freqsel] stage {si}: Adam lr={_lr:g} ({_what} {_scale:g}), "
                  "eps=1e-8, betas=(0.9, 0.999)", flush=True)

        _b = spec.model_bounds.get("vp") if spec.model_bounds else None
        vmin = float(_b.min) if _b is not None and _b.min is not None else None
        vmax = float(_b.max) if _b is not None and _b.max is not None else None

        loss_fn = fsl.SteadyGCNLoss(
            comb, targets, int(fspec.steady_samples),
            int(fspec.slack_samples), dev, distributed=dd_on,
            eps=float(fspec.eps))

        _dd_rc = _dd_config()[3] if dd_on else 0

        def _leaf():
            if not use_reparam:
                return vp
            if dd_on:
                # Mirror the plan-streaming DD path (_dd_render_tile): render ONLY
                # this rank's solver tile so the reparam render divides across
                # tiles, instead of every rank rendering the full grid each
                # iteration (the dominant per-iter cost at fine grids).
                return _dd_render_tile(net, _dd_tile_bounds(solver), _dd_rc)
            with torch.no_grad():
                m = net.render(chunk_rows=chunk_rows).detach().clone()
            return m.requires_grad_(True)

        # Multi-parameter reparam: render EVERY freed parameter as its own
        # detached leaf (vp, z, …) in solver-model order so the solver inverts
        # them jointly. _get_models() returns (leaves, models); single-param
        # keeps the Gardner-coupled model list via _solver_models.
        def _leaves():
            if dd_on:
                # DD: render ONLY this rank's solver tile, per channel -> N leaves.
                return _dd_render_tile_multi(
                    net, _dd_tile_bounds(solver), _dd_rc, n_free)
            with torch.no_grad():
                fields = net.render_all(chunk_rows=chunk_rows)
            return [fields[i].detach().clone().requires_grad_(True)
                    for i in range(n_free)]

        def _get_models():
            if free_multi:
                lv = _leaves()
                return lv, list(lv)
            lf = _leaf()
            return [lf], _solver_models(lf, spec)

        # ---- capture + ownership + steady-state QC ------------------------
        pool0 = sched.pools[0]
        bins0 = np.arange(len(pool0))
        leaves, models0 = _get_models()
        # No grad: this record only feeds the steady-state check, which takes
        # it detached and then drops it. Under grad the c backend saves the
        # boundary for a backward that never comes -- on a fine rung that is
        # most of the card, and the check itself then has nowhere to put its
        # own few GB. That is how a resumed fine stage died before its first
        # iteration.
        with torch.no_grad():
            rec0 = solver(
                fsl.encoded_wavelet(comb, bins0, nt, float(fspec.ramp_s), dev),
                targets.node_grid[pool0][None].astype(np.int32), rec_table,
                models=models0)
        own = getattr(solver, "_own_rec_idx", None)
        # Debug: audit receiver ownership across ranks (SWEEP_FREQSEL_OWN_AUDIT=1).
        # Duplicated/dropped receivers at tile cut planes would bias the GCN loss.
        if os.environ.get("SWEEP_FREQSEL_OWN_AUDIT") == "1" and dd_on:
            import torch.distributed as dist
            _cnt = torch.zeros(int(targets.n_union), device=dev)
            _idx = (np.arange(targets.n_union) if own is None
                    else np.asarray(own))
            _cnt[torch.as_tensor(_idx, device=dev, dtype=torch.long)] = 1.0
            dist.all_reduce(_cnt)
            _dup = int((_cnt > 1.5).sum()); _drop = int((_cnt < 0.5).sum())
            if rank == 0:
                print(f"[freqsel][own-audit] n_union={targets.n_union} "
                      f"duplicated={_dup} dropped={_drop} "
                      f"(sum_owned={int(_cnt.sum())})", flush=True)
        # Column pruning needs the draw schedule BEFORE the table is built.
        # sched.plan() is side-effect free (RNG snapshot/restore), so asking
        # for it here does not perturb the sequence draw() serves the loop.
        _bin_use = None
        if os.environ.get("SWEEP_FREQSEL_D_PRUNE") == "1":
            _plan = sched.plan(int(stage.epochs))
            _bin_use = np.zeros((targets.n_nodes, comb.n_bins), bool)
            for _p, _b in _plan:
                _bin_use[np.asarray(_p, np.int64), np.asarray(_b, np.int64)] = True
        targets.bind_ownership(
            np.arange(targets.n_union) if own is None else own, dev,
            bin_use=_bin_use)
        chk = loss_fn.two_window_check(
            rec0.detach(), pool0, bins0, int(fspec.steady_samples),
            int(fspec.slack_samples))
        print(f"[freqsel][rank{rank}] stage {si} steady-state two-window "
              f"check: median rel diff = {chk:.3e}", flush=True)
        del rec0, leaves

        # ---- coarse-to-fine hash schedule (per stage) ---------------------
        _c2f = getattr(getattr(spec.reparam, "hash", None), "c2f", None)
        c2f_on = bool(
            use_reparam and _c2f is not None and bool(_c2f.enabled)
            and _has_hash_schedule(net))
        if c2f_on and rank == 0:
            print(f"[freqsel] stage {si} c2f: base_levels={_c2f.base_levels} "
                  f"ramp={_c2f.ramp} warmup={_c2f.warmup} "
                  f"ramp_end={_c2f.ramp_end} "
                  f"final_levels={getattr(_c2f, 'final_levels', None)}",
                  flush=True)

        # ---- optional water-column gradient freeze, on THIS stage's grid --
        # Zero the vp gradient above the seabed so the inversion never updates
        # the (known) water velocity. Grid-mode path (reparam masks the water
        # itself). Rebuilt every stage: a mask cached across stages (or across
        # tasks on one runner) does not fit another grid.
        _fwl = getattr(spec, "freeze_water_layer", None)
        _freeze_mask_t = None
        if (_fwl is not None and getattr(_fwl, "enabled", False)
                and _fwl.seabed_depth_path is not None):
            from sweep_tasks.bathymetry import (
                load_seabed_depth_npz, water_mask_from_seabed_depth)
            _sd = load_seabed_depth_npz(_fwl.seabed_depth_path)
            _sdr = _seabed_to_grid(_sd, vp0_native.shape[1:], native_dh,
                                   gshape[1:], float(dh))
            _wm = water_mask_from_seabed_depth(
                _sdr, nz=int(nz), dh_z_m=float(dh),
                buffer_cells=int(_fwl.buffer_cells))
            if not is2d and (nyp, nxp) != (ny_, nx_):
                _wm = np.pad(_wm, ((0, 0), (0, nyp - ny_),
                                   (0, nxp - nx_)), mode="edge")
            _freeze_mask_t = torch.from_numpy(_wm).to(dev).bool()
            if rank == 0:
                print("[freqsel] freeze_water_layer: %d voxels zeroed"
                      % int(_freeze_mask_t.sum()), flush=True)

        stage_epochs = int(stage.epochs)
        snap_every = int(os.environ.get("SWEEP_SNAP_EVERY",
                                        str(max(1, stage_epochs // 10))))
        use_cuda = torch.cuda.is_available()
        _TPROF = os.environ.get("SWEEP_TASKS_TPROF") == "1"
        _pf = {"render": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0,
               "reparam": 0.0, "step": 0.0}

        def _pf_sync():
            if _TPROF and use_cuda:
                torch.cuda.synchronize()

        for it in range(stage_epochs):
            gi = epoch_offset + it       # global iteration index
            ti = _time.perf_counter()
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            pool, bins = sched.draw(it)
            if c2f_on:
                # whole-run progress: single network across all stages, schedule
                # advances monotonically (never reset per band). Single-stage
                # (total_epochs==stage_epochs, offset 0) reduces to it/stage.
                _act = _advance_hash_schedule(
                    net, (epoch_offset + it) / max(1, total_epochs - 1),
                    _c2f, optimizer)
                if _act is not None and rank == 0 and (
                        it < 3 or it % 10 == 0 or it == stage_epochs - 1):
                    print(f"[freqsel] s{si} c2f it {it}: active levels "
                          f"{_act[0]:.2f}/{_act[1]}", flush=True)
            optimizer.zero_grad()
            _pf_sync(); _pa = _time.perf_counter()
            leaves, models = _get_models()
            leaf = leaves[0]
            _pf_sync(); _pf["render"] += _time.perf_counter() - _pa; _pa = _time.perf_counter()
            # One retry, on the host, if the card cannot hold the boundary.
            # The switch has to be COLLECTIVE: a rank that quietly moved its
            # strips to the host while its neighbours stayed on the card would
            # desynchronise the next halo exchange. So every rank votes each
            # iteration -- one scalar all-reduce against a 35-130 s step -- and
            # they fall back together or not at all.
            for _attempt in (0, 1):
                _oom = 0
                try:
                    syn = solver(
                        fsl.encoded_wavelet(comb, bins, nt, float(fspec.ramp_s), dev),
                        targets.node_grid[pool][None].astype(np.int32), rec_table,
                        models=models)
                    _pf_sync(); _pf["fwd"] += _time.perf_counter() - _pa; _pa = _time.perf_counter()
                    # Debug: dump the raw forward record (SWEEP_FREQSEL_DUMP_REC=<dir>)
                    # + owned receiver indices — localizes DD-vs-single divergence to
                    # receivers/onset times. Diagnostic only.
                    _rdump = os.environ.get("SWEEP_FREQSEL_DUMP_REC")
                    if _rdump and it < int(os.environ.get("SWEEP_FREQSEL_DUMP_GRAD_ITERS", "1")):
                        os.makedirs(_rdump, exist_ok=True)
                        _own_i = getattr(solver, "_own_rec_idx", None)
                        np.savez(os.path.join(
                            _rdump, f"rec_s{si}_it{it}_r{rank}.npz"),
                            syn=syn.detach().cpu().numpy(),
                            own=(np.arange(targets.n_union) if _own_i is None
                                 else np.asarray(_own_i)),
                            pool=np.asarray(pool), bins=np.asarray(bins))
                    J, npool = loss_fn(syn, pool, bins)
                    _pf_sync(); _pf["loss"] += _time.perf_counter() - _pa; _pa = _time.perf_counter()
                    J.backward()
                    _pf_sync(); _pf["bwd"] += _time.perf_counter() - _pa; _pa = _time.perf_counter()
                except torch.OutOfMemoryError:
                    _oom = 1
                if world > 1:
                    import torch.distributed as _d
                    _v = torch.tensor([_oom], device=dev, dtype=torch.int32)
                    _d.all_reduce(_v, op=_d.ReduceOp.MAX)
                    _oom = int(_v.item())
                if not _oom:
                    break
                if world > 1:
                    # No in-process recovery under DD. The out-of-memory lands
                    # inside a backward, whose halo exchanges are collectives;
                    # an interrupted collective leaves the communicator unusable
                    # and rebuilding the wrapper from there aborts a rank
                    # (SIGABRT, with the NCCL watchdog close behind). The core
                    # also fixes the boundary's home at construction. So say
                    # what to do and stop, rather than corrupt the run.
                    raise RuntimeError(
                        f"[freqsel] s{si} it {it}: CUDA out of memory. Re-run "
                        "this stage with SWEEP_FREQSEL_BOUNDARY_STORAGE=cpu "
                        "(the boundary then stages through the host), or give "
                        "it more ranks.")
                if boundary_on_host:
                    raise RuntimeError(
                        f"[freqsel] s{si} it {it}: out of memory with the "
                        "boundary already staged on the host; lower the batch, "
                        "add ranks, or coarsen this stage")
                syn = J = None
                optimizer.zero_grad(set_to_none=True)
                del solver
                if use_cuda:
                    torch.cuda.empty_cache()
                solver = _make_solver(storage="cpu")
                boundary_on_host = True
                # The failed backward already freed the reparam half of the
                # graph, so the retry has to render again -- reusing ``models``
                # would walk a graph that is no longer there.
                leaves, models = _get_models()
                leaf = leaves[0]
                if rank == 0:
                    print(f"[freqsel] s{si} it {it}: boundary saving fell back "
                          "to host memory after a CUDA OOM (all ranks)",
                          flush=True)
            g = leaf.grad
            # Optional water-column gradient freeze (mask built for THIS
            # stage's grid above the loop). See :class:`FreezeWaterLayerSpec`.
            if _freeze_mask_t is not None and g is not None:
                g[_freeze_mask_t] = 0.0
            # Debug: dump the raw velocity gradient of selected iters for
            # DD-vs-single parity checks (SWEEP_FREQSEL_DUMP_GRAD=<dir>).
            # DD dumps this rank's TILE grad + its global bounds; single dumps
            # the full-grid grad. Diagnostic only, no effect on the update.
            _gdump = os.environ.get("SWEEP_FREQSEL_DUMP_GRAD")
            if _gdump and g is not None and it < int(
                    os.environ.get("SWEEP_FREQSEL_DUMP_GRAD_ITERS", "1")):
                os.makedirs(_gdump, exist_ok=True)
                if dd_on:
                    _tb = _dd_tile_bounds(solver)
                    np.savez(os.path.join(_gdump, f"grad_s{si}_it{it}_rank{rank}.npz"),
                             g=g.detach().cpu().numpy(),
                             bounds=np.asarray(_tb, dtype=np.int64))
                else:
                    np.savez(os.path.join(_gdump, f"grad_s{si}_it{it}_single.npz"),
                             g=g.detach().cpu().numpy())
            if free_multi:
                # Joint N-channel reparam backward. Each freed parameter's own
                # leaf carries dL/d(that parameter); map them all onto the shared
                # trunk. illumination precond is not supported on this path.
                if dd_on:
                    # DD: leaves are per-channel TILE renders; zero the water in
                    # this rank's tile window, push each channel through the net
                    # tile-locally, all_reduce param grads across tiles.
                    _tb = _dd_tile_bounds(solver)
                    _wt = water_t[_tb[0]:_tb[1], _tb[2]:_tb[3], _tb[4]:_tb[5]]
                    for _lf in leaves:
                        if _lf.grad is not None:
                            _lf.grad[_wt] = 0.0
                    _dd_backward_tile_multi(net, leaves, _tb, _dd_rc)
                else:
                    # single GPU: full-grid leaves -> one chunked backward pass.
                    grads = []
                    for _lf in leaves:
                        _gg = _lf.grad
                        if _gg is None:
                            _gg = torch.zeros_like(net.base_stack[0])
                        else:
                            _gg[water_t] = 0.0
                        grads.append(_gg)
                    net.backward_gradients(grads, chunk_rows=chunk_rows)
            elif dd_on and use_reparam:
                # Tile-grad path (mirrors plan-streaming _dd_backward_tile): g
                # covers only this rank's tile window; zero the water inside
                # the window, push it through the net tile-locally, and let
                # _dd_backward_tile all_reduce the PARAM grads. Called
                # UNCONDITIONALLY (zero-filling when g is None) so the
                # collective stays consistent across ranks. This replaces the
                # full-grid grad all_reduce + redundant full-model reparam
                # backward on every rank. illumination is force-disabled
                # under DD, so that branch is moot here.
                _tb = _dd_tile_bounds(solver)
                if g is not None:
                    g[water_t[_tb[0]:_tb[1], _tb[2]:_tb[3], _tb[4]:_tb[5]]] = 0.0
                _dd_backward_tile(net, leaf, _tb, _dd_rc)
            elif g is not None:
                sill_sum = rill_sum = None
                if illum_on:
                    sill_sum, rill_sum = _accumulate_illumination(
                        illum_solver, None, None)
                if dd_on:
                    import torch.distributed as dist
                    dist.all_reduce(g)
                    if illum_on and sill_sum is not None and rill_sum is not None:
                        dist.all_reduce(sill_sum)
                        dist.all_reduce(rill_sum)
                if illum_on:
                    _apply_illumination_precond(
                        g, sill_sum, rill_sum, eps=float(illum_spec.epsilon),
                        exponent=float(illum_spec.exponent),
                        relative_epsilon=getattr(illum_spec, "relative_epsilon", None))
                    if it == 0 and rank == 0 and sill_sum is None:
                        print("[freqsel] WARN illum on but solver illumination "
                              "is None (compute_illumination not honored?) — "
                              "precond is a NO-OP", flush=True)
                g[water_t] = 0.0
                if use_reparam:
                    net.backward_velocity_gradient(g, chunk_rows=chunk_rows)
            _pf_sync(); _pf["reparam"] += _time.perf_counter() - _pa; _pa = _time.perf_counter()
            optimizer.step()
            _pf_sync(); _pf["step"] += _time.perf_counter() - _pa
            if not use_reparam:
                with torch.no_grad():
                    if vmin is not None:
                        vp.clamp_(vmin, vmax)
                    vp[water_t] = base_t[water_t]
            losses.append(float(J.detach()) / npool)
            times.append(_time.perf_counter() - ti)
            peaks.append(torch.cuda.max_memory_allocated() / 2 ** 30
                         if use_cuda else 0.0)
            if rank == 0 and (it < 30 or it % max(1, stage_epochs // 40) == 0
                              or it == stage_epochs - 1):
                print(f"[freqsel] s{si} it {it:4d} "
                      f"{sched.iteration_label(it)}"
                      f"  mean(1-GCN)={losses[-1]:.5f}  "
                      f"iter_s={times[-1]:.1f}  peak_gb={peaks[-1]:.2f}",
                      flush=True)
            if _TPROF and rank == 0:
                n = it + 1
                print(f"[tprof] avg/iter ms: render={_pf['render']*1e3/n:5.0f} "
                      f"fwd={_pf['fwd']*1e3/n:5.0f} loss={_pf['loss']*1e3/n:5.0f} "
                      f"bwd={_pf['bwd']*1e3/n:5.0f} reparam={_pf['reparam']*1e3/n:5.0f} "
                      f"step={_pf['step']*1e3/n:5.0f}", flush=True)
            if rank == 0 and (it + 1) % snap_every == 0:
                with torch.no_grad():
                    m = (net.render(chunk_rows=chunk_rows).detach()
                         if use_reparam else vp.detach())
                np.save(task_dir / f"vp_iter{gi + 1:04d}.npy",
                        (m[:, :nx_] if is2d
                         else m[:, :ny_, :nx_]).cpu().numpy())
                if use_reparam and (spec.reparam.save_net
                                    or os.environ.get("SWEEP_SAVE_REPARAM_NET") == "1"):
                    torch.save(net.state_dict(),
                               task_dir / f"reparam_net_iter{gi + 1:04d}.pt")
                    # Save Adam's moments with the weights, or resuming from
                    # this snapshot is not equivalent to a continuous run.
                    if bool(getattr(spec.reparam, "save_optimizer", True)):
                        torch.save({"optimizer": optimizer.state_dict(),
                                    "meta": {"global_iter": gi + 1, "stage": si,
                                             "lr": float(spec.reparam.lr)}},
                                   task_dir / f"optim_iter{gi + 1:04d}.pt")
                np.savez(task_dir / "curves.npz",
                         losses=np.array(losses), iter_s=np.array(times),
                         peak_gb=np.array(peaks))

        # keep a handle to the final grid-vp so the caller can render output,
        # and the grid it lives on so the next stage can carry it over
        self._freqsel_last_vp = vp
        self._freqsel_last_dh = float(dh)
        self._freqsel_last_shape = tuple(int(n) for n in gshape)
        self._freqsel_last_chunk_rows = chunk_rows
        return net, optimizer, ny_, nx_, chk

    def _run_fwi_freqsel(self, spec: FWISpec, task_dir: Path):
        """Frequency-selection (steady-state comb) encoded FWI / iFWI.

        Deterministic zero-crosstalk source encoding (Tromp & Bachmann 2019)
        on the mode-B super-shot path: node pools rotate deterministically,
        every pool node emits one exclusive comb bin (permuted per iter),
        the loss is the per-node complex-cosine GCN on steady-window DFT
        coefficients — wavelet-free, plan-free, zero per-iter I/O. Obs are
        pre-extracted coefficient shards (field path) or synthesized from a
        true model once at setup (test path). Heavy lifting lives in
        :mod:`sweep_tasks.freqsel`.

        Multi-stage: when ``spec.stages`` is set, runs each stage in one
        process — per-stage ``frequency`` sub-spec (comb + coeff_shards) and
        ``dh_m``/``dt_s`` rebuild the comb/targets/solver, while the reparam
        network is carried across bands (its base resampled to the new grid,
        params + Adam state kept), and so is a grid model, its Adam rebuilt
        only by ``optimizer_reset`` or a grid change. A single-stage run is
        bit-identical to the original single-band path.
        """
        import json as _json

        import numpy as np
        import torch

        fspec_global = spec.source_encoding.frequency
        dd_on, dd_py, dd_px, dd_rc = _dd_config()
        rank, world = 0, 1
        if dd_on:
            import torch.distributed as dist
            if not dist.is_initialized():
                # NCCL watchdog default is 600 s; freqsel's iteration 0 at fine
                # grids runs ~520 s of one-time warmup (first adjoint launch,
                # boundary buffer allocs), so the default is one bad node away
                # from a spurious SIGABRT (seen on the finest rung of a
                # production cascade). 1800 s default, env-overridable.
                from datetime import timedelta
                dist.init_process_group("nccl", timeout=timedelta(seconds=int(
                    os.environ.get("SWEEP_DD_NCCL_TIMEOUT_S", "1800"))))
            rank, world = dist.get_rank(), dist.get_world_size()
            if dd_py * dd_px != world:
                raise ValueError(
                    f"SWEEP_DD_PY*PX={dd_py * dd_px} != world={world}")
            li = int(os.environ.get("LOCAL_RANK", rank)) % max(
                1, torch.cuda.device_count())
            torch.cuda.set_device(li)
            dev = torch.device(f"cuda:{li}")
        else:
            dev = _resolve_device(spec.device)
        _apply_seed(spec.seed)

        stages = _normalise_stage_list(spec)
        total_epochs = int(sum(int(s.epochs) for s in stages))
        use_reparam = spec.reparam is not None
        chunk_rows = (int(getattr(spec.reparam, "backward_chunk_rows", 8))
                      if use_reparam else 0)
        illum_spec = getattr(spec, "illumination_precondition", None)
        illum_on = bool(illum_spec is not None and illum_spec.enabled)
        if illum_on and dd_on:
            # solver illumination is tile-local (per-rank x/y sub-block) while the
            # reparam leaf grad is the full model; preconditioning would need a
            # gather/scatter that isn't implemented. Disable rather than crash.
            if rank == 0:
                print("[freqsel] WARN illumination_precondition not supported "
                      "under DD (tile-local illumination vs full-model grad) "
                      "— DISABLED for this DD run", flush=True)
            illum_on = False

        _init_refs = _normalize_fwi_init_models(spec)
        _vp_ref = next((m for m in _init_refs
                        if getattr(m, "name", None) == "vp"), _init_refs[0])
        vp0_native = _model_array(_vp_ref)
        if vp0_native.ndim not in (2, 3):
            raise ValueError("freqsel init must be 2-D (nz, nx) or 3-D "
                             f"(nz, ny, nx); got shape {vp0_native.shape}")
        if vp0_native.ndim == 2:
            # 2-D freqsel: same encoding/loss, single lateral axis. Domain
            # decomposition and the (ny, nx) bathymetry pin are 3-D-only.
            if (getattr(spec, "dd", None) is not None
                    and getattr(spec.dd, "enabled", False)):
                raise ValueError("freqsel 2-D does not support domain "
                                 "decomposition (dd.enabled); run single-device")
            if (spec.reparam is not None
                    and getattr(spec.reparam, "seabed_depth_path", None)):
                raise ValueError("reparam.seabed_depth_path is a 3-D (ny, nx) "
                                 "bathymetry; not available for 2-D freqsel")
        native_dh = float(spec.grid.dh)

        task_dir.mkdir(parents=True, exist_ok=True)
        net = optimizer = None
        # stage si>0 carries this forward as its starting model; clear it so a
        # reused runner instance cannot leak one task's model into the next
        self._freqsel_last_vp = None
        losses, times, peaks = [], [], []
        chk_first = None
        ny_ = nx_ = None
        for si, stage in enumerate(stages):
            fspec = getattr(stage, "frequency", None) or fspec_global
            if fspec is None:
                raise ValueError(
                    "frequency_selection: stage has no frequency sub-spec "
                    "and source_encoding.frequency is unset")
            if rank == 0:
                for line in stage_freq_override_warnings(fspec_global, fspec, si):
                    print(line, flush=True)
            dh = float(stage.dh_m) if stage.dh_m else native_dh
            dt = float(stage.dt_s) if stage.dt_s else float(spec.time.dt)
            net, optimizer, ny_, nx_, chk = self._freqsel_run_stage(
                spec, stage, si, fspec, dh, dt, dev,
                dd_on, dd_py, dd_px, rank, world, task_dir,
                use_reparam, net, optimizer, chunk_rows,
                vp0_native, native_dh, illum_spec, illum_on,
                losses, times, peaks, len(losses), total_epochs)
            if chk_first is None:
                chk_first = chk
            _release_card_between_stages(si, rank)

        artifacts, summary = {}, {}
        if rank == 0:
            if use_reparam:
                with torch.no_grad():
                    m = net.render(chunk_rows=chunk_rows).detach()
            else:
                m = self._freqsel_last_vp.detach()
            final = (m[:, :nx_] if m.ndim == 2
                     else m[:, :ny_, :nx_]).cpu().numpy()
            np.save(task_dir / "inverted_vp.npy", final)
            # Multi-parameter reparam: also dump every freed channel (z, …).
            if use_reparam and getattr(spec.reparam, "free_params", None):
                with torch.no_grad():
                    allf = net.render_all(chunk_rows=chunk_rows).detach()
                for _i, _fp in enumerate(spec.reparam.free_params):
                    np.save(task_dir / f"inverted_{_fp.name}.npy",
                            (allf[_i][:, :nx_] if allf[_i].ndim == 2
                             else allf[_i][:, :ny_, :nx_]).cpu().numpy())
            np.savez(task_dir / "curves.npz", losses=np.array(losses),
                     iter_s=np.array(times), peak_gb=np.array(peaks))
            summary = {
                "mode": "frequency_selection",
                "reparam": bool(use_reparam),
                "stages": len(stages),
                "nt": int(times and 0 or 0),
                "steady_check": float(chk_first)
                if chk_first is not None else 0.0,
                "loss_kind": "steady_gcn",  # loss_* below are mean(1-GCN)
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "mean_iter_s": float(np.mean(times[1:])) if len(times) > 1
                else (float(times[0]) if times else 0.0),
            }
            (task_dir / "summary.json").write_text(
                _json.dumps(summary, indent=2))
            artifacts = {"inverted_vp": str(task_dir / "inverted_vp.npy")}
            # opt-in: dump the reparam net weights (reparam.save_net or the
            # SWEEP_SAVE_REPARAM_NET=1 env override) so per-level hash features can
            # be rendered offline. Off by default (large file). Mirrors the
            # plan-streaming path; freqsel writes to task_dir root (alongside
            # inverted_vp.npy), not task_dir/output.
            if (use_reparam and net is not None
                    and (spec.reparam.save_net
                         or os.environ.get("SWEEP_SAVE_REPARAM_NET") == "1")):
                net_path = task_dir / "reparam_net.pt"
                torch.save(net.state_dict(), net_path)
                artifacts["reparam_net"] = str(net_path)
                if bool(getattr(spec.reparam, "save_optimizer", True)):
                    _op = task_dir / "optim.pt"
                    torch.save({"optimizer": optimizer.state_dict(),
                                "meta": {"global_iter": None, "stage": None,
                                         "lr": float(spec.reparam.lr)}}, _op)
                    artifacts["optimizer_state"] = str(_op)
                print(f"[freqsel] saved reparam net -> {net_path}", flush=True)
            print(f"[freqsel] DONE ({len(stages)} stage(s)) "
                  f"mean(1-GCN) {losses[0]:.4f} -> {losses[-1]:.4f}",
                  flush=True)
        return artifacts, summary
