"""Multiscale-stage prep + tensor/obs resampling + syn bandpass. Verbatim from runner.py."""
import os
import numpy as np
import torch

from sweep_tasks.preproc.filter import bandpass as _bandpass_cpu
from sweep_tasks.preproc.filter import bandpass_torch as _bandpass_torch_fft
from sweep_tasks._helpers.optimizer import (
    _apply_stage_lr_scale,
    _build_optimizer,
    _build_reparam_optimizer,
    _remember_initial_lrs,
)
from sweep_tasks._helpers.solver_build import _build_solver
from sweep_tasks._helpers.util import _is_cuda_dev
from sweep_tasks._helpers.wavelet_build import _build_wavelet

def _normalise_stage_list(spec) -> list:
    """Return the effective stage list (single-stage fallback when spec.stages is None)."""

    if spec.stages:
        return list(spec.stages)
    from sweep_tasks.schemas import StageSpec
    return [StageSpec(epochs=spec.epochs, wavelet=None, lr_scale=1.0)]


# ---------------------------------------------------------------------------
# Per-stage rebuilds (Gaps 4 + 5)
# ---------------------------------------------------------------------------


def _prepare_stage(
    *, spec, stage, state: dict, equation_cls, required_names: list[str],
    dev, dist_info, stage_idx: int,
) -> None:
    """Mutate ``state`` so the runner can run this stage end-to-end.

    Detects which of ``(dh, dt, nt, bandpass, wavelet, batch_size, lr_scale)``
    changed against the previous stage state. Triggers any combination of:

    - vp resample (bilinear) + fresh leaf tensors
    - solver rebuild at the new ``(shape, dh, dt, nt)``
    - geometry re-snap from the cached pristine PhysicalGeometry
    - obs rebuild from the cached pristine obs (time-resample + bandpass)
    - wavelet rebuild
    - optimizer re-init (Adam state is shape-bound) + initial_lrs cache
    """
    import torch

    new_dh = float(stage.dh_m) if stage.dh_m is not None else state["dh"]
    new_dt = float(stage.dt_s) if stage.dt_s is not None else state["dt"]
    if stage.nt is not None:
        new_nt = int(stage.nt)
    elif abs(new_dt - state["dt"]) > 1e-12:
        new_nt = int(round(state["nt"] * state["dt"] / new_dt))
    else:
        new_nt = state["nt"]

    grid_changed = abs(new_dh - state["dh"]) > 1e-12
    time_changed = (abs(new_dt - state["dt"]) > 1e-12) or (new_nt != state["nt"])
    wavelet_changed = stage.wavelet is not None or state.get("_stage_wavelet_idx", -1) != stage_idx

    if grid_changed and dist_info.is_root:
        print(f"[stage {stage_idx}] dh: {state['dh']} -> {new_dh}")
    if time_changed and dist_info.is_root:
        print(f"[stage {stage_idx}] dt: {state['dt']} -> {new_dt}, nt: {state['nt']} -> {new_nt}")

    # ---- Geometry re-snap from pristine physical positions -----------------
    # We re-snap whenever:
    #   - the stage changes dh (grid_changed)
    #   - state hasn't been initialised yet
    #   - dedupe is enabled and we haven't applied it at this dh yet (the
    #     initial geometry parse always keeps all receivers; per-stage
    #     dedupe is the first time it gets applied at the stage's dh)
    need_geom_resnap = (
        grid_changed
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
        from sweep_io.geometry import PhysicalGeometry
        pg: PhysicalGeometry = state["pristine_physical_geom"]
        dedupe_flag = state.get("dedupe_grid_snap", True)
        dedup_method = state.get("dedup_method", "nearest")
        gg, mask = pg.to_grid(
            dh=(float(new_dh),) * pg.ndim,
            dedupe=dedupe_flag,
            dedup_method=dedup_method,
        )
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

    # ---- Pick a new model shape (preserves physical extent) ----------------
    if grid_changed or "shape" not in state:
        new_shape = _shape_for_dh(state.get("pristine_shape", state["shape"]),
                                  state["pristine_dh"], new_dh)
        if dist_info.is_root and grid_changed:
            print(f"[stage {stage_idx}] vp shape: {state['shape']} -> {new_shape}")
        state["shape"] = new_shape

    # ---- vp resample ----------------------------------------------------
    # Two paths:
    #  - Reparam off: bilinear-resample each leaf tensor to the new shape
    #    and make a fresh `requires_grad=True` leaf (Adam state is shape-
    #    bound, so we rebuild the optimizer below).
    #  - Reparam on: keep the same VelocityINR. Update its `base_velocity`
    #    buffer from the pristine init, resampled to the new shape — and
    #    DO NOT rebuild the optimizer (its state on the network's params
    #    is independent of grid shape and is the entire point of the
    #    network-as-vp parameterization).
    if grid_changed:
        reparam_net = state.get("reparam_net")
        if reparam_net is not None:
            pristine_base = state["pristine_base_vp"]
            new_base = _resample_vp_tensor(pristine_base, state["shape"]).detach()
            # Rebuild the water-layer mask at the NEW grid and pass it in.
            # Otherwise update_base_velocity drops the stale (old-shape) mask
            # on the shape change and the water column is inverted freely from
            # stage 1 on. The pristine base has water == water_vp_m_s exactly
            # and bilinear resample preserves that in the water interior, so
            # `new_base == water_vp` reproduces the initial mask at the new shape.
            water_mask = None
            rspec = getattr(spec, "reparam", None)
            # In water_reset_each_stage mode we do NOT re-pin here — the reset
            # block below clears the render pin and re-bakes the water base so
            # the water inverts freely within the stage.
            if rspec is not None and bool(getattr(rspec, "mask_water_layer", False)) \
                    and not bool(getattr(rspec, "water_reset_each_stage", False)):
                water_vp_val = float(getattr(rspec, "water_vp_m_s", 1500.0))
                water_mask = (new_base == water_vp_val)
            reparam_net.update_base_velocity(new_base, water_mask=water_mask)
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
                new_t = _resample_vp_tensor(old_t, state["shape"])
                new_inv.append(new_t)
                new_by_name[name] = new_t
            state["inv_in_order"] = new_inv
            state["inv_by_name"] = new_by_name

    # ---- Water reset-each-stage (free within stage) ------------------------
    # reparam.water_reset_each_stage: snap the water column back to water_vp at
    # the START of every stage, then let it invert FREELY within the stage. The
    # INR is a GLOBAL additive net, so resetting the base alone is not enough —
    # the net's learned water delta persists across stages. So (a) clear the
    # render-time pin and (b) re-bake the base in the water region so the current
    # render there equals water_vp (``base[water] += water_vp - render[water]``);
    # as the net evolves the water drifts from water_vp, and the next stage snaps
    # it back. Runs EVERY stage (incl. the fixed-grid path, where grid_changed is
    # False and the block above is skipped). Deterministic → DDP-consistent.
    _rspec = getattr(spec, "reparam", None)
    _net = state.get("reparam_net")
    if _net is not None and _rspec is not None \
            and bool(getattr(_rspec, "water_reset_each_stage", False)):
        _water_vp = float(getattr(_rspec, "water_vp_m_s", 1500.0))
        _wr = state.get("water_region_mask")
        if _wr is None or tuple(_wr.shape) != tuple(_net.base_velocity.shape):
            # Prefer the seabed-based mask the net was built with; else fall back
            # to pristine == water_vp resampled to the current shape.
            _built = getattr(_net, "water_mask", None)
            if _built is not None and tuple(_built.shape) == tuple(_net.base_velocity.shape):
                _wr = _built.clone()
            else:
                _pri = _resample_vp_tensor(state["pristine_base_vp"], state["shape"])
                _wr = (_pri == _water_vp)
            state["water_region_mask"] = _wr
        _net.water_mask = None  # free within the stage: drop the render-time pin
        with torch.no_grad():
            _render = _net()
            _net.base_velocity[_wr] += (_water_vp - _render[_wr])
            _render_post = _net().detach()
        state["inv_by_name"]["vp"] = _render_post
        state["inv_in_order"] = [
            _render_post if n == "vp" else state["inv_by_name"][n]
            for n in required_names
        ]
        if getattr(dist_info, "is_root", True):
            print(f"[stage {stage_idx}] water reset-each-stage: snapped "
                  f"{int(_wr.sum())} water cells to {_water_vp:.0f} m/s, "
                  f"free within stage", flush=True)

    # ---- Rebuild solver ----------------------------------------------------
    if grid_changed or time_changed or "solver" not in state:
        state["solver"] = _build_solver(
            spec.physics, spec.backend, state["shape"], new_dh, new_dt, new_nt, dev,
        )
        # Drop cached local-window solvers — they were built for the
        # previous stage's (dh, dt, nt) and are no longer valid.
        if "local_solver_cache" in state:
            state["local_solver_cache"].clear()

    # ---- Rebuild wavelet ---------------------------------------------------
    # Rebuild from spec when:
    #   * time grid changed, or
    #   * stage explicitly overrides the wavelet, or
    #   * first stage (no cached wavelet yet), or
    #   * the stage is about to bandpass the wavelet (``target='wavelet'``):
    #     we MUST start from the pristine broadband wavelet rather than
    #     re-filtering whatever the previous stage's bandpass left behind,
    #     otherwise the per-stage filters compose (e.g. 1-4 Hz then 1-7 Hz
    #     stays effectively 1-4 Hz). The schema docs guarantee
    #     ``stages don't compose their filters``; this branch enforces it.
    rebuilt_wavelet = False
    bp_will_filter_wavelet = (
        stage.bandpass is not None
        and getattr(stage.bandpass, "target", "syn") == "wavelet"
    )
    if (
        time_changed
        or stage.wavelet is not None
        or "wavelet" not in state
        or bp_will_filter_wavelet
    ):
        wav_spec = stage.wavelet if stage.wavelet is not None else spec.wavelet
        state["wavelet"] = _build_wavelet(
            wav_spec, spec.time, override_dt=new_dt, override_nt=new_nt,
        )
        rebuilt_wavelet = True

    # ---- Optional wavelet bandpass (target='wavelet') ----------------------
    # When ``stage.bandpass.target == 'wavelet'``, we band-pass the source
    # wavelet ONCE at stage entry. Syn forward is then naturally band-limited
    # without per-iteration filtering in the autograd path. fwi_workflow-dev
    # filters syn (target='syn'); both yield equivalent gradients for a
    # linear wave equation.
    bp_spec = stage.bandpass
    if bp_spec is not None and getattr(bp_spec, "target", "syn") == "wavelet" \
            and (rebuilt_wavelet or state.get("_wavelet_bandpass_at") != (
                float(new_dt), int(new_nt), float(bp_spec.lo_hz), float(bp_spec.hi_hz),
                int(bp_spec.order), bp_spec.padtype)):
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
        state["_wavelet_bandpass_at"] = (
            float(new_dt), int(new_nt), float(bp_spec.lo_hz), float(bp_spec.hi_hz),
            int(bp_spec.order), bp_spec.padtype,
        )
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

    # Upload obs to the solver device. The per-iter loop does
    # ``obs[chunk].to(dev)`` which becomes a free no-op once obs is on
    # dev. Full obs is ~1 GB float32 on Marmousi-scale; trivial on a
    # multi-GB GPU and saves ~9 s of repeated H2D per 100 iters.
    obs_np = np.ascontiguousarray(obs_np.astype(np.float32, copy=False))
    obs_t = torch.from_numpy(obs_np)
    on_cuda = _is_cuda_dev(dev)
    if on_cuda:
        # Keep the full obs on GPU only when it fits with headroom. Large
        # field-data obs at fine stages (tens of GB) OOMs the device and can
        # exceed a V100's 32 GB; leave it on CPU and let the per-iter
        # ``obs[chunk].to(dev)`` stream each chunk on demand.
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
    # zero-phase Butterworth from sweep_tasks.preproc — same implementation
    # used by ``_run_fwi_multisource``'s per-iter encoded supershot path.
    # ~30× faster than scipy ``sosfiltfilt`` on Marmousi-scale obs
    # (1 GB float32), drops stage-entry time from ~10 s to <0.5 s.
    # CPU fallback: scipy ``sosfiltfilt`` with the configured padtype.
    if stage.bandpass is not None:
        if obs_t.is_cuda:   # obs may be CPU-resident for large field data (see upload above)
            with torch.no_grad():
                # Chunk the bandpass over the shot axis (dim 0) so each cuFFT
                # plan stays under the 2^31-element limit; a full dense-OBN obs
                # (nshots*nt*nrec) can exceed INT_MAX and trip
                # CUFFT_INVALID_SIZE. Per-shot spectra are independent, so
                # filtering shot-chunks in place is exact.
                _bp_chunk = 64
                if obs_t.shape[0] > _bp_chunk and time_axis_pristine != 0:
                    for _b0 in range(0, obs_t.shape[0], _bp_chunk):
                        _b1 = min(_b0 + _bp_chunk, obs_t.shape[0])
                        obs_t[_b0:_b1] = _bandpass_torch_fft(
                            obs_t[_b0:_b1].contiguous(),
                            lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                            dt=new_dt, order=stage.bandpass.order,
                            axis=time_axis_pristine,
                        )
                else:
                    obs_t = _bandpass_torch_fft(
                        obs_t, lo=stage.bandpass.lo_hz, hi=stage.bandpass.hi_hz,
                        dt=new_dt, order=stage.bandpass.order,
                        axis=time_axis_pristine,
                    )
            _flavor = "GPU FFT (sweep_tasks.preproc.bandpass_torch)"
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
    state["_active_bandpass"] = stage.bandpass
    state["obs"] = obs_t
    state["_stage_wavelet_idx"] = stage_idx

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
    # the old momentum has gone stale).
    force_reset = bool(getattr(stage, "optimizer_reset", False))
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
    # so ``inr_lr_scale`` multiplies on top of that. For the grid path
    # ``inr_lr_scale`` is ignored (no INR).
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


def _shape_for_dh(orig_shape: tuple[int, ...], orig_dh: float, new_dh: float) -> tuple[int, ...]:
    """Pick a new grid shape that preserves physical extent (within rounding)."""
    if abs(orig_dh - new_dh) < 1e-12:
        return tuple(orig_shape)
    ratio = orig_dh / new_dh
    return tuple(max(1, int(round(s * ratio))) for s in orig_shape)


def _resample_vp_tensor(vp: "torch.Tensor", new_shape: tuple[int, ...]) -> "torch.Tensor":
    """Resample a vp tensor between grid resolutions (bilinear for 2-D,
    trilinear for 3-D).

    Returns a fresh leaf tensor (requires_grad=True) — caller is responsible
    for re-initialising the optimizer because Adam state is shape-bound.
    """
    import torch
    import torch.nn.functional as F

    if tuple(vp.shape) == tuple(new_shape):
        return vp.detach().clone().requires_grad_(True)
    if vp.ndim == 2:
        if len(new_shape) != 2:
            raise ValueError(
                f"_resample_vp_tensor: vp is 2-D but new_shape={new_shape} is not."
            )
        mode = "bilinear"
    elif vp.ndim == 3:
        if len(new_shape) != 3:
            raise ValueError(
                f"_resample_vp_tensor: vp is 3-D but new_shape={new_shape} is not."
            )
        mode = "trilinear"
    else:
        raise ValueError(
            f"_resample_vp_tensor: unsupported vp.ndim {vp.ndim} (expected 2 or 3)."
        )
    src = vp.detach().unsqueeze(0).unsqueeze(0)
    dst = F.interpolate(src, size=tuple(new_shape), mode=mode, align_corners=True)
    return dst.squeeze(0).squeeze(0).contiguous().clone().requires_grad_(True)


def _resample_obs_time(obs_np: "np.ndarray", dt_old: float, dt_new: float,
                       *, time_axis: int = -1) -> "np.ndarray":
    """Resample obs along the time axis via sweep_tasks.preproc.resample.resample_time."""
    if abs(dt_old - dt_new) < 1e-12:
        return obs_np
    from sweep_tasks.preproc.resample import resample_time
    return resample_time(obs_np, dt_old, dt_new, axis=time_axis)


def _resample_obs_to_solver_dt(obs_t, dt_segy: float, dt_solver: float,
                               nt_solver: int):
    """Resample a torch obs tensor along the last axis to solver dt + length.

    Mirrors the legacy 3-D CRG FWI runner's per-iter obs prep: drop to
    numpy, run scipy.signal.resample_poly via sweep_tasks.preproc, truncate /
    zero-pad to nt_solver, and ship back to the original device.
    """
    if abs(dt_segy - dt_solver) < 1.0e-12 and obs_t.shape[-1] == nt_solver:
        return obs_t
    import torch
    from sweep_tasks.preproc.resample import resample_time

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


def _bandpass_syn_torch(syn: "torch.Tensor", lo: float, hi: float, dt: float,
                        *, order: int) -> "torch.Tensor":
    """Differentiable bandpass on a synthetic torch tensor.

    Thin wrapper around :func:`sweep_tasks.preproc.filter.bandpass_torch` —
    the same canonical zero-phase GPU FFT Butterworth used by the obs
    stage-entry filter and by the multisource-encoded FWI path. Going
    through one impl across syn, obs, and wavelet ensures the filter
    response is identical and avoids the latent NaN risk of
    ``torchaudio.functional.filtfilt`` without padding on narrow bands.

    Both sweep backends (eager + c, after geophyai 21041c5) emit syn in
    canonical 4-D ``(n, nt, nrec, nchan)`` with time at axis 1, so this
    helper only accepts 4-D and raises on anything else. The historical
    3-D fallback ``(nt, nrec, 1)`` (time at axis 0) is removed because
    (a) sweep no longer produces it, and (b) the layout it assumed
    disagreed with ``sweep_loss.to_canonical``'s 3-D convention
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
