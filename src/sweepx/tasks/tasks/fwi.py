"""Conventional FWI task runner (mixin). Verbatim from runner.py."""
import os
from pathlib import Path
import numpy as np
from sweep_tasks.schemas import FWISpec
from sweep_tasks._helpers.bounds import (
    _apply_bounds,
    _effective_bound,
)
from sweep_tasks._helpers.run_checkpoint import (
    _load_checkpoint,
    _save_checkpoint,
    _zero_top_rows,
)
from sweep_tasks._helpers.data_loading import (
    _load_segy_index_payload,
    _load_segy_single_file_payload,
)
from sweep_tasks._helpers.dd import (
    _compute_local_window,
    _rebase_geometry_to_window,
)
from sweep_tasks._helpers.geometry import (
    _build_geometry_2d,
    _segy_geometry_to_grid_indices,
)
from sweep_tasks._helpers.illumination import (
    _accumulate_illumination,
    _apply_illumination_precond,
)
from sweep_tasks._helpers.loss import (
    _loss_sum,
    _mask_chunk,
    set_loss_dt,
)
from sweep_tasks._helpers.metadata import _dump_run_metadata
from sweep_tasks._helpers.model import (
    _get_equation_class,
    _load_model_tensor,
    _model_names_for_equation,
)
from sweep_tasks._helpers.optimizer import (
    _build_optimizer,
    _build_reparam_optimizer,
    _build_scheduler,
    _remember_initial_lrs,
)
def _MASK_WINDOW(spec):
    """True when loss.data_mask_path should be applied MUTE-THEN-MISFIT.

    Off by default so existing ``data_mask_path`` runs (post-hoc per-sample
    weight) stay bit-identical. Needed for ``trace_cosine``, whose per-trace
    value is broadcast over time: weighting it by a time-varying mask only
    rescales the trace, it never windows it."""
    return bool(getattr(getattr(spec, "loss", None), "data_mask_window_mode", False))


def _MASK_WINDOW_TRACE_COSINE(spec):
    """Windowed (mute-then-misfit) mask combined with the trace_cosine misfit."""
    return (_MASK_WINDOW(spec)
            and getattr(getattr(spec, "loss", None), "kind", None) == "trace_cosine")


from sweep_tasks._helpers.plan_apply import (
    _apply_data_plan_to_fwi,
    _apply_model_plan_to_fwi,
    _build_inv_tensors,
    _normalize_fwi_init_models,
)
from sweep_tasks._helpers.plotting import (
    _plot_loss_curve,
    _save_illumination,
)
from sweep_tasks._helpers.reparam import (
    _advance_hash_schedule,
    _build_reparam_net,
    _has_hash_schedule,
)
from sweep_tasks._helpers.solver_build import (
    _build_solver,
    _cfl_check,
    _resolve_modeling_inputs,
)
from sweep_tasks._helpers.stages import (
    _bandpass_syn_torch,
    _normalise_stage_list,
    _prepare_stage,
)
from sweep_tasks._helpers.stop import _GracefulStopper
from sweep_tasks._helpers.util import _apply_seed
from sweep_tasks._helpers.wavelet_build import (
    _build_wavelet,
    _get_wavelet_source_delay_s,
)


def _build_grad_smoother(spec, dev):
    """Return a closure that Gaussian-smooths a vp gradient tensor.

    Separable (depthwise) Gaussian along each spatial axis, reflect-padded so
    the field edges aren't pulled toward zero. Handles 2-D (nz, nx) and 3-D
    (nz, ny, nx) gradients. Sigmas are in grid cells; an axis with sigma<=0 is
    left un-smoothed. Built once; the kernels live on ``dev``.
    """
    import torch
    import torch.nn.functional as F

    def _kernel1d(sigma):
        if sigma is None or sigma <= 0:
            return None
        rad = max(1, int(round(3.0 * sigma)))
        t = torch.arange(-rad, rad + 1, dtype=torch.float32, device=dev)
        k = torch.exp(-0.5 * (t / sigma) ** 2)
        return (k / k.sum())

    kz = _kernel1d(float(spec.sigma_z_cells))
    kx = _kernel1d(float(spec.sigma_x_cells))
    ky = _kernel1d(float(getattr(spec, "sigma_y_cells", 0.0)))
    z_lo = int(getattr(spec, "mask_above_row", -1))   # zero rows < z_lo
    z_hi = int(getattr(spec, "mask_below_row", -1))   # zero rows > z_hi
    mask_taper = int(getattr(spec, "mask_taper_rows", 0))  # cosine taper width

    def _depth_mask_(x):
        # x is (..., nz, ...) with z on dim 0 of the last-2/last-3 image; here
        # the gradient's first axis is always z (nz, nx) or (nz, ny, nx).
        # A hard cutoff makes the gradient pile up at the boundary row; a linear
        # cosine taper over ``mask_taper`` rows spreads it and removes the pile.
        if z_lo > 0:
            if mask_taper > 0:
                r = min(mask_taper, z_lo)
                ramp = torch.linspace(0.0, 1.0, r + 2, device=x.device, dtype=x.dtype)[1:-1]
                x[:z_lo - r] = 0.0
                x[z_lo - r:z_lo] *= ramp.view([-1] + [1] * (x.ndim - 1))
            else:
                x[:z_lo] = 0.0
        if z_hi >= 0:
            if mask_taper > 0:
                r = min(mask_taper, x.shape[0] - z_hi - 1)
                ramp = torch.linspace(1.0, 0.0, r + 2, device=x.device, dtype=x.dtype)[1:-1]
                x[z_hi + 1: z_hi + 1 + r] *= ramp.view([-1] + [1] * (x.ndim - 1))
                x[z_hi + 1 + r:] = 0.0
            else:
                x[z_hi + 1:] = 0.0
        return x

    def _smooth(grad):
        g = grad
        orig_shape = g.shape
        if g.ndim == 2:            # (nz, nx): conv2d over a 1x1x nz x nx image
            x = g[None, None]
            if kz is not None:
                w = kz.view(1, 1, -1, 1); p = kz.numel() // 2
                x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), w)
            if kx is not None:
                w = kx.view(1, 1, 1, -1); p = kx.numel() // 2
                x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), w)
            return _depth_mask_(x.view(orig_shape))
        if g.ndim == 3:            # (nz, ny, nx): conv3d over a 1x1x nz x ny x nx
            x = g[None, None]
            for k, axis in ((kz, 2), (ky, 3), (kx, 4)):
                if k is None:
                    continue
                shp = [1, 1, 1, 1, 1]; shp[axis] = -1
                w = k.view(*shp); p = k.numel() // 2
                pad = [0, 0, 0, 0, 0, 0]
                # F.pad order is (x_l,x_r, y_l,y_r, z_l,z_r) for 3-D last-3 dims
                idx = {2: 4, 3: 2, 4: 0}[axis]
                pad[idx] = p; pad[idx + 1] = p
                x = F.conv3d(F.pad(x, pad, mode="reflect"), w)
            return _depth_mask_(x.view(orig_shape))
        return _depth_mask_(g)

    return _smooth


class FWIRunnerMixin:
    def _run_fwi(self, spec: FWISpec, task_dir: Path):
        # Dispatch the frequency-selection encoded path first: it is
        # plan-free (geometry and data live in the coefficient shards) and
        # shares none of the sampler/prefetch machinery below.
        if (spec.source_encoding is not None
                and spec.source_encoding.enabled
                and spec.source_encoding.mode == "frequency_selection"):
            return self._run_fwi_freqsel(spec, task_dir)
        # Dispatch the OBN 3-D multisource supershot path early — it has
        # its own per-iter SEG-Y reads + UTM→model rotation + source-
        # encoded loop that don't fit the static-(sources, obs)
        # assumptions of the main ``_fwi_train_step`` machinery. Triggered
        # by setting ``obs.plan.sampling`` (a PlanSamplingConfig) on a
        # ``grouping='crg'`` plan.
        _obs_plan = getattr(spec.obs, "plan", None) if spec.obs is not None else None
        if _obs_plan is not None and getattr(_obs_plan, "sampling", None) is not None:
            return self._run_fwi_multisource(spec, task_dir)

        import torch

        from sweep_tasks.runtime import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        if dist_info.is_distributed and spec.optimizer.kind == "lbfgs":
            raise ValueError(
                "LBFGS optimiser is not supported under torchrun in Phase 1 "
                "(its line-search closure semantics conflict with shot-parallel "
                "all-reduce). Use adam or sgd."
            )

        _apply_seed(spec.seed)
        dev = _dist.resolve_dist_device(spec.device, dist_info.local_rank)
        equation_cls = _get_equation_class(spec.physics.equation)

        # Resolved config + run metadata for the standard (non-multisource)
        # FWI path. The CRG/multisource path (`_run_fwi_multisource`) already
        # writes these; mirror it here so every FWI run dir is self-documenting
        # (config_resolved.yaml + run_meta.json: host/CUDA/env/git). rank-0 only.
        if dist_info.local_rank == 0:
            try:
                _dump_run_metadata(spec, task_dir)
                print(f"[run-meta] wrote {task_dir/'config_resolved.yaml'} "
                      f"+ {task_dir/'run_meta.json'}")
            except Exception as meta_err:  # noqa: BLE001
                print(f"[run-meta] dump skipped: {meta_err}")

        # Graceful ctrl-c / SIGTERM: install the handler *before* the heavy
        # setup phase (obs synth, solver build, etc.) so that a signal
        # delivered early just arms the stop flag instead of crashing the
        # process with KeyboardInterrupt. The stage loop checks the flag
        # at each iter boundary; if armed before training even starts,
        # we fall straight through to the final-outputs path.
        stopper = _GracefulStopper()
        stopper.install(label="fwi")
        interrupted = False
        interrupted_at_epoch: int | None = None

        # 1) Init models: accept either init_model (single) or init_models (list).
        init_models = _normalize_fwi_init_models(spec)

        # 2) Shape: grid.shape > first model file shape > first model constant shape.
        if spec.grid.shape is not None:
            shape = tuple(int(v) for v in spec.grid.shape)
        elif init_models[0].constant is not None:
            shape = tuple(int(v) for v in init_models[0].shape)
        else:
            shape = tuple(np.load(init_models[0].path, mmap_mode="r").shape)

        # 3) Build solver, wavelet, geometry.
        #    Gap 1: when data_plan.dt_target_s is set, the obs gets resampled
        #    to that dt — so the solver must run at the same dt or syn/obs
        #    will mismatch. Auto-sync here (preserves total time = dt * nt).
        effective_dt = float(spec.time.dt)
        effective_nt = int(spec.time.nt)
        if spec.data_plan is not None and spec.data_plan.dt_target_s is not None:
            target_dt = float(spec.data_plan.dt_target_s)
            if abs(target_dt - effective_dt) > 1e-12:
                new_nt = int(round(effective_dt * effective_nt / target_dt))
                if dist_info.is_root:
                    print(f"[fwi] data_plan.dt_target_s={target_dt}s → auto-sync "
                          f"solver: dt {effective_dt} -> {target_dt}, nt "
                          f"{effective_nt} -> {new_nt}")
                effective_dt = target_dt
                effective_nt = new_nt
        # Register the solver dt so the cc_traveltime misfit reports the shift in
        # seconds (no-op for every other loss kind).
        set_loss_dt(effective_dt)

        # CFL pre-check from the init_model vmax (cheap mmap peek).
        try:
            ref = init_models[0]
            if ref.path is not None:
                vmax_estimate = float(np.load(ref.path, mmap_mode="r").max())
            elif ref.constant is not None:
                vmax_estimate = float(ref.constant)
            else:
                vmax_estimate = 0.0
            if vmax_estimate > 0 and dist_info.is_root:
                _cfl_check(vmax_estimate, float(spec.grid.dh), effective_dt)
        except FileNotFoundError:
            pass

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, effective_dt, effective_nt, dev
        )
        wavelet = _build_wavelet(
            spec.wavelet, spec.time,
            override_dt=effective_dt, override_nt=effective_nt,
        )
        # Shared cache so SEG-Y-backed geometry + obs don't scan the file twice.
        segy_cache: dict = {}
        # ``obs.plan`` with ``sampling=None`` → conventional single-source FWI
        # materialised from a SeismicPlan (CSG or CRG, 2-D/3-D field data). The
        # OBN supershot/encoding path (sampling != None) is dispatched earlier
        # to ``_run_fwi_multisource``; here we build ONE static dataset and run
        # the normal ``_fwi_train_step`` loop (per-shot gradient accumulation).
        _obs_plan_mat = (
            spec.obs is not None and spec.obs.plan is not None
            and spec.obs.plan.sampling is None
        )
        model_plan_cropped: dict | None = None
        model_plan_keep: "np.ndarray | None" = None
        if _obs_plan_mat:
            from sweep_tasks.plan_materialize import materialize_plan_dataset
            _mat = materialize_plan_dataset(
                spec, init_models,
                effective_dt=effective_dt, effective_nt=effective_nt,
                source_delay_s=_get_wavelet_source_delay_s(spec.wavelet),
                verbose=dist_info.is_root,
            )
            sources = _mat["sources"]
            receivers = _mat["receivers"]
            obs = torch.from_numpy(_mat["obs"])
            shape = tuple(_mat["shape"])
            model_plan_cropped = _mat["cropped_models"] or None
            nshots = int(_mat["nshots"])
            # Rebuild solver at the materialised (cropped) grid.
            solver = _build_solver(
                spec.physics, spec.backend, shape, spec.grid.dh,
                effective_dt, effective_nt, dev,
            )
            # Mark obs as already at solver dt so the pristine_dt detector
            # below does not schedule a redundant per-stage resample.
            segy_cache["__materialized_plan__"] = {"dt_s": float(_mat["native_dt"])}
            if dist_info.is_root:
                print(f"[fwi] obs.plan materialised (single-source): "
                      f"nshots={nshots} nrec={_mat['nrec']} shape={shape}")
        else:
            sources, receivers = _build_geometry_2d(
                spec.geometry, shape, dh=spec.grid.dh, segy_cache=segy_cache,
            )
            nshots = int(sources.shape[0])

            # 3c) Gap 2 — apply model_plan if set: crop every loaded model array,
            # drop out-of-window sources, rebase geometry indices to the crop.
            if spec.model_plan is not None:
                loaded_originals = []
                for ref in init_models:
                    if ref.constant is not None:
                        loaded_originals.append(
                            (ref.name, np.full(tuple(ref.shape), float(ref.constant), dtype=np.float32))
                        )
                    else:
                        loaded_originals.append(
                            (ref.name, np.load(ref.path).astype(np.float32))
                        )
                model_plan_cropped, sources, receivers, model_plan_keep = _apply_model_plan_to_fwi(
                    spec, loaded_originals, sources, receivers, spec.grid.dh,
                )
                new_shape = next(iter(model_plan_cropped.values())).shape
                if dist_info.is_root:
                    print(f"[model_plan] vp shape {shape} -> {new_shape}, "
                          f"sources kept: {int((model_plan_keep is None) or model_plan_keep.sum())}"
                          f"/{nshots}")
                shape = tuple(new_shape)
                nshots = int(sources.shape[0])
                # Rebuild solver at the cropped grid.
                solver = _build_solver(
                    spec.physics, spec.backend, shape, spec.grid.dh,
                    effective_dt, effective_nt, dev,
                )

        # 4) Build inv_tensors (ordered to match equation.MODEL_SPECS).
        inv_in_order, inv_by_name, required_names = _build_inv_tensors(
            init_models, dev, equation_cls,
            overrides=model_plan_cropped,
        )
        if dist_info.is_root:
            print(f"[fwi] shape={shape} nshots={nshots} inverted_models={required_names} "
                  f"world_size={dist_info.world_size}")

        # 4b) Optional NN reparameterization of vp (sweep-nn).
        # When active, vp is rendered each forward by a hash-encoded SIREN
        # whose parameters become the optimization variables. The vp leaf
        # tensor in inv_by_name is repurposed as a buffer holding the
        # current rendered vp (refreshed after each optimizer step) so
        # snapshots / checkpoints see the right values.
        reparam_net = None
        pristine_base_vp = None
        if spec.reparam is not None and "vp" in inv_by_name:
            reparam_net = _build_reparam_net(
                spec.reparam, inv_by_name["vp"], _effective_bound(spec.model_bounds, "vp"),
            )
            # Keep the pristine base around for stage transitions — at each
            # new stage we bilinear-resample THIS to the new shape and feed
            # the network. The network's learned parameters carry over.
            pristine_base_vp = inv_by_name["vp"].detach().clone()
            # Replace inv_by_name["vp"] with the rendered velocity — and
            # turn off requires_grad on it so the optimizer never sees it.
            with torch.no_grad():
                rendered = reparam_net().detach()
            inv_by_name["vp"] = rendered
            inv_in_order = [
                rendered if n == "vp" else inv_by_name[n] for n in required_names
            ]
            if dist_info.is_root:
                n_params = sum(p.numel() for p in reparam_net.parameters())
                print(f"[fwi] reparam=velocity_inr  net_params={n_params:,}  "
                      f"lr={spec.reparam.lr:.3e}  hash_enabled={spec.reparam.hash.enabled}")

        # 5) Generate or load obs. Every rank does this independently (deterministic).
        # When model_plan dropped shots, the obs loader still produces the
        # ORIGINAL shot count for npy / SEG-Y paths (they don't know about the
        # crop). Generate obs at the original nshots first, then mask.
        # (The obs.plan single-source path already produced ``obs`` above.)
        if not _obs_plan_mat:
            if model_plan_keep is not None and (
                spec.obs.npy_path is not None or spec.obs.segy is not None
                or spec.obs.segy_index is not None
            ):
                original_nshots = int(model_plan_keep.size)
                obs = self._fwi_generate_obs(
                    spec, equation_cls, solver, wavelet,
                    sources, receivers, shape, dev, original_nshots,
                    segy_cache=segy_cache,
                )
                obs = obs[model_plan_keep]
            else:
                obs = self._fwi_generate_obs(
                    spec, equation_cls, solver, wavelet,
                    sources, receivers, shape, dev, nshots,
                    segy_cache=segy_cache,
                )

            # 5b) Apply data_plan (and reject unimplemented model_plan) — see
            # _apply_data_plan_to_fwi for the supported subset. Updates
            # `nshots` if shot subsetting kicked in.
            sources, receivers, obs = _apply_data_plan_to_fwi(
                spec, sources, receivers, obs, spec.grid.dh, spec.time.dt, spec.time.nt, dev,
            )
            nshots = int(sources.shape[0])

        # 6) Optimizer + scheduler.
        total_epochs = (sum(s.epochs for s in spec.stages)
                        if spec.stages else int(spec.epochs))
        if reparam_net is not None:
            # In reparam mode the optimizer trains the network's parameters,
            # not the vp tensor. The top-level optimizer.lr is meaningful
            # for grid-FWI (~25) but completely wrong for SIREN/hash params
            # (~1e-4); we use spec.reparam.lr instead. The kind/betas/eps
            # still come from spec.optimizer.
            optimizer = _build_reparam_optimizer(
                spec.optimizer, reparam_net.parameters(), float(spec.reparam.lr),
            )
        else:
            optimizer = _build_optimizer(spec.optimizer, inv_by_name, required_names)
        scheduler = _build_scheduler(spec.scheduler, optimizer, total_epochs)
        initial_lrs = _remember_initial_lrs(optimizer)

        # Smoothing prior (TVPrior) — applied to the velocity gradient each
        # iteration. The OBN multisource (encoded) path builds + applies this;
        # the single-source path historically dropped it (smooth_regularization
        # was a silent no-op), leaving the per-shot gradient unregularised
        # (high-wavenumber speckle). Build it here and thread it into
        # _fwi_train_step so conventional FWI matches the encoded run's
        # regularisation.
        tv_prior = None
        smooth_weight = 0.0
        _smooth_spec = getattr(spec, "smooth_regularization", None)
        if _smooth_spec is not None and float(_smooth_spec.weight) > 0.0:
            from sweep_nn import TVPrior
            tv_prior = TVPrior(
                order=_smooth_spec.order,
                x_weight=float(_smooth_spec.x_weight),
                y_weight=float(_smooth_spec.y_weight),
                z_weight=float(_smooth_spec.z_weight),
                velocity_scale_m_s=float(_smooth_spec.velocity_scale_m_s),
            )
            smooth_weight = float(_smooth_spec.weight)
            if dist_info.is_root:
                print(f"[fwi] smooth_regularization (TVPrior) ON: "
                      f"weight={smooth_weight:.2e} order={_smooth_spec.order}")

        # Plug-and-play DDPM diffusion prior (sweep-nn). Built once here and
        # threaded into _fwi_train_step, mirroring tv_prior: a scalar
        # ``weight * red_loss(vp)`` (or sds_loss) whose .backward() accumulates
        # onto the data gradient before the optimizer step. RED is deterministic
        # (DiffPIR/DDIM) so it is DDP-safe added post all-reduce; SDS is reseeded
        # identically per rank. Applied every ``every`` steps from ``start_band``.
        diff_prior = None
        diff_weight = 0.0
        diff_weight_mode = "relative"
        diff_kind = "red"
        diff_every = 1
        diff_start_band = 0
        diff_sds_t = (0.02, 0.5)
        _diff_spec = getattr(spec, "diffusion_prior", None)
        if _diff_spec is not None and getattr(_diff_spec, "enabled", True) \
                and float(_diff_spec.weight) > 0.0:
            from sweep_nn.diffusion import DiffusionVelocityPrior
            diff_prior = DiffusionVelocityPrior(
                ckpt_path=str(_diff_spec.ckpt_path), device=str(dev),
                mode=_diff_spec.mode, strength=float(_diff_spec.strength),
                ddim_steps=int(_diff_spec.ddim_steps), patch=int(_diff_spec.patch),
                stride=int(_diff_spec.stride), vmin=_diff_spec.vmin,
                vmax=_diff_spec.vmax, use_ema=bool(_diff_spec.use_ema),
            )
            diff_weight = float(_diff_spec.weight)
            diff_weight_mode = _diff_spec.weight_mode
            diff_kind = _diff_spec.kind
            diff_every = int(_diff_spec.every)
            diff_start_band = int(_diff_spec.start_band)
            diff_sds_t = (float(_diff_spec.sds_t_lo), float(_diff_spec.sds_t_hi))
            if dist_info.is_root:
                print(f"[fwi] diffusion_prior ({diff_kind}, {diff_weight_mode}) ON: "
                      f"weight={diff_weight:.3g} mode={_diff_spec.mode} "
                      f"strength={_diff_spec.strength} every={diff_every} "
                      f"start_band={diff_start_band} "
                      f"vmin={diff_prior.vmin:.0f} vmax={diff_prior.vmax:.0f} "
                      f"ckpt={_diff_spec.ckpt_path.name}")

        # Gaussian gradient smoother (tomographic preconditioner). Built once as
        # separable 1-D kernels; applied to the grid-mode vp gradient each step.
        _gsmooth_spec = getattr(spec, "grad_smooth", None)
        _gsmooth = None
        if _gsmooth_spec is not None and bool(getattr(_gsmooth_spec, "enabled", True)):
            _gsmooth = _build_grad_smoother(_gsmooth_spec, dev)
            if dist_info.is_root:
                print(f"[fwi] grad_smooth (Gaussian) ON: sigma_z="
                      f"{_gsmooth_spec.sigma_z_cells} sigma_x="
                      f"{_gsmooth_spec.sigma_x_cells} sigma_y="
                      f"{_gsmooth_spec.sigma_y_cells} every={_gsmooth_spec.every}")

        # 7) Resume from checkpoint if requested. Rank 0 loads + broadcasts state.
        # Two modes:
        #   * ``spec.resume_from``: load from a *different* task_id under
        #     output_dir (the original cross-run continuation path).
        #   * ``spec.resume``: auto-resume from THIS task_dir's checkpoint
        #     if one is on disk. Pairs with ``ctrl-c → re-run same yaml``.
        #     Silent no-op when no checkpoint is found.
        losses: list[float] = []
        start_epoch = 0
        resume_src: str | None = None
        if spec.resume_from:
            resume_src = spec.resume_from
            ckpt_dir = Path(spec.output_dir).expanduser() / spec.resume_from
        elif spec.resume and (task_dir / "checkpoint.pt").exists():
            resume_src = task_dir.name
            ckpt_dir = task_dir
        else:
            ckpt_dir = None
        if ckpt_dir is not None:
            if dist_info.is_root:
                ckpt = _load_checkpoint(ckpt_dir)
            else:
                ckpt = None
            ckpt = _dist.broadcast_object(ckpt, dist_info, src=0)
            for name, t in inv_by_name.items():
                t.data.copy_(ckpt["models"][name].to(dev))
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            losses = list(ckpt.get("losses", []))
            start_epoch = int(ckpt["epoch"]) + 1
            if dist_info.is_root:
                torch.set_rng_state(ckpt["torch_rng"])
                np.random.set_state(ckpt["numpy_rng"])
                print(f"[fwi] resumed from '{resume_src}' at epoch {start_epoch}")

        # 8) Multi-stage training loop.
        stages = _normalise_stage_list(spec)
        out_dir = task_dir / "output"
        snapshots_dir = out_dir / "epochs"
        if dist_info.is_root:
            snapshots_dir.mkdir(exist_ok=True)
        _dist.barrier(dist_info)
        epoch_global = start_epoch

        # Build the pristine snapshot used to re-derive each stage's obs
        # (so per-stage bandpass / dt / receiver-mask don't compound).
        # Canonical layout for both backends: (nshots, nt, nrec[, nfield]).
        # 4-D shapes carry the trailing channel; 3-D obs lacks it (rare,
        # e.g. user-supplied npy without channel axis).
        from sweep_io.geometry import PhysicalGeometry
        pristine_time_axis = -3 if obs.ndim >= 4 else -2
        pristine_recv_axis = -2 if obs.ndim >= 4 else -1
        pristine_obs_np = (obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor)
                           else np.asarray(obs))
        # Multi-rank memory share: when the runner is torchrun-launched
        # (world_size > 1), every rank otherwise holds its own ~per-shot
        # × per-trace × nt float32 copy of pristine_obs_np (no slicing
        # at this point). For Viking-scale 2-D obs this is ~2 GB/rank ×
        # 8 ranks = 16 GB JUST for the pristine snapshot, plus transient
        # bandpass/resample doubles during stage transitions, easily
        # OOMs at finer stages. Mitigate by writing the snapshot once
        # (rank 0) to a tmpfile under /dev/shm and re-opening it via
        # ``np.load(..., mmap_mode='r')`` everywhere — Linux shares the
        # file's pages across processes through the page cache, so the
        # per-rank RSS attribution stays tiny.
        if dist_info.is_distributed and pristine_obs_np.nbytes > 64 * 1024 * 1024:
            import os
            import tempfile

            jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
            shm_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
            shared_path = shm_dir / f"sweep_tasks_pristine_obs_{jid}.npy"
            if dist_info.is_root:
                np.save(shared_path, pristine_obs_np)
            _dist.barrier(dist_info)
            try:
                pristine_obs_np = np.load(shared_path, mmap_mode="r")
                if dist_info.is_root:
                    print(f"[fwi] pristine_obs_np ({pristine_obs_np.nbytes / (1<<30):.2f} GB) "
                          f"shared via {shared_path} (mmap'd by every rank)")
            except Exception as err:  # noqa: BLE001
                if dist_info.is_root:
                    print(f"[fwi] pristine_obs_np shared-mmap fallback "
                          f"failed ({err}); each rank keeps its own copy.")
        # Pristine physical geometry: when SEG-Y was loaded, prefer the
        # ORIGINAL un-rounded physical positions (in meters from SEG-Y
        # headers) over the dh-quantized versions. This is critical for
        # ``dedupe=True`` workflows because at finer stages we want
        # ``pg.to_grid(new_dh)`` to use the original 25 m spacing — not
        # 75 m-quantized positions which would collapse identically at
        # every dh ≥ initial.
        pristine_physical_geom = None
        seg_geom_kinds = ("from_segy_headers", "from_segy_index")
        if getattr(spec.geometry, "kind", None) in seg_geom_kinds and segy_cache:
            # The geometry resolver populated segy_cache; grab the payload
            # back and use its original physical_geometry (un-rounded).
            for payload in segy_cache.values():
                pg_orig = payload.get("physical_geometry") if isinstance(payload, dict) else None
                if pg_orig is not None and hasattr(pg_orig, "sources_xyz_m"):
                    pristine_physical_geom = PhysicalGeometry(
                        sources_xyz_m=np.asarray(pg_orig.sources_xyz_m, dtype="float64"),
                        receivers_xyz_m=np.asarray(pg_orig.receivers_xyz_m, dtype="float64"),
                        dt=effective_dt, nt=effective_nt,
                    )
                    break
        if pristine_physical_geom is None:
            pristine_physical_geom = PhysicalGeometry(
                sources_xyz_m=sources.astype("float64") * float(spec.grid.dh),
                receivers_xyz_m=receivers.astype("float64") * float(spec.grid.dh),
                dt=effective_dt, nt=effective_nt,
            )

        # ``pristine_dt`` is the sampling interval of the obs we just snapped
        # — NOT the solver's dt. For SEG-Y obs the runner did NOT resample
        # at load time (per-stage helpers do the dt-sync), so pristine_dt
        # must reflect the SEG-Y file's native dt; otherwise the first
        # stage's ``_resample_obs_time(obs, pristine_dt, new_dt)`` mis-
        # interprets the time axis and either decimates valid samples or
        # smears the trace via fake up-sampling. ``data_plan.dt_target_s``
        # already resampled the obs to ``effective_dt`` upstream, so in
        # that path we keep the effective_dt label.
        obs_native_dt = float(effective_dt)
        if (spec.data_plan is None or spec.data_plan.dt_target_s is None) and (
            (getattr(spec.obs, "segy", None) is not None
             or getattr(spec.obs, "segy_index", None) is not None
             or getattr(spec.obs, "plan", None) is not None)
        ) and segy_cache:
            for payload in segy_cache.values():
                if isinstance(payload, dict) and "dt_s" in payload:
                    obs_native_dt = float(payload["dt_s"])
                    break
            if abs(obs_native_dt - float(effective_dt)) > 1e-12 and dist_info.is_root:
                print(f"[fwi] obs native dt={obs_native_dt}s differs from solver "
                      f"dt={effective_dt}s; pristine_dt set to SEG-Y native so "
                      f"per-stage resample lines up. Set spec.data_plan.dt_target_s "
                      f"to force an upfront resample.")

        # Per-stage mutable state. Seeded with the post-step-7 baseline.
        # Per-stage helpers consult / mutate this dict; everything the
        # train step needs lives in here.
        state: dict = {
            "dh": float(spec.grid.dh), "pristine_dh": float(spec.grid.dh),
            "dt": float(effective_dt),
            "nt": int(effective_nt),
            "shape": tuple(shape), "pristine_shape": tuple(shape),
            "sources": sources, "receivers": receivers, "nshots": int(nshots),
            "inv_in_order": inv_in_order, "inv_by_name": inv_by_name,
            "solver": solver, "wavelet": wavelet,
            "optimizer": optimizer, "initial_lrs": initial_lrs,
            # Reparam (sweep-nn): the network + pristine base for stage
            # transitions. Both None when reparam is not configured.
            "reparam_net": reparam_net,
            "pristine_base_vp": pristine_base_vp,
            "obs": obs, "batchsize": int(spec.batchsize),
            # pristine sources for re-derivation
            "pristine_obs_np": pristine_obs_np,
            "pristine_dt": obs_native_dt,
            "pristine_obs_time_axis": pristine_time_axis,
            "pristine_obs_receiver_axis": pristine_recv_axis,
            "pristine_physical_geom": pristine_physical_geom,
            # Inherit dedupe / dedup_method from geometry spec when available.
            # The default (True / nearest) matches the top-level resolver.
            "dedupe_grid_snap": getattr(spec.geometry, "dedupe", True),
            "dedup_method": getattr(spec.geometry, "dedup_method", "nearest"),
            # QC: snapshot the initial vp at coarsest grid for Δvp plots.
            # The QC orchestrator resamples on demand to the current stage.
            "qc_initial_vp": inv_by_name["vp"].detach().cpu().clone()
                if "vp" in inv_by_name else None,
        }

        # QC setup. The qc_dir + cadence guard live outside the stage loop
        # so the per-epoch hook is just a single function call.
        qc_dir = task_dir / "qc"
        qc_enabled = spec.qc is not None and spec.qc.every_n_epochs > 0
        stage_epoch_boundaries: list[int] = []  # cumulative epoch indices
        # coarse-to-fine hash schedule (reparam.hash.c2f): open encoder levels
        # coarse->fine over the WHOLE run (epoch_global/total_epochs), so a
        # single-network multi-band run (e.g. ±1 encoding, fixed grid) grows
        # resolution across bands. No-op when c2f disabled or no CoarseToFine
        # encoder. Per-stage frequency-band c2f is not needed here: one net,
        # one continuous schedule.
        _c2f = (getattr(getattr(spec.reparam, "hash", None), "c2f", None)
                if spec.reparam is not None else None)
        _c2f_on = bool(_c2f is not None and bool(_c2f.enabled))
        if _c2f_on and dist_info.is_root:
            print(f"[fwi] c2f whole-run: base_levels={_c2f.base_levels} "
                  f"ramp={_c2f.ramp} warmup={_c2f.warmup} "
                  f"ramp_end={_c2f.ramp_end} final_levels={_c2f.final_levels}")

        # If a signal landed during setup, skip training entirely and
        # fall through to final-outputs (which will write whatever state
        # we have — typically just the init model).
        if stopper.should_stop(dist_info):
            interrupted = True
            interrupted_at_epoch = epoch_global - 1 if epoch_global > 0 else None
            if dist_info.is_root:
                print(f"[fwi] signal received during setup; skipping training loop.")

        for stage_idx, stage in enumerate(stages):
            if interrupted:
                break
            stage_offset = sum(s.epochs for s in stages[:stage_idx])
            already_done_in_stage = max(0, epoch_global - stage_offset)
            remaining = stage.epochs - already_done_in_stage
            if remaining <= 0:
                continue

            # Free GPU memory held by the previous stage's solver / state
            # before allocating the next one — boundary buffers / wavefield
            # caches can hold hundreds of MB to many GB.
            if stage_idx > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            _prepare_stage(
                spec=spec, stage=stage, state=state,
                equation_cls=equation_cls, required_names=required_names,
                dev=dev, dist_info=dist_info, stage_idx=stage_idx,
            )

            # Record stage-end epoch index for the final loss-curve QC
            # (shading boundaries). Use the absolute epoch counter.
            stage_epoch_boundaries.append(epoch_global + remaining)

            if dist_info.is_root:
                print(f"[fwi] stage {stage_idx} ({remaining}/{stage.epochs} epochs) "
                      f"lr_scale={stage.lr_scale} wavelet_overridden={stage.wavelet is not None}")
            # Build the per-batch local-window context once per stage and
            # pass it through; the chunk dispatcher inside _fwi_train_step
            # uses it to compute fresh windows + cached solvers per chunk.
            local_window_ctx = None
            if (spec.local_model_window is not None
                    and spec.local_model_window.enabled):
                local_window_ctx = {
                    "spec": spec.local_model_window,
                    "shape": state["shape"],
                    "dh": state["dh"],
                    "dt": state["dt"],
                    "nt": state["nt"],
                    "solver_cache": state.setdefault("local_solver_cache", {}),
                }
            # Per-stage syn bandpass: matches obs's per-stage bandpass so the
            # loss compares like-for-like. fwi_workflow-dev does the same via
            # ``_apply_torch_filter`` on syn (sweep_fwi.py:1040). When the
            # stage's bandpass ``target == 'wavelet'``, the wavelet was
            # pre-filtered at stage entry and syn is naturally band-limited;
            # we skip the per-iteration syn filter to save autograd cost.
            stage_bandpass = stage.bandpass
            if stage_bandpass is not None \
                    and getattr(stage_bandpass, "target", "syn") == "wavelet":
                stage_bandpass = None  # already applied to wavelet
            for _ in range(remaining):
                # coarse-to-fine: advance the encoder level mask by whole-run
                # progress before rendering/backprop this epoch (no-op unless
                # reparam.hash.c2f is on and the encoder is CoarseToFine).
                if _c2f_on:
                    _net = state.get("reparam_net")
                    if _has_hash_schedule(_net):
                        _act = _advance_hash_schedule(
                            _net, epoch_global / max(1, total_epochs - 1),
                            _c2f, state.get("optimizer"))
                        if _act is not None and dist_info.is_root and (
                                epoch_global < 3 or epoch_global % 20 == 0):
                            print(f"[fwi] c2f epoch {epoch_global}: active levels "
                                  f"{_act[0]:.2f}/{_act[1]}")
                # Only save the obs/syn snapshot for QC when QC actually fires
                # this iter. Without this gate every iter pays a ~250 ms D2H
                # copy of syn+obs_chunk (~300 MB) into a dict that's discarded
                # next iter.
                _take_qc_snap = qc_enabled and dist_info.is_root and (
                    epoch_global % spec.qc.every_n_epochs == 0
                    or epoch_global == total_epochs - 1
                )
                # Diffusion prior fires every ``diff_every`` steps, only from
                # stage ``diff_start_band`` on (let the low bands invert freely
                # before projecting onto the — possibly OOD — prior manifold).
                _diff_apply = (
                    diff_prior is not None
                    and (epoch_global % diff_every == 0)
                    and (stage_idx >= diff_start_band)
                )
                loss_value = self._fwi_train_step(
                    spec, state["solver"], state["wavelet"],
                    state["sources"], state["receivers"],
                    state["inv_in_order"], state["inv_by_name"],
                    state["obs"], state["optimizer"],
                    state["nshots"], dev,
                    dist_info=dist_info,
                    stage_batchsize=state["batchsize"],
                    reparam_net=state.get("reparam_net"),
                    local_window_ctx=local_window_ctx,
                    syn_bandpass=stage_bandpass,
                    stage_dt=state["dt"],
                    tv_prior=tv_prior,
                    smooth_weight=smooth_weight,
                    diff_prior=diff_prior,
                    diff_weight=diff_weight,
                    diff_weight_mode=diff_weight_mode,
                    diff_kind=diff_kind,
                    diff_apply=_diff_apply,
                    diff_sds_t=diff_sds_t,
                    diff_seed=epoch_global,
                    grad_smoother=_gsmooth,
                    state_for_dump=state,
                    take_qc_snapshot=_take_qc_snap,
                )
                losses.append(loss_value)
                if scheduler is not None:
                    scheduler.step()
                # In reparam mode the network's render-time clamp already
                # enforces vp bounds; only clamp the non-reparam tensors.
                reparam_skip = {"vp"} if state.get("reparam_net") is not None else set()
                _apply_bounds(state["inv_by_name"], spec.model_bounds, skip_names=reparam_skip)
                # Refresh inv_by_name["vp"] from the network so snapshots /
                # checkpoints save the actual current rendered velocity.
                if state.get("reparam_net") is not None:
                    with torch.no_grad():
                        rendered = state["reparam_net"]().detach()
                    state["inv_by_name"]["vp"] = rendered
                    state["inv_in_order"] = [
                        rendered if n == "vp" else state["inv_by_name"][n]
                        for n in required_names
                    ]
                if dist_info.is_root:
                    print(f"[fwi] stage {stage_idx} epoch {epoch_global:04d} loss={loss_value:.6e}")

                snapshot_now = (epoch_global % spec.show_every == 0
                                or epoch_global == total_epochs - 1)
                if snapshot_now and dist_info.is_root:
                    for name, t in state["inv_by_name"].items():
                        np.save(snapshots_dir / f"{name}_epoch_{epoch_global:04d}.npy",
                                t.detach().cpu().numpy())
                    if spec.save_illumination:
                        _save_illumination(state["solver"], snapshots_dir, epoch_global)

                # QC artefacts (vp / Δvp / gradient / shot gather) — only
                # at the configured cadence and only on the rank 0 process.
                if qc_enabled and dist_info.is_root and (
                    epoch_global % spec.qc.every_n_epochs == 0
                    or epoch_global == total_epochs - 1
                ):
                    self._run_epoch_qc_safe(
                        spec=spec, state=state, qc_dir=qc_dir,
                        epoch=epoch_global, dev=dev,
                    )

                if dist_info.is_root:
                    _save_checkpoint(task_dir, {
                        "epoch": epoch_global,
                        "models": {name: t.detach().cpu().clone()
                                   for name, t in state["inv_by_name"].items()},
                        "optimizer": state["optimizer"].state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "losses": losses,
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                    })
                # Honor graceful-stop request now that the checkpoint for the
                # just-completed epoch is on disk. The next ``sweep-tasks run``
                # with ``resume: true`` will pick up at epoch_global + 1.
                if stopper.should_stop(dist_info):
                    interrupted = True
                    interrupted_at_epoch = epoch_global
                    if dist_info.is_root:
                        print(f"[fwi] stopping after epoch {epoch_global} "
                              f"(checkpoint.pt saved). Re-run the same YAML "
                              f"with `resume: true` to continue.")
                    epoch_global += 1
                    break
                epoch_global += 1
            if interrupted:
                break

        stopper.uninstall()

        # Stage loop done — keep references aligned for the final-outputs step.
        inv_by_name = state["inv_by_name"]
        inv_in_order = state["inv_in_order"]
        solver = state["solver"]
        sources = state["sources"]
        receivers = state["receivers"]

        # 9) Final outputs (rank 0 only).
        artifacts: list[Path] = []
        if dist_info.is_root:
            for name, t in inv_by_name.items():
                p = out_dir / f"inverted_{name}.npy"
                np.save(p, t.detach().cpu().numpy())
                artifacts.append(p)
            loss_path = out_dir / "loss.npy"
            np.save(loss_path, np.array(losses, dtype=np.float64))
            artifacts.append(loss_path)
            try:
                artifacts.append(_plot_loss_curve(losses, out_dir / "loss.png", title="FWI Loss"))
            except Exception as plot_err:  # noqa: BLE001
                print(f"[fwi] loss plot skipped: {plot_err}")
            # QC final products (loss curve with stage shading etc.)
            if spec.qc is not None:
                try:
                    from sweep_tasks.qc import save_final_qc
                    final_qc = save_final_qc(
                        qc_spec=spec.qc, qc_dir=qc_dir, losses=losses,
                        stage_epoch_boundaries=stage_epoch_boundaries,
                    )
                    artifacts.extend(final_qc)
                except Exception as qc_err:  # noqa: BLE001
                    print(f"[fwi] final QC skipped: {qc_err}")
        _dist.barrier(dist_info)

        summary = {
            "epochs": total_epochs,
            "epochs_completed": len(losses),
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "models_inverted": list(required_names),
            "num_stages": len(stages),
            "resumed_from": resume_src,
            "interrupted": interrupted,
            "interrupted_at_epoch": interrupted_at_epoch,
            "world_size": dist_info.world_size,
        }
        return artifacts, summary

    def _fwi_generate_obs(self, spec, equation_cls, solver, wavelet,
                          sources, receivers, shape, dev, nshots,
                          *, segy_cache: dict | None = None):
        """Return obs tensor in canonical layout (nshots, nt, nrec, 1)
        for both backends (geophyai 21041c5 aligned the c-backend record
        output). Always CPU-resident.

        For SEG-Y sources, when the same file / index was already scanned by
        the geometry resolver, the cached obs is reused without a second read.
        """

        import torch

        obs_spec = spec.obs
        if obs_spec.npy_path is not None:
            obs_np = np.load(obs_spec.npy_path).astype(np.float32)
            obs = torch.from_numpy(obs_np)
            if obs.shape[0] != nshots:
                raise ValueError(
                    f"obs.npy_path first-axis size {obs.shape[0]} != nshots {nshots}."
                )
            return obs

        def _adapt_segy_obs_to_backend(obs_nrec_nt: np.ndarray) -> "torch.Tensor":
            """SEGYReader always returns (nshots, nrec, nt). Match sweep's
            canonical syn layout shared by both backends after geophyai
            commit 21041c5: (nshots, nt, nrec, 1).
            """
            arr = obs_nrec_nt.astype(np.float32, copy=False)
            arr = np.ascontiguousarray(arr.transpose(0, 2, 1))[..., None]
            return torch.from_numpy(arr)

        if obs_spec.segy is not None:
            cfg = obs_spec.segy
            payload = _load_segy_single_file_payload(
                cfg.path,
                byte_map=cfg.byte_map,
                source_depth_m_override=cfg.source_depth_m_override,
                receiver_depth_m_override=cfg.receiver_depth_m_override,
                coord_scalar_override=cfg.coord_scalar_override,
                shot_ids=cfg.shot_ids,
                cache=segy_cache,
            )
            # Initial parse: keep ALL receivers (dedupe=False). Runner-side
            # per-stage dedupe (when geometry.dedupe=True) operates on the
            # pristine 120-receiver obs, allowing finer stages to access the
            # full receiver count even if a coarser stage drops some.
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=False, dedup_method="nearest",
            )
            if obs_aligned.shape[0] != nshots:
                raise ValueError(
                    f"obs.segy gave {obs_aligned.shape[0]} shots but geometry-derived nshots={nshots}."
                )
            return _adapt_segy_obs_to_backend(obs_aligned)

        if obs_spec.segy_index is not None:
            cfg = obs_spec.segy_index
            payload = _load_segy_index_payload(
                cfg.index_path,
                shot_ids=cfg.shot_ids,
                cache=segy_cache,
                lazy=cfg.lazy,
                coalesce_gap=cfg.coalesce_gap,
            )
            # Initial parse: keep ALL receivers (per-stage dedupe handled
            # later by ``_prepare_stage``).
            _, _, obs_aligned = _segy_geometry_to_grid_indices(
                payload, float(spec.grid.dh), dedupe=False, dedup_method="nearest",
            )
            if obs_aligned.shape[0] != nshots:
                raise ValueError(
                    f"obs.segy_index gave {obs_aligned.shape[0]} shots but "
                    f"geometry-derived nshots={nshots}."
                )
            return _adapt_segy_obs_to_backend(obs_aligned)

        # ``obs.plan`` never reaches here: FWI materialises it earlier
        # (``_obs_plan_mat``) or dispatches to ``_run_fwi_multisource``, and
        # RTM opens its own PlanReader. Guard the invariant so a future
        # re-route fails loudly instead of falling into the synthetic branch.
        if obs_spec.plan is not None:
            raise AssertionError(
                "_fwi_generate_obs reached with obs.plan set — the plan paths "
                "produce obs themselves (plan_materialize / _run_fwi_multisource "
                "/ rtm PlanReader). This call site should not be routed here."
            )

        # synthetic: collect true models in equation MODEL_SPECS order
        if obs_spec.synthetic_from_models is not None:
            true_refs = list(obs_spec.synthetic_from_models)
        else:
            true_refs = [obs_spec.synthetic_from]
        required = _model_names_for_equation(equation_cls)
        by_name = {m.name: m for m in true_refs}
        if set(by_name) != set(required):
            raise ValueError(
                f"obs.synthetic_from* model names {sorted(by_name)} must match "
                f"equation '{equation_cls.__name__}' models {required}."
            )
        # When model_plan is active, crop the synthetic-from "true" tensors
        # the same way the init vp was cropped, so the solver (built at the
        # cropped shape) gets matching inputs.
        if spec.model_plan is not None:
            from sweep_io.plan import ModelPlan, apply_model_plan
            mp = ModelPlan(
                x_window_m=spec.model_plan.x_window_m,
                y_window_m=spec.model_plan.y_window_m,
                z_window_m=spec.model_plan.z_window_m,
                drop_outside_sources=False,
                drop_outside_receivers=False,
            )
            true_in_order = []
            for n in required:
                arr = np.load(by_name[n].path).astype(np.float32) if by_name[n].path is not None \
                      else np.full(by_name[n].shape, float(by_name[n].constant), dtype=np.float32)
                dh_tuple = (float(spec.grid.dh),) * arr.ndim
                cropped, _, _ = apply_model_plan(mp, arr, dh=dh_tuple, geom=None)
                true_in_order.append(torch.from_numpy(np.ascontiguousarray(cropped)).to(dev))
        else:
            true_in_order = [_load_model_tensor(by_name[n]).to(dev) for n in required]

        mod_wavelet, mod_sources, mod_receivers, used_override = _resolve_modeling_inputs(
            spec, wavelet, sources, receivers, shape
        )
        if used_override:
            print(f"[fwi] modeling_override applied (wavelet={spec.modeling_override.wavelet is not None}, "
                  f"geometry={spec.modeling_override.geometry is not None})")
        with torch.no_grad():
            # Chunk the obs-generation forward over shots so peak solver
            # workspace stays O(chunk) instead of O(nshots). Generating all
            # shots in one call allocates an adjoint workspace for the whole
            # batch and OOMs on large OBN surveys; the per-shot obs are
            # independent, so concatenating chunk outputs is exact. Chunk by
            # ``batchsize`` (the training shot batch); wavelet is shared.
            _n_src = int(mod_sources.shape[0])
            _cn = max(1, int(getattr(spec, "batchsize", 0) or _n_src))
            if _cn >= _n_src:
                obs = solver(mod_wavelet, mod_sources, mod_receivers,
                             models=true_in_order).detach().cpu()
            else:
                _obs_parts = []
                for _s0 in range(0, _n_src, _cn):
                    _s1 = min(_s0 + _cn, _n_src)
                    _part = solver(mod_wavelet, mod_sources[_s0:_s1],
                                   mod_receivers[_s0:_s1],
                                   models=true_in_order).detach().cpu()
                    _obs_parts.append(_part)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                obs = torch.cat(_obs_parts, dim=0)
        del true_in_order
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return obs

    def _run_epoch_qc_safe(self, *, spec, state, qc_dir: Path, epoch: int, dev) -> None:
        """Run all enabled QC products; never let plotting break the inversion.

        Optionally captures a forward-modelled syn for shot-gather QC when
        ``spec.qc.shot_gather`` is on. The shot indices used are spread
        evenly across the current obs to give a representative cross-section
        of acquisition geometry.
        """
        import torch

        try:
            from sweep_tasks import qc as qc_mod
        except Exception as err:  # noqa: BLE001
            print(f"[qc] import failed at epoch {epoch}: {err}")
            return

        extract = None
        if spec.qc.shot_gather:
            # Reuse the syn / obs snapshot captured at the end of the most
            # recent train step's last chunk. This avoids running an extra
            # solver forward inside the QC path (which doubles per-iter
            # GPU work) AND guarantees we visualise exactly what the loss
            # compared — including the syn bandpass applied in the train
            # step.
            snap = getattr(self, "_last_qc_snapshot", None)
            if snap is None:
                # First QC fires before the first train step on rare cold
                # paths; nothing to plot. Skip silently.
                extract = None
            else:
                n_pick = max(1, min(int(spec.qc.shot_gather_n_shots),
                                    int(snap["syn"].shape[0])))
                # Evenly subsample within the last chunk to give a
                # representative sample at low cost.
                snap_n = int(snap["syn"].shape[0])
                pick_within = np.linspace(0, snap_n - 1, n_pick, dtype=int)
                shot_ids = [int(snap["chunk_indices"][int(i)]) for i in pick_within]

            def _extract():
                syn_np = snap["syn"][pick_within].numpy()
                obs_chunk = snap["obs"][pick_within].numpy()
                # Both backends emit canonical (n_shots, nt, nrec, nchan).
                # Squeeze the trailing channel for the QC plotter which
                # expects (n_shots, nt, nrec) with time on axis -2.
                if syn_np.ndim == 4 and syn_np.shape[-1] == 1:
                    syn_np = syn_np[..., 0]
                if obs_chunk.ndim == 4 and obs_chunk.shape[-1] == 1:
                    obs_chunk = obs_chunk[..., 0]

                # Per-shot dedupe: at coarse dh, multiple physical receivers
                # snap to the same grid cell, producing identical syn traces.
                # The loss already gets these as duplicates; the QC should
                # display only ONE trace per unique grid cell so the user
                # sees what the inversion actually "sees".
                dh_m = float(state.get("dh", 1.0))
                src_grid = np.asarray(state["sources"][shot_ids], dtype=np.int64)
                rec_grid = np.asarray(state["receivers"][shot_ids], dtype=np.int64)
                src_xy_m = src_grid.astype(np.float64) * dh_m       # (n_shots, 2)
                rec_xy_m = rec_grid.astype(np.float64) * dh_m       # (n_shots, nrec, 2)
                # Build per-shot unique-cell indices.
                unique_idx_per_shot = []
                for s in range(len(shot_ids)):
                    seen: dict = {}
                    keep_in_order: list[int] = []
                    for r in range(rec_grid.shape[1]):
                        key = (int(rec_grid[s, r, 0]), int(rec_grid[s, r, 1]))
                        if key not in seen:
                            seen[key] = r
                            keep_in_order.append(r)
                    unique_idx_per_shot.append(np.asarray(keep_in_order, dtype=np.int64))
                return {
                    "obs": obs_chunk,                # (n_shots, nt, nrec) raw
                    "syn": syn_np,                   # (n_shots, nt, nrec) raw
                    "shot_ids": shot_ids,
                    "src_xy_m": src_xy_m,            # (n_shots, 2) in meters
                    "rec_xy_m": rec_xy_m,            # (n_shots, nrec, 2) in meters
                    "unique_idx_per_shot": unique_idx_per_shot,
                    "dh_m": dh_m,
                    "dt_s": float(state.get("dt", 1.0)),
                }

            extract = _extract

        try:
            qc_mod.run_epoch_qc(
                qc_spec=spec.qc, qc_dir=qc_dir, epoch=epoch,
                state=state, spec=spec, extract_obs_syn_panels=extract,
            )
        except Exception as err:  # noqa: BLE001
            print(f"[qc] epoch {epoch} failed: {err}")

    def _maybe_dump_loss_inputs(self, obs_chunk, syn, chunk, dist_info,
                                syn_bandpass, stage_dt,
                                state: dict | None = None) -> None:
        """One-shot debug dump of the exact obs/syn pair the loss consumed.

        Activated by env ``DUMP_LOSS_INPUTS=<path.npz>``. Fires once per
        process — saves the last chunk's CPU-side obs and syn (post
        bandpass), the chunk shot indices, and the current stage's dt /
        bandpass metadata. With ``DUMP_LOSS_INPUTS_EXIT=1`` the runner
        exits cleanly right after the dump, so users can capture a
        single iter's loss inputs with one sbatch and read them back
        offline.
        """
        if not dist_info.is_root:
            return
        dump_path = os.environ.get("DUMP_LOSS_INPUTS")
        if not dump_path or getattr(self, "_loss_inputs_dumped", False):
            return
        payload: dict = {
            "obs": obs_chunk.detach().cpu().numpy(),
            "syn": syn.detach().cpu().numpy(),
            "chunk_indices": np.asarray(chunk, dtype=np.int64),
            "stage_dt_s": float(stage_dt) if stage_dt is not None else float("nan"),
        }
        if syn_bandpass is not None:
            payload["bandpass_lo_hz"] = float(syn_bandpass.lo_hz)
            payload["bandpass_hi_hz"] = float(syn_bandpass.hi_hz)
            payload["bandpass_order"] = int(syn_bandpass.order)
        # Optional: snapshot the runner-side per-shot positions + pre-bandpass
        # pristine obs slice for the same chunk, so the user can verify
        # exactly what raw data the bandpass was applied to.
        if state is not None:
            try:
                src = np.asarray(state.get("sources"))[np.asarray(chunk, dtype=np.int64)]
                payload["source_grid_idx"] = np.asarray(src, dtype=np.int64)
                rec = np.asarray(state.get("receivers"))[np.asarray(chunk, dtype=np.int64)]
                payload["receiver_grid_idx"] = np.asarray(rec, dtype=np.int64)
                payload["grid_dh_m"] = float(state.get("dh", float("nan")))
                pristine = state.get("pristine_obs_np")
                if pristine is not None:
                    payload["pristine_obs_for_chunk"] = (
                        np.asarray(pristine)[np.asarray(chunk, dtype=np.int64)]
                        .astype(np.float32, copy=False)
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[debug] dump-state extras failed: {exc}", flush=True)
        out_dir = os.path.dirname(dump_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(dump_path, **payload)
        self._loss_inputs_dumped = True
        print(f"[debug] dumped loss inputs to {dump_path} "
              f"(obs={payload['obs'].shape}, syn={payload['syn'].shape})",
              flush=True)
        if os.environ.get("DUMP_LOSS_INPUTS_EXIT"):
            print("[debug] DUMP_LOSS_INPUTS_EXIT set; exiting cleanly",
                  flush=True)
            raise SystemExit(0)

    def _get_data_mask(self, spec, obs, dev):
        """Optional per-sample DATA mute mask (the on/off switch). Returns a CPU
        float tensor broadcastable to ``obs`` (nshots, nt, nrec[, nchan]) loaded
        from ``spec.loss.data_mask_path``, or None when unset -> the misfit runs
        exactly as before. Cached after first load. Indexed per-chunk like obs."""
        import torch
        path = getattr(getattr(spec, "loss", None), "data_mask_path", None)
        if not path:
            return None
        cached = getattr(self, "_data_mask_cache", None)
        if cached is not None and cached[0] == path:
            return cached[1]
        m = torch.as_tensor(np.asarray(np.load(path)), dtype=torch.float32)
        while m.ndim < obs.ndim:
            m = m.unsqueeze(-1)          # per-trace (nshots,nt,nrec) -> add channel axis
        try:
            torch.broadcast_shapes(tuple(m.shape), tuple(obs.shape))
        except RuntimeError as e:
            raise ValueError(
                f"data_mask_path shape {tuple(m.shape)} not broadcastable to "
                f"obs {tuple(obs.shape)}: {e}"
            )
        self._data_mask_cache = (path, m)
        return m

    def _apply_diffusion_prior(self, diff_prior, diff_kind, diff_weight,
                               diff_weight_mode, diff_sds_t, diff_seed,
                               leaves, render, dist_info):
        """Accumulate the diffusion-prior gradient IN-PLACE onto each leaf's
        ``.grad`` (which already holds the data gradient). Called once per
        optimizer step, only when the prior fires.

        ``leaves`` are the tensors whose ``.grad`` receives the contribution and
        the relative-scale reference; ``render()`` returns the velocity tensor
        the diffusion loss consumes. Three call shapes:
          * grid mode        — leaves ``[vp]``,        render ``lambda: vp``
          * reparam single   — leaves net params,      render ``lambda: net()``
          * reparam two-pass — leaves ``[base_leaf]``, render ``lambda: base_leaf``
            (the velocity leaf; its ``.grad`` is later pushed through the net).

        ``weight_mode="relative"`` (default): scale the diffusion gradient so its
        norm equals ``weight * ||current data gradient||`` (the raw RED gradient
        is normalised + mean-reduced, hence tiny and dataset-dependent, so an
        absolute weight is unintuitive). ``"absolute"``: add ``weight * grad``.

        Uses ``torch.autograd.grad`` (returns the isolated diffusion gradient
        without touching ``.grad``) then accumulates in place with ``add_`` — so
        any external alias of ``.grad`` (e.g. two-pass ``v_grad``) stays valid.

        RED (deterministic DiffPIR/DDIM) gives an identical gradient on every
        rank, so adding it post all-reduce keeps DDP params in sync; SDS is
        stochastic, so its RNG is reseeded identically per rank first.
        """
        import math
        import torch  # fwi.py imports torch per-method, not at module scope
        leaves = [p for p in leaves if p is not None and p.requires_grad]
        if not leaves:
            return
        if diff_kind == "sds":
            torch.manual_seed(int(diff_seed))
            loss = diff_prior.sds_loss(render(), diff_sds_t[0], diff_sds_t[1])
        else:
            loss = diff_prior.red_loss(render())
        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        if diff_weight_mode == "absolute":
            scale = float(diff_weight)
            g_ref = g_diff = float("nan")
        else:
            g_ref = math.sqrt(sum(
                float(p.grad.detach().pow(2).sum())
                for p in leaves if p.grad is not None))
            g_diff = math.sqrt(sum(
                float(g.detach().pow(2).sum()) for g in grads if g is not None))
            scale = (diff_weight * g_ref / g_diff) if g_diff > 0.0 else 0.0
        for p, g in zip(leaves, grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.mul(scale)
            else:
                p.grad.add_(g, alpha=scale)      # in-place -> preserves aliases
        if dist_info is not None and getattr(dist_info, "is_root", True):
            print(f"[diff] {diff_kind}/{diff_weight_mode}: |g_data|={g_ref:.3e} "
                  f"|g_diff_raw|={g_diff:.3e} scale={scale:.3e} "
                  f"(target {diff_weight:.3g}x |g_data|)")

    def _fwi_train_step(self, spec, solver, wavelet, sources, receivers,
                        inv_in_order, inv_by_name, obs, optimizer, nshots, dev,
                        *, dist_info=None, stage_batchsize: int | None = None,
                        reparam_net=None, local_window_ctx=None,
                        syn_bandpass=None, stage_dt: float | None = None,
                        state_for_dump: dict | None = None,
                        take_qc_snapshot: bool = True,
                        tv_prior=None, smooth_weight: float = 0.0,
                        diff_prior=None, diff_weight: float = 0.0,
                        diff_weight_mode: str = "relative",
                        diff_kind: str = "red", diff_apply: bool = False,
                        diff_sds_t=(0.02, 0.5), diff_seed: int = 0,
                        grad_smoother=None, grad_smooth_every: int = 1) -> float:
        """One outer optimizer step.

        In single-process mode the rank picks a `batchsize` shot batch, breaks
        it into `train_shot_batchsize` chunks, runs forward+backward on each
        chunk, and steps. In distributed mode (world_size > 1) the **global**
        batchsize is split round-robin across ranks (rank 0 picks + broadcasts
        the global indices, then `shot_idx[rank::world_size]` is local). The
        loss is normalised by the GLOBAL element count so that
        ``all_reduce(SUM)`` of gradients yields the gradient of the global
        per-element mean loss.

        ``stage_batchsize`` (if given) overrides ``spec.batchsize`` — used
        by the multi-stage loop so each stage can pick its own shot count
        (Gap 4).
        """

        import torch

        from sweep_tasks.runtime import distributed as _dist

        if dist_info is None:
            dist_info = getattr(self, "_dist", None) or _dist.init_distributed_if_needed()

        global_batchsize = min(
            int(stage_batchsize) if stage_batchsize is not None else int(spec.batchsize),
            nshots,
        )

        # Rank 0 picks the global shot indices, then broadcasts.
        if dist_info.is_root:
            shot_idx_global = np.random.choice(nshots, size=global_batchsize, replace=False)
        else:
            shot_idx_global = None
        shot_idx_global = _dist.broadcast_shot_indices(
            shot_idx_global, global_batchsize, dist_info, src=0
        )

        local_shots = _dist.split_for_rank(shot_idx_global, dist_info)
        chunk_size = spec.train_shot_batchsize or max(len(local_shots), 1)
        if len(local_shots) > 0:
            chunks = [local_shots[i:i + chunk_size]
                      for i in range(0, len(local_shots), chunk_size)]
        else:
            chunks = []

        # Global element count used for loss normalisation. Pulled from a sample
        # obs slice so any inferred dimension (nt, nrec) stays in sync.
        sample = obs[:1]
        per_shot_numel = int(sample.numel())
        global_norm = float(per_shot_numel * global_batchsize)
        data_mask = self._get_data_mask(spec, obs, dev)
        if data_mask is not None and not _MASK_WINDOW_TRACE_COSINE(spec):
            # Pointwise misfits lose the muted samples from the sum, so the norm
            # must shrink by the kept fraction. NOT so for windowed trace_cosine:
            # its per-trace value is broadcast over ALL nt samples regardless of
            # the mask, so the sum keeps its full-length scale and shrinking the
            # norm would inflate the reported loss by 1/mask.mean().
            global_norm = float(global_norm * max(float(data_mask.float().mean()), 1.0e-6))

        # Per-chunk forward input. In reparam mode we render a fresh vp
        # each chunk so the autograd graph from solver -> loss -> backward
        # flows into the network's parameters. The other inv tensors (if
        # any) pass through unchanged.
        def _models_for_chunk():
            if reparam_net is None:
                return inv_in_order
            rendered = reparam_net()
            return [
                rendered if name == "vp" else inv_by_name[name]
                for name in [m for m in inv_by_name]
            ] if len(inv_by_name) > 1 else [rendered]

        # Per-chunk local-window dispatcher. When ``local_window_ctx`` is
        # set, each chunk gets a (z0, z1, x0, x1) crop of the model + a
        # shape-keyed cached solver + window-local geometry indices.
        # Otherwise we fall back to the full-model path (above).
        def _chunk_inputs(chunk_idx):
            chunk_src = sources[chunk_idx]
            chunk_rec = receivers[chunk_idx]
            if local_window_ctx is None:
                return _models_for_chunk(), solver, chunk_src, chunk_rec
            win_spec = local_window_ctx["spec"]
            full_shape = local_window_ctx["shape"]
            dh = local_window_ctx["dh"]
            window = _compute_local_window(
                chunk_src, chunk_rec, full_shape, dh, win_spec,
            )
            if len(window) == 4:
                z0, z1, x0, x1 = window
                local_shape = (z1 - z0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0}
                vp_slice = (slice(z0, z1), slice(x0, x1))
            else:
                z0, z1, y0, y1, x0, x1 = window
                local_shape = (z1 - z0, y1 - y0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0, "y0": y0}
                vp_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            cache = local_window_ctx["solver_cache"]
            cache_key = local_shape
            if cache_key not in cache:
                # Bound the cache so randomly varying windows don't accumulate
                # solvers indefinitely (each c-backend solver pins GPU memory
                # for its wavefield + PML buffers). Evict the least-recently-
                # used entry when over the cap.
                _LOCAL_SOLVER_CACHE_CAP = 4
                if len(cache) >= _LOCAL_SOLVER_CACHE_CAP:
                    # dict insertion-ordered: drop the oldest entry.
                    oldest = next(iter(cache))
                    del cache[oldest]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                cache[cache_key] = _build_solver(
                    spec.physics, spec.backend, local_shape,
                    dh, local_window_ctx["dt"], local_window_ctx["nt"], dev,
                )
            else:
                # Move-to-end so this entry is "most recently used".
                cache[cache_key] = cache.pop(cache_key)
            local_solver = cache[cache_key]
            local_src, local_rec = _rebase_geometry_to_window(chunk_src, chunk_rec, **rebase_kwargs)
            # Build the model list. Only vp is currently supported as
            # the windowed model; non-vp models pass through unmodified.
            if reparam_net is not None:
                if len(window) != 4:
                    raise NotImplementedError(
                        "Reparam render_window is only implemented for 2-D "
                        "grids; disable spec.reparam for 3-D local windowing."
                    )
                local_vp = reparam_net.render_window(z0, z1, x0, x1)
            else:
                # Slicing a leaf tensor yields a view; PyTorch's autograd
                # scatters the gradient back to the leaf at vp_slice.
                local_vp = inv_by_name["vp"][vp_slice]
            local_vp = local_vp.contiguous()
            ordered_names = list(inv_by_name.keys())
            models = [
                local_vp if name == "vp" else inv_by_name[name]
                for name in ordered_names
            ] if len(inv_by_name) > 1 else [local_vp]
            # perf/acoustic-bwd-skip-illum (3d6fe97): the c-backend now makes
            # backward illumination OPT-IN (default off for speed). Enable it on
            # each (cached) solver when illumination_precondition is active, else
            # solver.source_illumination stays None and the precond silently
            # no-ops. Harmless on older cores (plain attr, ignored).
            if getattr(getattr(spec, "illumination_precondition", None), "enabled", False):
                try:
                    local_solver.compute_illumination = True
                except Exception:
                    pass
            return models, local_solver, local_src, local_rec

        if spec.optimizer.kind == "lbfgs":
            # LBFGS in single-process only (the runner blocks dist + LBFGS earlier).
            def _closure():
                optimizer.zero_grad()
                acc_loss = 0.0
                for chunk in chunks:
                    models, chunk_solver, chunk_src, chunk_rec = _chunk_inputs(chunk)
                    syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                    if syn_bandpass is not None and stage_dt is not None:
                        syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                                  stage_dt, order=syn_bandpass.order)
                    obs_chunk = obs[chunk].to(dev)
                    loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev),
                                   window_mode=_MASK_WINDOW(spec))
                    (loss_t / global_norm).backward()
                    acc_loss += float(loss_t.detach().cpu())
                if reparam_net is None:
                    _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                return torch.tensor(acc_loss / global_norm if global_norm else 0.0)
            result = optimizer.step(_closure)
            return float(result) if result is not None else 0.0

        # Adam / SGD path. Two top-level branches:
        #   (A) Single-step (no reparam, OR reparam with backward_mode='single_step'):
        #       one autograd graph through net -> solver -> loss -> backward.
        #   (B) Two-pass reparam (default, mirrors fwi_workflow-dev): render
        #       net under no_grad to a leaf, solver backward into leaf.grad,
        #       then a second pass pushes leaf.grad through net in either
        #       'two_pass_full' (one big backward) or 'two_pass_chunked'
        #       (row-chunked re-render via VelocityINR.backward_velocity_gradient).
        reparam_mode = (
            spec.reparam.backward_mode
            if (reparam_net is not None and spec.reparam is not None)
            else "single_step"
        )

        illum_spec = getattr(spec, "illumination_precondition", None)
        illum_on = bool(illum_spec is not None and illum_spec.enabled)
        if illum_on and local_window_ctx is not None:
            raise NotImplementedError(
                "illumination_precondition is not supported together with "
                "local_model_window: the per-window illumination tensor "
                "shape varies per chunk, so per-window accumulation is "
                "not implemented. Disable one of the two."
            )
        # The eager (autograd PyTorch) backend does not populate
        # ``solver.source_illumination`` / ``receiver_illumination`` — those
        # are a feature of the compiled C kernel. Warn once per train step
        # so the user isn't silently running a no-op precondition.
        if illum_on and dist_info.is_root and getattr(spec.backend, "impl", "eager") == "eager":
            if not getattr(self, "_illum_eager_warned", False):
                print(
                    "[illum] WARN: illumination_precondition is enabled but "
                    "spec.backend.impl='eager' — the eager autograd backend "
                    "does not populate solver.source_illumination / "
                    "receiver_illumination, so the precondition is a no-op. "
                    "Switch to backend.impl='c' for the production path."
                )
                self._illum_eager_warned = True
        if reparam_net is None or reparam_mode == "single_step":
            # ---- (A) single-step: everything inside one autograd graph ----
            # Optional per-iter profiling: set ``SWEEP_TASKS_TPROF=1`` in
            # the environment to print a millisecond-level breakdown of
            # each step (zero_grad / forward / bandpass / obs-H2D / loss /
            # backward / item / snapshot-D2H / reduce+clamp / opt.step).
            # Each phase boundary is followed by ``torch.cuda.synchronize()``
            # so the numbers reflect real GPU time, not kernel launch latency.
            import time as _t
            _TPROF = os.environ.get("SWEEP_TASKS_TPROF") == "1"
            def _sync():
                if _TPROF and torch.cuda.is_available(): torch.cuda.synchronize()
            if _TPROF: _sync(); _t0 = _t.perf_counter()
            optimizer.zero_grad()
            if _TPROF: _sync(); _t_zero = _t.perf_counter()
            acc_loss_local = 0.0
            sill_sum = None
            rill_sum = None
            for chunk_idx_in_iter, chunk in enumerate(chunks):
                models, chunk_solver, chunk_src, chunk_rec = _chunk_inputs(chunk)
                if _TPROF: _sync(); _t_pre = _t.perf_counter()
                syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                if _TPROF: _sync(); _t_fwd = _t.perf_counter()
                if syn_bandpass is not None and stage_dt is not None:
                    syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                              stage_dt, order=syn_bandpass.order)
                if _TPROF: _sync(); _t_bp = _t.perf_counter()
                obs_chunk = obs[chunk].to(dev)
                if _TPROF: _sync(); _t_obs = _t.perf_counter()
                loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev),
                                   window_mode=_MASK_WINDOW(spec))
                if _TPROF: _sync(); _t_loss = _t.perf_counter()
                (loss_t / global_norm).backward()
                if _TPROF: _sync(); _t_bwd = _t.perf_counter()
                acc_loss_local += float(loss_t.detach().cpu())
                if _TPROF: _sync(); _t_item = _t.perf_counter()
                if illum_on:
                    sill_sum, rill_sum = _accumulate_illumination(
                        chunk_solver, sill_sum, rill_sum,
                    )
                # Snapshot last chunk for QC reuse (avoid running a fresh
                # forward inside ``_run_epoch_qc_safe._extract``). Only the
                # final chunk's syn/obs survive across iters; QC at every
                # show_every iter will see the most recent train batch.
                if (take_qc_snapshot
                        and dist_info.is_root
                        and chunk_idx_in_iter == len(chunks) - 1):
                    self._last_qc_snapshot = {
                        "chunk_indices": np.asarray(chunk, dtype=np.int64).copy(),
                        "syn": syn.detach().to("cpu"),
                        "obs": obs_chunk.detach().to("cpu"),
                    }
                    self._maybe_dump_loss_inputs(
                        obs_chunk, syn, chunk, dist_info, syn_bandpass, stage_dt,
                        state=state_for_dump,
                    )
                if _TPROF: _sync(); _t_snap = _t.perf_counter()
            if reparam_net is None:
                _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                _dist.all_reduce_grad_sum(inv_in_order, dist_info)
                if illum_on:
                    # Match the gradient's all-reduce so multi-rank accumulation
                    # is consistent. We only support per-cell vp illumination
                    # here (the only inverted model in OBN FWI).
                    if sill_sum is not None and rill_sum is not None:
                        _dist.all_reduce_sum_inplace(sill_sum, dist_info)
                        _dist.all_reduce_sum_inplace(rill_sum, dist_info)
                    vp_grad = inv_by_name.get("vp").grad if "vp" in inv_by_name else None
                    _apply_illumination_precond(
                        vp_grad, sill_sum, rill_sum,
                        eps=illum_spec.epsilon, exponent=illum_spec.exponent,
                        relative_epsilon=getattr(illum_spec, "relative_epsilon", None),
                    )
            else:
                _dist.all_reduce_grad_sum(
                    [p for p in reparam_net.parameters() if p.grad is not None],
                    dist_info,
                )
                if illum_on and dist_info.is_root:
                    # Reparam + single_step: net params don't have a per-cell
                    # illumination correspondence. Use two_pass_full instead
                    # to get illumination preconditioning on the leaf gradient
                    # before the network backward.
                    print("[illum] WARN: illumination_precondition is "
                          "a no-op under reparam.backward_mode='single_step'. "
                          "Switch to 'two_pass_full' to enable it.")
            # Smoothing prior (single-step / grid path): add ∂(w·TV)/∂v after
            # the data-gradient all-reduce, before the step.
            if tv_prior is not None:
                if reparam_net is None:
                    _vp = inv_by_name.get("vp")
                    if _vp is not None and _vp.requires_grad:
                        (smooth_weight * tv_prior(_vp)).backward()
                else:
                    (smooth_weight * tv_prior(reparam_net())).backward()
            # Diffusion (DDPM) prior — accumulate its gradient onto the (already
            # all-reduced) data gradient, gated by diff_apply (every N steps /
            # from start_band). Deterministic RED is identical on every rank so
            # it keeps params in sync; SDS is reseeded per rank inside the helper.
            if diff_prior is not None and diff_apply:
                if reparam_net is None:
                    _dvp = inv_by_name.get("vp")
                    _dleaves, _drender = [_dvp], (lambda: _dvp)
                else:
                    _dleaves = list(reparam_net.parameters())
                    _drender = (lambda: reparam_net())
                self._apply_diffusion_prior(
                    diff_prior, diff_kind, diff_weight, diff_weight_mode,
                    diff_sds_t, diff_seed, _dleaves, _drender, dist_info,
                )
            # Gaussian gradient smoothing (grid-mode vp only), just before the
            # step so it smooths the FINAL (data + TV) gradient.
            if grad_smoother is not None and reparam_net is None:
                _vp = inv_by_name.get("vp")
                if _vp is not None and _vp.grad is not None:
                    with torch.no_grad():
                        _vp.grad.copy_(grad_smoother(_vp.grad))
                    # Re-apply the top-row freeze AFTER smoothing: a Gaussian
                    # z-conv smears gradient from unfrozen rows back into the
                    # frozen top rows, which optimizer.step would then update
                    # (silently defeating freeze_top_n_rows). Cheap re-zero.
                    _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
            if _TPROF: _sync(); _t_reduce = _t.perf_counter()
            optimizer.step()
            if _TPROF:
                _sync(); _t_step = _t.perf_counter()
                _ms = lambda a, b: (b - a) * 1000.0
                print(f"[tprof] zero={_ms(_t0,_t_zero):5.1f} "
                      f"fwd={_ms(_t_pre,_t_fwd):6.1f} "
                      f"bp={_ms(_t_fwd,_t_bp):4.1f} "
                      f"obs_h2d={_ms(_t_bp,_t_obs):5.1f} "
                      f"loss={_ms(_t_obs,_t_loss):4.1f} "
                      f"bwd={_ms(_t_loss,_t_bwd):6.1f} "
                      f"item={_ms(_t_bwd,_t_item):4.1f} "
                      f"snap_d2h={_ms(_t_item,_t_snap):5.1f} "
                      f"reduce={_ms(_t_snap,_t_reduce):4.1f} "
                      f"step={_ms(_t_reduce,_t_step):5.1f} "
                      f"TOTAL={_ms(_t0,_t_step):6.1f}ms")
        else:
            # ---- (B) two-pass: solver phase against a leaf, then net phase ----
            optimizer.zero_grad()
            acc_loss_local = 0.0
            # Optional SYNCED per-phase profile (SWEEP_TASKS_TPROF=1). The
            # two-pass timing the QC line prints is NOT cuda-synced, so the
            # solver-adjoint async tail bleeds into reparam_bwd. This block
            # measures real wall-time per phase.
            import time as _tm
            _TPROF = os.environ.get("SWEEP_TASKS_TPROF") == "1"
            def _sync():
                if _TPROF and torch.cuda.is_available(): torch.cuda.synchronize()
            _pf = {"render": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0, "reparam": 0.0}
            if _TPROF: _sync(); _t_a = _tm.perf_counter()

            # Render the full-grid base velocity once, detached. The leaf is
            # the autograd boundary; solver backward accumulates onto leaf.grad.
            with torch.no_grad():
                base_leaf = reparam_net().detach().clone()
            base_leaf = base_leaf.requires_grad_(True)
            if _TPROF: _sync(); _pf["render"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()

            # Per-chunk forward + backward (autograd graph from solver to leaf).
            # ``_chunk_inputs`` already handles local-window slicing if enabled;
            # we just substitute the leaf for the network output and reuse the
            # window logic.
            def _two_pass_chunk_inputs(chunk_idx, leaf):
                chunk_src = sources[chunk_idx]
                chunk_rec = receivers[chunk_idx]
                if local_window_ctx is None:
                    models = [leaf]
                    # Opt-in backward illumination on the new c-core (perf/
                    # acoustic-bwd-skip-illum): the two-pass reparam path must
                    # enable it too, else solver.source_illumination stays None
                    # and the illum precond silently no-ops (the single-step
                    # chunk helper already does this). Harmless on older cores.
                    if getattr(getattr(spec, "illumination_precondition", None), "enabled", False):
                        try:
                            solver.compute_illumination = True
                        except Exception:
                            pass
                    return models, solver, chunk_src, chunk_rec
                win_spec = local_window_ctx["spec"]
                full_shape = local_window_ctx["shape"]
                dh = local_window_ctx["dh"]
                window = _compute_local_window(
                    chunk_src, chunk_rec, full_shape, dh, win_spec,
                )
                if len(window) == 4:
                    z0, z1, x0, x1 = window
                    local_shape = (z1 - z0, x1 - x0)
                    rebase_kwargs = {"z0": z0, "x0": x0}
                    leaf_slice = (slice(z0, z1), slice(x0, x1))
                else:
                    z0, z1, y0, y1, x0, x1 = window
                    local_shape = (z1 - z0, y1 - y0, x1 - x0)
                    rebase_kwargs = {"z0": z0, "x0": x0, "y0": y0}
                    leaf_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                cache = local_window_ctx["solver_cache"]
                cache_key = local_shape
                if cache_key not in cache:
                    # See note above _chunk_inputs: bound the cache to 4
                    # entries (LRU) so varying batches don't accumulate
                    # solvers indefinitely.
                    if len(cache) >= 4:
                        oldest = next(iter(cache))
                        del cache[oldest]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    cache[cache_key] = _build_solver(
                        spec.physics, spec.backend, local_shape,
                        dh, local_window_ctx["dt"], local_window_ctx["nt"], dev,
                    )
                else:
                    cache[cache_key] = cache.pop(cache_key)  # touch (LRU)
                local_solver = cache[cache_key]
                local_src, local_rec = _rebase_geometry_to_window(chunk_src, chunk_rec, **rebase_kwargs)
                # View into leaf — gradient scatters back to leaf.grad on backward.
                models = [leaf[leaf_slice].contiguous()]
                if getattr(getattr(spec, "illumination_precondition", None), "enabled", False):
                    try:
                        local_solver.compute_illumination = True
                    except Exception:
                        pass
                return models, local_solver, local_src, local_rec

            sill_sum = None
            rill_sum = None
            for chunk_idx_in_iter, chunk in enumerate(chunks):
                models, chunk_solver, chunk_src, chunk_rec = _two_pass_chunk_inputs(chunk, base_leaf)
                if _TPROF: _sync(); _t_a = _tm.perf_counter()
                syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=models)
                if _TPROF: _sync(); _pf["fwd"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()
                if syn_bandpass is not None and stage_dt is not None:
                    syn = _bandpass_syn_torch(syn, syn_bandpass.lo_hz, syn_bandpass.hi_hz,
                                              stage_dt, order=syn_bandpass.order)
                obs_chunk = obs[chunk].to(dev)
                loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev),
                                   window_mode=_MASK_WINDOW(spec))
                if _TPROF: _sync(); _pf["loss"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()
                (loss_t / global_norm).backward()
                if _TPROF: _sync(); _pf["bwd"] += _tm.perf_counter() - _t_a; _t_a = _tm.perf_counter()
                acc_loss_local += float(loss_t.detach().cpu())
                if illum_on:
                    sill_sum, rill_sum = _accumulate_illumination(
                        chunk_solver, sill_sum, rill_sum,
                    )
                # Snapshot last chunk for QC reuse (see single-step branch).
                if dist_info.is_root and chunk_idx_in_iter == len(chunks) - 1:
                    self._last_qc_snapshot = {
                        "chunk_indices": np.asarray(chunk, dtype=np.int64).copy(),
                        "syn": syn.detach().to("cpu"),
                        "obs": obs_chunk.detach().to("cpu"),
                    }
                    self._maybe_dump_loss_inputs(
                        obs_chunk, syn, chunk, dist_info, syn_bandpass, stage_dt,
                        state=state_for_dump,
                    )

            # leaf.grad now holds the full-grid FWI gradient ∂L/∂v.
            # Push it through the network in a second pass.
            v_grad = base_leaf.grad
            # Debug: dump the RAW velocity gradient dL/dvp on the FIRST outer
            # step (= init model), before illum/smooth. SWEEP_DUMP_GRAD=1.
            # v_grad here is this rank's partial (the leaf all-reduce happens
            # implicitly via the network backward), so sum across ranks first.
            if (os.environ.get("SWEEP_DUMP_GRAD") == "1"
                    and not getattr(self, "_csg_grad_dumped", False)
                    and v_grad is not None):
                _gd = v_grad.detach().clone()
                if dist_info.is_distributed:
                    _dist.all_reduce_sum_inplace(_gd, dist_info)
                if dist_info.is_root:
                    import numpy as _np
                    _dd = os.environ.get("SWEEP_DUMP_DIR", "/tmp")
                    _np.save(_dd + "/grad_raw.npy", _gd.detach().cpu().numpy())
                    print(f"[dump] raw dL/dvp -> {_dd}/grad_raw.npy "
                          f"shape={tuple(_gd.shape)}", flush=True)
                self._csg_grad_dumped = True
            # Illumination precondition the leaf gradient BEFORE the
            # network backward so the network sees the preconditioned
            # velocity gradient. The all-reduce of the leaf gradient
            # happens implicitly via the network backward (each rank
            # computes its own contribution and gradients on net params
            # are all_reduced after).
            if illum_on and v_grad is not None:
                if sill_sum is not None and rill_sum is not None:
                    _dist.all_reduce_sum_inplace(sill_sum, dist_info)
                    _dist.all_reduce_sum_inplace(rill_sum, dist_info)
                _apply_illumination_precond(
                    v_grad, sill_sum, rill_sum,
                    eps=illum_spec.epsilon, exponent=illum_spec.exponent,
                    relative_epsilon=getattr(illum_spec, "relative_epsilon", None),
                )
            # Smoothing prior: add ∂(w·TV)/∂v to the (illum-preconditioned)
            # leaf gradient BEFORE pushing it through the network — matches the
            # multisource path's order (illum, then smooth). base_leaf carries
            # requires_grad, so reg_loss.backward() accumulates into
            # base_leaf.grad (which v_grad aliases).
            if tv_prior is not None and v_grad is not None:
                reg_loss = smooth_weight * tv_prior(base_leaf)
                reg_loss.backward()
            if v_grad is None:
                # Nothing actually backpropped (e.g. all chunks were empty);
                # nothing to do.
                pass
            elif reparam_mode == "two_pass_full":
                # One full-graph backward through the network with v_grad as
                # the gradient at the (re-rendered) output.
                rendered = reparam_net()
                # Make sure shapes agree (they always do — net renders to
                # base_velocity.shape and leaf came from net()).
                rendered.backward(v_grad)
            elif reparam_mode == "two_pass_chunked":
                # Chunked re-render — memory-conscious for large grids.
                reparam_net.backward_velocity_gradient(
                    v_grad, chunk_rows=int(spec.reparam.backward_chunk_rows),
                )
            else:
                raise ValueError(f"unknown reparam backward_mode {reparam_mode!r}")

            if _TPROF: _sync(); _pf["reparam"] += _tm.perf_counter() - _t_a
            if _TPROF and dist_info.is_root:
                _tot = sum(_pf.values()) * 1e3
                print(f"[TPROF-B synced] render={_pf['render']*1e3:6.0f} solver_fwd={_pf['fwd']*1e3:6.0f} "
                      f"loss={_pf['loss']*1e3:5.0f} solver_bwd_adjoint={_pf['bwd']*1e3:6.0f} "
                      f"reparam={_pf['reparam']*1e3:6.0f}  sum={_tot:6.0f} ms", flush=True)

            _dist.all_reduce_grad_sum(
                [p for p in reparam_net.parameters() if p.grad is not None],
                dist_info,
            )
            # Diffusion (DDPM) prior at the PARAM level, AFTER the all-reduce so
            # the reference gradient is global and the (deterministic RED)
            # contribution is identical on every rank -> params stay in sync.
            # Equivalent to a leaf-level term pushed through the net (J^T dL/dvp),
            # but correct under two-pass DDP (base_leaf.grad is a per-rank
            # partial before the network backward).
            if diff_prior is not None and diff_apply:
                self._apply_diffusion_prior(
                    diff_prior, diff_kind, diff_weight, diff_weight_mode,
                    diff_sds_t, diff_seed, list(reparam_net.parameters()),
                    (lambda: reparam_net()), dist_info,
                )
            optimizer.step()
            # Two-pass leaks references to the leaf's grad tensor across
            # Adam steps unless we explicitly drop the leaf. Without this,
            # GPU memory grows ~1 GB / step. ``empty_cache`` is cheap and
            # gives the allocator a chance to defragment between steps.
            del base_leaf
            if v_grad is not None:
                del v_grad
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        acc_loss_global = _dist.all_reduce_scalar_sum(acc_loss_local, dist_info)
        return acc_loss_global / global_norm if global_norm else 0.0
