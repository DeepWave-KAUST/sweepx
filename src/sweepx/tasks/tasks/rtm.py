"""RTM task runner (mixin). Verbatim from runner.py."""
from pathlib import Path
from sweep_preproc.filter import bandpass as _bandpass_cpu
import numpy as np
from sweep_tasks.schemas import RTMSpec
from sweep_tasks._helpers.dd import (
    _compute_local_window,
    _rebase_geometry_to_window,
)
from sweep_tasks._helpers.geometry import _build_geometry_2d
from sweep_tasks._helpers.loss import _compute_loss
from sweep_tasks._helpers.model import (
    _get_equation_class,
    _load_model_tensor,
)
from sweep_tasks._helpers.plan_apply import _apply_data_plan_to_fwi
from sweep_tasks._helpers.plotting import (
    _crop_padded_volume_to_model,
    _save_rtm_qc_pngs,
)
from sweep_tasks._helpers.solver_build import (
    _build_solver,
    _cfl_check,
    _validate_single_model,
)
from sweep_tasks._helpers.stages import (
    _bandpass_syn_torch,
    _resample_obs_time,
    _trim_or_pad_time,
)
from sweep_tasks._helpers.util import _apply_seed
from sweep_tasks._helpers.wavelet import _build_wavelet


class RTMRunnerMixin:
    def _run_rtm(self, spec: RTMSpec, task_dir: Path):
        """Post-FWI Reverse Time Migration: one pass over all shots, no iteration.

        Per batch: forward, bandpass syn/obs, backward → the gradient image
        IS the RTM cross-correlation under the chosen matching loss. The
        c-backend forward call (re)populates ``solver.source_illumination``
        and ``solver.receiver_illumination`` during backward, so the two
        illumination maps come for free; no separate ``solver.rtm`` call.
        Accumulate sums across all shots, then normalise by
        ``sqrt(S * R + eps)`` and save raw + normalised products.

        Legacy reference: :func:`fwi_workflow.imaging.rtm.run_sweep_imaging`
        (which DID call ``solver.rtm`` separately because the older sweep
        backend did not expose illumination via the regular forward path).
        """
        import torch

        from sweep_tasks.runtime import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        _apply_seed(spec.seed)
        dev = _dist.resolve_dist_device(spec.device, dist_info.local_rank)
        equation_cls = _get_equation_class(spec.physics.equation)
        _validate_single_model(equation_cls, spec.velocity_model)

        # ---- 1) Shape, solver, wavelet, geometry, obs ------------------------
        # Mirror _run_fwi's data_plan.dt_target_s sync: when obs is resampled
        # via data_plan, the solver runs at the matching dt so syn/obs align.
        effective_dt = float(spec.time.dt)
        effective_nt = int(spec.time.nt)
        if spec.data_plan is not None and spec.data_plan.dt_target_s is not None:
            target_dt = float(spec.data_plan.dt_target_s)
            if abs(target_dt - effective_dt) > 1e-12:
                new_nt = int(round(effective_dt * effective_nt / target_dt))
                if dist_info.is_root:
                    print(f"[rtm] data_plan.dt_target_s={target_dt}s -> auto-sync "
                          f"solver: dt {effective_dt} -> {target_dt}, nt "
                          f"{effective_nt} -> {new_nt}")
                effective_dt = target_dt
                effective_nt = new_nt

        vm_ref = spec.velocity_model
        if spec.grid.shape is not None:
            shape = tuple(int(v) for v in spec.grid.shape)
        elif vm_ref.constant is not None:
            shape = tuple(int(v) for v in vm_ref.shape)
        else:
            shape = tuple(np.load(vm_ref.path, mmap_mode="r").shape)

        try:
            if vm_ref.path is not None:
                vmax_estimate = float(np.load(vm_ref.path, mmap_mode="r").max())
            elif vm_ref.constant is not None:
                vmax_estimate = float(vm_ref.constant)
            else:
                vmax_estimate = 0.0
            if vmax_estimate > 0 and dist_info.is_root:
                _cfl_check(vmax_estimate, float(spec.grid.dh), effective_dt)
        except FileNotFoundError:
            pass

        solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh,
            effective_dt, effective_nt, dev,
        )
        wavelet = _build_wavelet(
            spec.wavelet, spec.time,
            override_dt=effective_dt, override_nt=effective_nt,
        )

        segy_cache: dict = {}
        sources, receivers = _build_geometry_2d(
            spec.geometry, shape, dh=spec.grid.dh, segy_cache=segy_cache,
        )
        nshots = int(sources.shape[0])

        # Load velocity (as a fresh leaf each batch — we need grad on vp to
        # derive the FWI gradient image; the input itself is constant).
        vp_base = _load_model_tensor(vm_ref).to(dev)

        # ---- Obs source decision: lazy PlanReader vs eager fallback ----------
        # When ``spec.obs.plan`` is set we use a PlanReader directly so each
        # batch's traces are read on demand (cache_all=True keeps them hot in
        # RAM after the first pass; cache_all=False keeps memory bounded to
        # one batch — required for production OBN-scale obs). Per-batch resample +
        # bandpass move into the batch loop so the lazy path doesn't have to
        # materialise a full (nshots, nt, nrec, 1) tensor.
        #
        # All other obs sources (synthetic, npy, segy, segy_index) fall back
        # to the eager `_fwi_load_obs` route. They could be lazy-fied later
        # by adding similar reader objects; for now they stay one-shot.
        from sweep_io.seismic_plan import PlanReader, SeismicPlan

        plan_reader: "PlanReader | None" = None
        plan_obj: "SeismicPlan | None" = None
        obs_np = None
        if getattr(spec.obs, "plan", None) is not None:
            plan_obj = SeismicPlan.load(spec.obs.plan.plan_path)
            if plan_obj.grouping != "csg":
                raise NotImplementedError(
                    f"[rtm] obs.plan lazy reader currently supports "
                    f"grouping='csg' only (got {plan_obj.grouping!r}); "
                    "use the legacy CRG path for OBN datasets."
                )
            counts = plan_obj.per_group_row_counts()
            if not (counts == counts[0]).all():
                raise ValueError(
                    "[rtm] obs.plan with non-uniform receiver count per shot; "
                    "rebuild the plan with a filter that yields constant nrec."
                )
            if int(plan_obj.n_groups) != nshots:
                raise ValueError(
                    f"[rtm] obs.plan n_groups={plan_obj.n_groups} != "
                    f"geometry-derived nshots={nshots}. The plan must match "
                    "the geometry source (use geometry.kind=from_plan with "
                    "the same plan_path)."
                )
            plan_reader = PlanReader(plan_obj, cache_all=bool(spec.obs.plan.cache_all))
            if dist_info.is_root:
                cache_label = ("cache_all=True (pre-loaded to RAM)"
                               if spec.obs.plan.cache_all
                               else "cache_all=False (per-batch disk read)")
                print(f"[rtm] obs.plan lazy reader open: {plan_obj.n_groups} "
                      f"groups, {int(counts[0])} rec/group, dt={plan_obj.dt_s:.4g}s, "
                      f"{cache_label}")
        else:
            # Eager fallback for non-plan obs (synthetic / npy / segy / segy_index).
            obs = self._fwi_generate_obs(
                spec, equation_cls, solver, wavelet,
                sources, receivers, shape, dev, nshots,
                segy_cache=segy_cache,
            )
            sources, receivers, obs = _apply_data_plan_to_fwi(
                spec, sources, receivers, obs, spec.grid.dh,
                spec.time.dt, spec.time.nt, dev,
            )
            nshots = int(sources.shape[0])
            obs_np = (obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor)
                      else np.asarray(obs))
            if obs_np.ndim != 4:
                raise ValueError(
                    f"[rtm] eager obs.ndim={obs_np.ndim} not supported; "
                    "expected canonical 4-D (nshots, nt, nrec, 1)."
                )
            obs_np = np.ascontiguousarray(obs_np.astype(np.float32, copy=False))

        # ---- 2) pristine_dt fix (mirror _run_fwi exactly) --------------------
        # SEG-Y / plan-backed obs is read at the file's native dt; we resample
        # per-batch below. Don't let a mis-labelled pristine_dt corrupt the
        # resample (the same bug that bit the Viking benchmark).
        obs_native_dt = float(effective_dt)
        if spec.data_plan is None or spec.data_plan.dt_target_s is None:
            if plan_obj is not None:
                obs_native_dt = float(plan_obj.dt_s)
            elif segy_cache and (
                getattr(spec.obs, "segy", None) is not None
                or getattr(spec.obs, "segy_index", None) is not None
            ):
                for payload in segy_cache.values():
                    if isinstance(payload, dict) and "dt_s" in payload:
                        obs_native_dt = float(payload["dt_s"])
                        break
        if abs(obs_native_dt - float(effective_dt)) > 1e-12 and dist_info.is_root:
            print(f"[rtm] obs native dt={obs_native_dt}s differs from solver "
                  f"dt={effective_dt}s; per-batch resample will sync.")

        imaging = spec.imaging
        lo_hz = imaging.filter_lowcut_hz
        hi_hz = imaging.filter_highcut_hz
        bandpass_obs = (lo_hz is not None and hi_hz is not None)

        # ---- 2c) Optional wavelet bandpass (filter_target='wavelet') ---------
        # Mirrors the FWI ``stage.bandpass.target == 'wavelet'`` path: pre-
        # filter the source wavelet ONCE so the solver outputs naturally band-
        # limited syn, and we then skip the per-batch syn bandpass entirely.
        # Useful when the user wants to keep syn out of the autograd-filter
        # path (cheaper) or to mirror legacy FWI imaging that applied filter
        # on obs + wavelet rather than on syn.
        filter_target = getattr(imaging, "filter_target", "syn")
        if bandpass_obs and filter_target == "wavelet":
            pad_arg = imaging.filter_padtype
            if pad_arg == "none":
                pad_arg = None
            is_torch = hasattr(wavelet, "detach")
            if is_torch:
                wav_np = wavelet.detach().cpu().numpy()
                wav_dtype = wav_np.dtype
                wav_device = wavelet.device
            else:
                wav_np = np.asarray(wavelet)
                wav_dtype = wav_np.dtype
                wav_device = None
            wav_filt = _bandpass_cpu(
                wav_np, lo=float(lo_hz), hi=float(hi_hz), dt=effective_dt,
                order=int(imaging.filter_order), axis=-1, padtype=pad_arg,
            ).astype(wav_dtype, copy=False)
            wav_filt = np.ascontiguousarray(wav_filt)
            if is_torch:
                wavelet = torch.from_numpy(wav_filt).to(wav_device)
            else:
                wavelet = wav_filt
            if dist_info.is_root:
                peak = float(np.abs(wav_filt).max())
                print(f"[rtm] wavelet bandpass {lo_hz}-{hi_hz} Hz applied "
                      f"(peak now {peak:.3f}) -> syn naturally band-limited; "
                      "syn filter SKIPPED per-batch")
        elif bandpass_obs and filter_target not in ("syn", "wavelet"):
            raise ValueError(
                f"[rtm] imaging.filter_target must be 'syn' or 'wavelet'; "
                f"got {filter_target!r}"
            )

        # ---- 3) Distribute shot batches across DDP ranks ---------------------
        shots_per_batch = max(1, int(imaging.shots_per_batch))
        shot_indices_all = np.arange(nshots, dtype=np.int64)
        all_batches = [
            shot_indices_all[start:start + shots_per_batch]
            for start in range(0, nshots, shots_per_batch)
        ]
        # Round-robin batches across ranks (each rank processes its slice; the
        # final all_reduce SUMs the four accumulators across ranks).
        local_batches = (
            all_batches[dist_info.rank::dist_info.world_size]
            if dist_info.is_distributed else all_batches
        )

        if dist_info.is_root:
            print(f"[rtm] shape={shape} nshots={nshots} shots_per_batch={shots_per_batch} "
                  f"n_batches={len(all_batches)} world_size={dist_info.world_size} "
                  f"normalize_by_illumination={imaging.normalize_by_illumination}")

        # ---- 4) Per-batch local-window context (optional) --------------------
        local_window_ctx = None
        if (spec.local_model_window is not None
                and spec.local_model_window.enabled):
            local_window_ctx = {
                "spec": spec.local_model_window,
                "shape": tuple(shape),
                "dh": float(spec.grid.dh),
                "dt": float(effective_dt),
                "nt": int(effective_nt),
                "solver_cache": {},
            }

        # ---- 5) Output dirs + history CSV ------------------------------------
        out_dir = task_dir / "output"
        qc_dir = task_dir / "qc"
        per_shot_dir = out_dir / "per_shot"
        if dist_info.is_root:
            out_dir.mkdir(exist_ok=True)
            qc_dir.mkdir(parents=True, exist_ok=True)
            if imaging.save_per_shot:
                per_shot_dir.mkdir(parents=True, exist_ok=True)
        _dist.barrier(dist_info)

        # ---- 6) Accumulators -------------------------------------------------
        full_shape = tuple(int(v) for v in shape)
        gradient_sum = np.zeros(full_shape, dtype=np.float32)
        rtm_sum = np.zeros(full_shape, dtype=np.float32)
        source_illum_sum = np.zeros(full_shape, dtype=np.float32)
        receiver_illum_sum = np.zeros(full_shape, dtype=np.float32)
        # Per-shot illumination-normalised accumulators: each batch's
        # gradient / RTM image gets divided by its OWN illumination before
        # the sum, so deep / weakly-illuminated cells are not drowned by
        # high-illumination shallow regions. This is what the legacy Viking
        # imaging step ships as the canonical product
        # (illumination_normalized_rtm_per_shot.npy + the matching FWI
        # gradient variant). Mirrors fwi_workflow.imaging.rtm lines 563-576.
        gradient_per_shot_norm_sum = np.zeros(full_shape, dtype=np.float32)
        rtm_per_shot_norm_sum = np.zeros(full_shape, dtype=np.float32)
        history: list[dict] = []

        history_path = task_dir / "history.csv"
        history_file = None
        history_writer = None
        if dist_info.is_root:
            import csv as _csv
            history_file = history_path.open("w", newline="")
            history_writer = _csv.DictWriter(
                history_file,
                fieldnames=[
                    "batch_index", "shot_count", "shot_id_min", "shot_id_max",
                    "loss", "gradient_rms", "gradient_abs_max",
                    "rtm_abs_max", "local_z0", "local_z1",
                    "local_x0", "local_x1",
                ],
            )
            history_writer.writeheader()
            history_file.flush()

        # ---- 7) Main one-pass loop -------------------------------------------
        # Derive the per-batch loss spec. ``imaging.loss_kind`` overrides
        # ``spec.loss.kind`` (the imaging block is the authoritative source for
        # post-FWI products), but secondary knobs (huber_delta, trace_cosine_eps)
        # still come from ``spec.loss``.
        from sweep_tasks.schemas import LossSpec as _LossSpec
        loss_spec_for_grad = _LossSpec(
            kind=imaging.loss_kind,
            huber_delta=float(spec.loss.huber_delta),
            trace_cosine_demean=bool(imaging.trace_cosine_demean),
            trace_cosine_eps=float(spec.loss.trace_cosine_eps),
        )

        # Bandpass kwargs for the per-batch obs prep (shared by both lazy
        # and eager paths). Wavelet was already pre-filtered above when
        # filter_target == 'wavelet', so we still bandpass obs here either
        # way — only the syn bandpass inside _rtm_process_batch is gated.
        if bandpass_obs:
            _bp_pad = imaging.filter_padtype
            if _bp_pad == "none":
                _bp_pad = None
            bandpass_kwargs = dict(
                lo=float(lo_hz), hi=float(hi_hz),
                dt=float(effective_dt),
                order=int(imaging.filter_order),
                padtype=_bp_pad,
            )
        else:
            bandpass_kwargs = None

        for local_idx, batch_indices in enumerate(local_batches):
            global_batch_index = (
                (local_idx * dist_info.world_size + dist_info.rank)
                if dist_info.is_distributed else local_idx
            )
            try:
                obs_chunk = self._rtm_prep_obs_batch(
                    batch_indices=batch_indices,
                    plan_reader=plan_reader,
                    obs_np=obs_np,
                    obs_native_dt=obs_native_dt,
                    effective_dt=float(effective_dt),
                    effective_nt=int(effective_nt),
                    bandpass_kwargs=bandpass_kwargs,
                    dev=dev,
                )
                self._rtm_process_batch(
                    spec=spec,
                    solver=solver,
                    wavelet=wavelet,
                    sources=sources,
                    receivers=receivers,
                    obs_chunk=obs_chunk,
                    vp_base=vp_base,
                    full_shape=full_shape,
                    dev=dev,
                    batch_indices=batch_indices,
                    global_batch_index=global_batch_index,
                    loss_spec=loss_spec_for_grad,
                    imaging=imaging,
                    bandpass_obs=bandpass_obs,
                    filter_target=filter_target,
                    effective_dt=float(effective_dt),
                    local_window_ctx=local_window_ctx,
                    gradient_sum=gradient_sum,
                    rtm_sum=rtm_sum,
                    source_illum_sum=source_illum_sum,
                    receiver_illum_sum=receiver_illum_sum,
                    gradient_per_shot_norm_sum=gradient_per_shot_norm_sum,
                    rtm_per_shot_norm_sum=rtm_per_shot_norm_sum,
                    history=history,
                    history_writer=history_writer,
                    history_file=history_file,
                    per_shot_dir=per_shot_dir,
                    qc_dir=qc_dir,
                    dist_info=dist_info,
                )
            except Exception as berr:  # noqa: BLE001
                # One bad batch (e.g. boundary saving issue at extreme window
                # shape) should not nuke the whole RTM run. Log + continue.
                import traceback as _tb
                print(f"[rtm] batch {global_batch_index} failed: "
                      f"{type(berr).__name__}: {berr}")
                _tb.print_exc()

            # Periodic live QC re-render (rank 0 only — uses local accumulators
            # which differ between ranks, but the user just wants something to
            # eyeball; the final reduced result lives in qc/ after the loop).
            if (dist_info.is_root
                    and (local_idx + 1) % int(imaging.live_update_every_batches) == 0):
                try:
                    _save_rtm_qc_pngs(
                        gradient_sum=gradient_sum,
                        rtm_sum=rtm_sum,
                        source_illum_sum=source_illum_sum,
                        receiver_illum_sum=receiver_illum_sum,
                        gradient_per_shot_norm_sum=gradient_per_shot_norm_sum,
                        rtm_per_shot_norm_sum=rtm_per_shot_norm_sum,
                        qc_dir=qc_dir,
                        suffix="_latest",
                        eps=float(imaging.illumination_epsilon),
                        normalize=bool(imaging.normalize_by_illumination),
                    )
                except Exception as qerr:  # noqa: BLE001
                    print(f"[rtm] live QC skipped at batch {local_idx + 1}: {qerr}")

        if history_file is not None:
            history_file.close()

        # Close the lazy plan reader (closes SEG-Y handles + frees the cache
        # tensor when cache_all=True). Eager path has nothing to close.
        if plan_reader is not None:
            plan_reader.close()

        # ---- 8) All-reduce across ranks --------------------------------------
        if dist_info.is_distributed:
            for buf in (gradient_sum, rtm_sum,
                        source_illum_sum, receiver_illum_sum,
                        gradient_per_shot_norm_sum, rtm_per_shot_norm_sum):
                t = torch.from_numpy(buf)
                if dev.type == "cuda":
                    t = t.to(dev)
                _dist.all_reduce_sum_inplace(t, dist_info)
                buf[...] = t.detach().cpu().numpy()

        # ---- 9) Save outputs (rank 0 only) -----------------------------------
        artifacts: list[Path] = []
        if dist_info.is_root:
            np.save(out_dir / "fwi_gradient_image.npy", gradient_sum)
            np.save(out_dir / "rtm_image.npy", rtm_sum)
            np.save(out_dir / "source_illumination.npy", source_illum_sum)
            np.save(out_dir / "receiver_illumination.npy", receiver_illum_sum)
            eps = float(imaging.illumination_epsilon)
            denom = np.sqrt(np.maximum(source_illum_sum * receiver_illum_sum, 0.0) + eps)
            grad_norm = (gradient_sum / denom).astype(np.float32)
            rtm_norm = (rtm_sum / denom).astype(np.float32)
            np.save(out_dir / "fwi_gradient_image_normalised.npy", grad_norm)
            np.save(out_dir / "rtm_image_normalised.npy", rtm_norm)
            # Per-shot illumination-normalised products: each shot's image was
            # divided by its OWN sqrt(S*R+eps) BEFORE summation, so deep /
            # weakly-illuminated cells survive the stack. Legacy Viking's
            # canonical RTM display is this variant — strongly recommended
            # over the global-normalised one for plotting.
            np.save(out_dir / "fwi_gradient_image_per_shot_normalised.npy",
                    gradient_per_shot_norm_sum)
            np.save(out_dir / "rtm_image_per_shot_normalised.npy",
                    rtm_per_shot_norm_sum)
            np.savez_compressed(
                out_dir / "rtm_result.npz",
                fwi_gradient_image=gradient_sum,
                rtm_image=rtm_sum,
                source_illumination=source_illum_sum,
                receiver_illumination=receiver_illum_sum,
                fwi_gradient_image_normalised=grad_norm,
                rtm_image_normalised=rtm_norm,
                fwi_gradient_image_per_shot_normalised=gradient_per_shot_norm_sum,
                rtm_image_per_shot_normalised=rtm_per_shot_norm_sum,
                illumination_epsilon=np.float32(eps),
                dh=np.float32(spec.grid.dh),
                dt=np.float32(effective_dt),
                nt=np.int64(effective_nt),
                shape=np.asarray(full_shape, dtype=np.int64),
                n_shots=np.int64(nshots),
                n_batches=np.int64(len(all_batches)),
            )
            artifacts.extend([
                out_dir / "fwi_gradient_image.npy",
                out_dir / "rtm_image.npy",
                out_dir / "source_illumination.npy",
                out_dir / "receiver_illumination.npy",
                out_dir / "fwi_gradient_image_normalised.npy",
                out_dir / "rtm_image_normalised.npy",
                out_dir / "fwi_gradient_image_per_shot_normalised.npy",
                out_dir / "rtm_image_per_shot_normalised.npy",
                out_dir / "rtm_result.npz",
                history_path,
            ])
            try:
                qc_artefacts = _save_rtm_qc_pngs(
                    gradient_sum=gradient_sum,
                    rtm_sum=rtm_sum,
                    source_illum_sum=source_illum_sum,
                    receiver_illum_sum=receiver_illum_sum,
                    gradient_per_shot_norm_sum=gradient_per_shot_norm_sum,
                    rtm_per_shot_norm_sum=rtm_per_shot_norm_sum,
                    qc_dir=qc_dir,
                    suffix="",
                    eps=eps,
                    normalize=bool(imaging.normalize_by_illumination),
                )
                artifacts.extend(qc_artefacts)
            except Exception as qerr:  # noqa: BLE001
                print(f"[rtm] final QC PNG skipped: {qerr}")

            # ---- 9b) Optional post-filter (depth-tapered z-low-cut) ---------
            # Mirrors legacy ``07_filter_imaging.py``. Runs only when the user
            # opts in via ``imaging.post_filter``; outputs land next to the raw
            # npy files as ``<stem>_shallow_zlowcut.{npy,png}``. The same
            # algorithm is exposed standalone as ``sweep-tasks filter-image``
            # so users can iterate on params without re-running the RTM.
            post_filter = getattr(imaging, "post_filter", None)
            if post_filter is not None and bool(post_filter.enabled):
                from sweep_tasks.postproc.filter_image import filter_image_file

                all_targets = [
                    "fwi_gradient_image",
                    "rtm_image",
                    "fwi_gradient_image_normalised",
                    "rtm_image_normalised",
                    "fwi_gradient_image_per_shot_normalised",
                    "rtm_image_per_shot_normalised",
                ]
                if post_filter.targets == "all":
                    pf_targets = all_targets
                else:
                    pf_targets = [str(t) for t in post_filter.targets]
                dh = float(spec.grid.dh)
                nx_full = int(full_shape[1])
                x_max_m = float((nx_full - 1) * dh)
                for stem in pf_targets:
                    src = out_dir / f"{stem}.npy"
                    if not src.is_file():
                        print(f"[rtm] post_filter skip {stem}: {src} not found")
                        continue
                    try:
                        meta = filter_image_file(
                            src,
                            output_dir=out_dir,
                            output_name=f"{stem}_shallow_zlowcut",
                            dz_m=dh,
                            dx_m=dh,
                            wavelength_m=float(post_filter.wavelength_m),
                            depth_m=float(post_filter.depth_m),
                            taper_m=float(post_filter.taper_m),
                            clip_percentile=float(post_filter.clip_percentile),
                            display_scale=float(post_filter.display_scale),
                            x_origin_m=0.0,
                            z_origin_m=0.0,
                            x_max_m=x_max_m,
                            cmap=str(post_filter.cmap),
                            save_png=bool(post_filter.save_png),
                        )
                    except Exception as ferr:  # noqa: BLE001
                        print(f"[rtm] post_filter {stem} skipped: "
                              f"{type(ferr).__name__}: {ferr}")
                        continue
                    out_stem = out_dir / f"{stem}_shallow_zlowcut"
                    artifacts.append(out_stem.with_suffix(".npy"))
                    artifacts.append(Path(meta["removed"]))
                    artifacts.append(Path(meta["z_taper"]))
                    artifacts.append(out_dir / f"{stem}_shallow_zlowcut_metadata.json")
                    if bool(post_filter.save_png):
                        artifacts.append(out_stem.with_suffix(".png"))
                        artifacts.append(out_dir / f"{stem}_shallow_zlowcut_comparison.png")
                    print(f"[rtm] post_filter -> {out_stem.name}.npy "
                          f"(wavelength={post_filter.wavelength_m}m, "
                          f"depth={post_filter.depth_m}m, "
                          f"taper={post_filter.taper_m}m)")

        _dist.barrier(dist_info)

        summary = {
            "n_shots": int(nshots),
            "n_batches": int(len(all_batches)),
            "shots_per_batch": int(shots_per_batch),
            "image_abs_max": float(np.abs(rtm_sum).max()),
            "image_per_shot_norm_abs_max": float(np.abs(rtm_per_shot_norm_sum).max()),
            "gradient_abs_max": float(np.abs(gradient_sum).max()),
            "gradient_per_shot_norm_abs_max": float(np.abs(gradient_per_shot_norm_sum).max()),
            "source_illum_max": float(source_illum_sum.max()),
            "receiver_illum_max": float(receiver_illum_sum.max()),
            "world_size": dist_info.world_size,
            "normalize_by_illumination": bool(imaging.normalize_by_illumination),
        }
        return artifacts, summary

    # -- rtm per-batch obs prep (lazy / eager unified) --------------------

    def _rtm_prep_obs_batch(
        self,
        *,
        batch_indices,
        plan_reader,
        obs_np,
        obs_native_dt: float,
        effective_dt: float,
        effective_nt: int,
        bandpass_kwargs: dict | None,
        dev,
    ):
        """Materialise one shot batch's obs as a torch ``(B, nt, nrec, 1)`` tensor.

        Two routes selected by argument presence:

        * ``plan_reader`` set → **lazy**: read each shot's traces via
          :meth:`PlanReader.read_group`. With ``cache_all=True`` on the
          reader, traces are already in RAM (slice is cheap); with
          ``cache_all=False`` it's a per-batch SEG-Y read.
        * ``obs_np`` set → **eager**: slice the in-RAM
          ``(nshots, nt, nrec, 1)`` tensor produced by ``_fwi_load_obs``.

        Per-batch ops (resample → trim/pad → bandpass) are applied here so
        the lazy path doesn't have to materialise the full obs. Returns a
        contiguous ``torch.float32`` tensor on ``dev``.
        """
        import torch

        batch_arr = np.asarray(batch_indices, dtype=np.int64)
        if plan_reader is not None:
            # (B, nrec, nt_native) from PlanReader
            chunks = [plan_reader.read_group(int(s)) for s in batch_arr]
            raw = np.stack(chunks, axis=0).astype(np.float32, copy=False)
            # Canonical layout: (B, nt, nrec, 1)
            raw = np.ascontiguousarray(raw.transpose(0, 2, 1))[..., None]
        elif obs_np is not None:
            raw = np.take(obs_np, batch_arr, axis=0)
        else:
            raise RuntimeError(
                "[rtm] _rtm_prep_obs_batch: neither plan_reader nor obs_np "
                "provided. This is a runner-internal bug."
            )

        # time_axis on canonical 4-D is axis=1; resample/trim/pad/bandpass
        # helpers default to axis=-1, so pass time_axis explicitly.
        time_axis = 1
        if abs(obs_native_dt - effective_dt) > 1.0e-12:
            raw = _resample_obs_time(raw, obs_native_dt, effective_dt,
                                     time_axis=time_axis)
        raw = _trim_or_pad_time(raw, effective_nt, time_axis=time_axis)

        if bandpass_kwargs is not None:
            raw = _bandpass_cpu(raw, axis=time_axis, **bandpass_kwargs)

        raw = np.ascontiguousarray(raw.astype(np.float32, copy=False))
        return torch.as_tensor(raw, dtype=torch.float32, device=dev)

    # -- rtm per-batch kernel ---------------------------------------------

    def _rtm_process_batch(
        self,
        *,
        spec: "RTMSpec",
        solver,
        wavelet,
        sources,
        receivers,
        obs_chunk,
        vp_base,
        full_shape: tuple,
        dev,
        batch_indices,
        global_batch_index: int,
        loss_spec,
        imaging,
        bandpass_obs: bool,
        filter_target: str = "syn",
        effective_dt: float,
        local_window_ctx,
        gradient_sum,
        rtm_sum,
        source_illum_sum,
        receiver_illum_sum,
        gradient_per_shot_norm_sum=None,
        rtm_per_shot_norm_sum=None,
        history,
        history_writer,
        history_file,
        per_shot_dir: Path,
        qc_dir: Path,
        dist_info,
    ) -> None:
        """Run one shot batch: forward + backward populates both the
        gradient image (= RTM cross-correlation under the matching loss) and
        the solver-side ``source_illumination`` / ``receiver_illumination``
        attributes. No separate ``solver.rtm`` invocation is needed — the
        c-backend's regular adjoint pass owns illumination since the layout
        unification (sweep commit 21041c5).

        Mutates the four accumulator buffers (gradient_sum / rtm_sum /
        *_illum_sum) in-place, writes one history row, and (when
        ``imaging.save_per_shot``) dumps a per-shot npz. The ``rtm_sum`` and
        ``gradient_sum`` accumulators receive the same per-batch image; the
        duplication is kept so downstream consumers expecting the legacy
        filename split (rtm_image.npy vs fwi_gradient_image.npy) still get
        both outputs without surprises.
        """
        import torch

        # ---- Resolve solver + window slice for this batch --------------------
        ndim = len(full_shape)
        if local_window_ctx is None:
            chunk_solver = solver
            if ndim == 3:
                z0, z1 = 0, int(full_shape[0])
                y0, y1 = 0, int(full_shape[1])
                x0, x1 = 0, int(full_shape[2])
                vp_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            else:
                z0, z1 = 0, int(full_shape[0])
                x0, x1 = 0, int(full_shape[1])
                y0, y1 = 0, 0
                vp_slice = (slice(z0, z1), slice(x0, x1))
            chunk_src = sources[batch_indices]
            chunk_rec = receivers[batch_indices]
            chunk_vp_base = vp_base
        else:
            win_spec = local_window_ctx["spec"]
            dh = local_window_ctx["dh"]
            window = _compute_local_window(
                sources[batch_indices], receivers[batch_indices],
                full_shape, dh, win_spec,
            )
            if len(window) == 4:
                z0, z1, x0, x1 = window
                y0, y1 = 0, 0
                local_shape = (z1 - z0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0}
                vp_slice = (slice(z0, z1), slice(x0, x1))
            else:
                z0, z1, y0, y1, x0, x1 = window
                local_shape = (z1 - z0, y1 - y0, x1 - x0)
                rebase_kwargs = {"z0": z0, "x0": x0, "y0": y0}
                vp_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            cache = local_window_ctx["solver_cache"]
            if local_shape not in cache:
                # Mirror _fwi_train_step's cache cap.
                if len(cache) >= 4:
                    oldest = next(iter(cache))
                    del cache[oldest]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                cache[local_shape] = _build_solver(
                    spec.physics, spec.backend, local_shape, dh,
                    local_window_ctx["dt"], local_window_ctx["nt"], dev,
                )
            else:
                cache[local_shape] = cache.pop(local_shape)  # LRU touch
            chunk_solver = cache[local_shape]
            chunk_src, chunk_rec = _rebase_geometry_to_window(
                sources[batch_indices], receivers[batch_indices], **rebase_kwargs,
            )
            chunk_vp_base = vp_base[vp_slice].contiguous()

        # ---- Forward + bandpass syn + loss + backward → FWI grad image ------
        # Opt-in illumination on the new c-core (perf/acoustic-bwd-skip-illum):
        # RTM ALWAYS needs solver.source/receiver_illumination after backward,
        # so enable it unconditionally here. Harmless on older cores that
        # compute illumination regardless.
        try:
            chunk_solver.compute_illumination = True
        except Exception:
            pass
        # Fresh leaf each batch (we discard grads between batches; the RTM
        # accumulator on the caller side does the summation).
        local_vp = chunk_vp_base.detach().clone().requires_grad_(True)
        syn = chunk_solver(wavelet, chunk_src, chunk_rec, models=[local_vp])

        # Syn bandpass is needed ONLY when ``filter_target='syn'``: with
        # ``filter_target='wavelet'`` the wavelet itself was pre-filtered in
        # the driver, so the solver output is already band-limited and an
        # extra autograd-aware filter on syn would just double-apply the
        # response.
        if bandpass_obs and filter_target == "syn":
            syn_filt = _bandpass_syn_torch(
                syn, float(imaging.filter_lowcut_hz),
                float(imaging.filter_highcut_hz),
                effective_dt, order=int(imaging.filter_order),
            )
        else:
            syn_filt = syn

        # ``obs_chunk`` arrives pre-prepared (sliced + resampled + bandpassed
        # + on-device) from the caller's ``_rtm_prep_obs_batch`` — that helper
        # serves both the lazy PlanReader and the eager full-RAM paths so we
        # do not have to materialise a production OBN-scale obs tensor in this loop.

        loss = _compute_loss(syn_filt, obs_chunk, loss_spec).sum()
        loss.backward()
        if local_vp.grad is None:
            raise RuntimeError(
                f"[rtm] batch {global_batch_index}: vp.grad is None — backward "
                "did not populate the gradient image."
            )
        gradient_local = (
            local_vp.grad.detach().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .cpu().numpy().astype(np.float32)
        )
        loss_value = float(loss.detach().cpu())

        # ---- Illumination snapshot ------------------------------------------
        # The c-backend forward call zeros + (re)populates
        # ``solver.source_illumination`` and ``solver.receiver_illumination``
        # during the backward pass.  No separate ``solver.rtm`` invocation is
        # required — the gradient we just took IS the imaging-condition
        # cross-correlation (gradient = RTM image under the chosen loss).
        sill_attr = getattr(chunk_solver, "source_illumination", None)
        rill_attr = getattr(chunk_solver, "receiver_illumination", None)
        if sill_attr is None or rill_attr is None:
            raise RuntimeError(
                "[rtm] backend did not expose solver.source_illumination / "
                "receiver_illumination after backward. Confirm backend.impl='c' "
                "(the eager autograd backend does not populate these)."
            )

        target_shape = tuple(int(v) for v in local_vp.shape)
        src_illum = _crop_padded_volume_to_model(
            sill_attr.detach(), target_shape, spec.physics,
        )
        rec_illum = _crop_padded_volume_to_model(
            rill_attr.detach(), target_shape, spec.physics,
        )

        # RTM image == FWI gradient under the chosen matching loss.  Keep the
        # two output arrays identical so downstream consumers that expect the
        # legacy filename split (rtm_image.npy vs fwi_gradient_image.npy) keep
        # working without surprises.
        rtm_image = gradient_local

        # ---- Scatter local batch products into the full-grid accumulators ----
        gradient_sum[vp_slice] += gradient_local
        rtm_sum[vp_slice] += rtm_image
        source_illum_sum[vp_slice] += src_illum
        receiver_illum_sum[vp_slice] += rec_illum

        # Per-shot illumination-normalised contribution: divide this batch's
        # image by its OWN sqrt(S*R+eps) BEFORE adding to the stack. Cleans
        # deep / weakly-illuminated cells that the global-normalised stack
        # buries under shallow energy (legacy
        # ``illumination_*_per_shot.npy``). Optional buffers — only filled
        # when the caller passes them.
        if gradient_per_shot_norm_sum is not None or rtm_per_shot_norm_sum is not None:
            eps_ps = float(imaging.illumination_epsilon)
            denom_local = np.sqrt(
                np.maximum(src_illum.astype(np.float64) * rec_illum.astype(np.float64), 0.0)
                + eps_ps
            ).astype(np.float32)
            if gradient_per_shot_norm_sum is not None:
                gradient_per_shot_norm_sum[vp_slice] += (
                    gradient_local / denom_local
                ).astype(np.float32)
            if rtm_per_shot_norm_sum is not None:
                rtm_per_shot_norm_sum[vp_slice] += (
                    rtm_image / denom_local
                ).astype(np.float32)

        # ---- History row -----------------------------------------------------
        grad_abs_max = float(np.abs(gradient_local).max())
        grad_rms = float(np.sqrt(np.mean(gradient_local.astype(np.float64) ** 2)))
        rtm_abs_max = float(np.abs(rtm_image).max())
        entry = {
            "batch_index": int(global_batch_index),
            "shot_count": int(np.asarray(batch_indices).size),
            "shot_id_min": int(np.min(batch_indices)),
            "shot_id_max": int(np.max(batch_indices)),
            "loss": loss_value,
            "gradient_rms": grad_rms,
            "gradient_abs_max": grad_abs_max,
            "rtm_abs_max": rtm_abs_max,
            "local_z0": int(z0), "local_z1": int(z1),
            "local_x0": int(x0), "local_x1": int(x1),
        }
        history.append(entry)
        if history_writer is not None and dist_info.is_root:
            history_writer.writerow(entry)
            if history_file is not None:
                history_file.flush()

        # ---- Optional per-shot npz dump -------------------------------------
        if imaging.save_per_shot and dist_info.is_root:
            for k, shot_id in enumerate(np.asarray(batch_indices)):
                np.savez_compressed(
                    per_shot_dir / f"shot_{int(shot_id):05d}.npz",
                    gradient=gradient_local,
                    rtm_image=rtm_image,
                    source_illumination=src_illum,
                    receiver_illumination=rec_illum,
                    shot_id=np.int64(int(shot_id)),
                    z0=np.int32(z0), z1=np.int32(z1),
                    x0=np.int32(x0), x1=np.int32(x1),
                    loss=np.float32(loss_value),
                )

        if ndim == 3:
            win_str = f"z[{z0},{z1})y[{y0},{y1})x[{x0},{x1})"
        else:
            win_str = f"z[{z0},{z1})x[{x0},{x1})"
        print(f"[rtm] batch {global_batch_index:04d} "
              f"shots={batch_indices.size} loss={loss_value:.6e} "
              f"grad_rms={grad_rms:.3e} rtm_max={rtm_abs_max:.3e} "
              f"window={win_str}", flush=True)

        # Free per-batch tensors so subsequent batches see a clean allocator.
        del syn, syn_filt, obs_chunk, local_vp
        if dev.type == "cuda":
            torch.cuda.empty_cache()
