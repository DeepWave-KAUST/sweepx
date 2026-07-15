"""Frequency-selection (source-encoding) FWI task runner (mixin). Verbatim from runner.py."""
from pathlib import Path
import os
from sweep_tasks.schemas import FWISpec
from sweep_tasks._helpers.dd import (
    _dd_backward_tile,
    _dd_config,
    _dd_render_tile,
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
from sweep_tasks._helpers.stages import (
    _normalise_stage_list,
    _resample_vp_tensor,
    _shape_for_dh,
)
from sweep_tasks._helpers.util import (
    _apply_seed,
    _resolve_device,
)


class FreqselRunnerMixin:
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
        grid via ``update_base_velocity``; net params + Adam state kept), runs
        ``stage.epochs`` iterations appending to ``losses/times/peaks``. Returns
        ``(net, optimizer, ny_, nx_, chk)``. Single-stage runs reduce to the
        original single-band behaviour bit-for-bit.
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
        if abs(float(dh) - float(native_dh)) < 1e-9:
            vp0 = vp0_native
        else:
            new_shape = _shape_for_dh(vp0_native.shape, native_dh, float(dh))
            vp0 = _resample_vp_tensor(
                torch.tensor(vp0_native), new_shape).detach().cpu().numpy()
        gshape = vp0.shape
        nz, ny_, nx_ = gshape
        nyp = -(-ny_ // dd_py) * dd_py if dd_on else ny_
        nxp = -(-nx_ // dd_px) * dd_px if dd_on else nx_
        vp0p = np.pad(vp0, ((0, 0), (0, nyp - ny_), (0, nxp - nx_)),
                      mode="edge")
        frozen = vp0p == float(getattr(spec.reparam, "water_vp_m_s", 1500.0)
                               if spec.reparam else 1500.0)
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
                if abs(float(dh) - float(native_dh)) < 1e-9:
                    a0 = native_arr
                else:
                    _ns = _shape_for_dh(native_arr.shape, native_dh, float(dh))
                    a0 = _resample_vp_tensor(torch.tensor(native_arr), _ns
                                             ).detach().cpu().numpy()
                a0p = np.pad(a0, ((0, 0), (0, nyp - ny_), (0, nxp - nx_)), mode="edge")
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
                param_bases.append(_prep_base(np.load(_ref.path).astype(np.float32)))

        solver = _build_solver(spec.physics, spec.backend,
                               (nz, nyp, nxp), float(dh),
                               float(dt), nt, dev)
        if dd_on:
            from sweep.parallel import MeshTopology
            mesh = MeshTopology(py=dd_py, px=dd_px, shot_groups=1,
                                world_size=world, rank=rank)
            solver = _dd_wrap(solver, mesh)

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
            if vpt.shape != gshape:
                # Resample the true model to this stage's grid (mirrors the init
                # resample) so synthesize_from_true supports multi-stage ladders.
                vpt = _resample_vp_tensor(
                    torch.tensor(vpt), gshape).detach().cpu().numpy().astype(np.float32)
            n_nodes = int(fspec.synth_n_nodes)
            sx = np.linspace(8, nx_ - 9, n_nodes).astype(np.int64)
            nodes = np.stack([sx, np.full(n_nodes, ny_ // 2, np.int64),
                              np.full(n_nodes, fspec.synth_node_z,
                                      np.int64)], -1)
            rx = np.arange(4, nx_ - 4, int(fspec.synth_rec_stride))
            recs = np.stack([rx, np.full(len(rx), ny_ // 2, np.int64),
                             np.zeros(len(rx), np.int64)], -1)
            shard = str(task_dir / f"freqsel_synth_obs_s{si}.npz")
            t = np.arange(nt, dtype=np.float64) * comb.dt
            f0 = 1.5 * float(np.mean(comb.freqs))
            a = np.pi * f0 * (t - 0.25)
            ricker = ((1 - 2 * a * a) * np.exp(-a * a)).astype(np.float32)
            vpt_t = torch.tensor(
                np.pad(vpt, ((0, 0), (0, nyp - ny_), (0, nxp - nx_)), mode="edge"),
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
        targets = fsl.FreqSelTargets(shards_glob, comb, ny_,
                                     verbose=rank == 0)
        sched = fsl.PoolScheduler(targets.node_grid, int(fspec.n_pools),
                                  comb.n_bins, seed=int(spec.seed) + 17,
                                  random_batch=getattr(fspec, "random_batch", None))
        rec_table = targets.union_xyz[None]
        if rank == 0:
            print(f"[freqsel] stage {si}: {targets.n_nodes} nodes, "
                  f"{targets.n_items} items, {targets.n_union} union cells, "
                  f"{sched.n_pools} pools, comb {comb.freqs[0]:.4f}-"
                  f"{comb.freqs[-1]:.4f} Hz, nt={nt}, dh={dh}, dt={dt}",
                  flush=True)

        # ---- parameterisation: grid vp or reparam INR ---------------------
        vp = None
        if si == 0:
            if use_reparam:
                net = _build_reparam_net(spec.reparam, base_t,
                                         spec.model_bounds.get("vp"),
                                         param_bases=param_bases)
                net = net.to(dev)
                if dd_on:
                    import torch.distributed as dist
                    for p in net.parameters():
                        dist.broadcast(p.data, src=0)
                optimizer = torch.optim.Adam(net.parameters(),
                                             lr=float(spec.reparam.lr))
            else:
                vp = base_t.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([vp], lr=float(spec.optimizer.lr))
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
                _wm = ((base_t.detach() == _wvp)
                       if bool(getattr(spec.reparam, "mask_water_layer", False))
                       else None)
                if free_multi:
                    # Multi-parameter: carry the shared trunk, resample EVERY
                    # channel's base to the new grid (param_bases was rebuilt
                    # for this stage's dh above). vp (channel 0) drives _wm.
                    net.update_base_models(param_bases, water_mask=_wm)
                else:
                    net.update_base_velocity(base_t, water_mask=_wm)
                if bool(getattr(stage, "optimizer_reset", False)):
                    optimizer = torch.optim.Adam(net.parameters(),
                                                 lr=float(spec.reparam.lr))
            else:
                # grid-vp mode: Adam state is shape-bound, must rebuild.
                vp = base_t.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([vp], lr=float(spec.optimizer.lr))
        # per-stage lr scale on the carried optimizer (INR: inr_lr_scale)
        _scale = (float(stage.inr_lr_scale) if use_reparam
                  else float(stage.lr_scale))
        if abs(_scale - 1.0) > 1e-12:
            for g in optimizer.param_groups:
                g["lr"] = float(spec.reparam.lr if use_reparam
                                else spec.optimizer.lr) * _scale

        _b = spec.model_bounds.get("vp") if spec.model_bounds else None
        vmin = float(_b.min) if _b is not None and _b.min is not None else None
        vmax = float(_b.max) if _b is not None and _b.max is not None else None

        loss_fn = fsl.SteadyGCNLoss(
            comb, targets, int(fspec.steady_samples),
            int(fspec.slack_samples), dev, distributed=dd_on,
            eps=float(fspec.eps))

        _dd_rc = _dd_config()[3] if dd_on else 0
        # Multi-parameter reparam (VRZ option C etc.): the net has one output
        # channel per freed solver-model parameter (free_multi / n_free were set
        # above). DD needs per-channel tile render/backward — not yet wired.
        if free_multi and dd_on:
            raise NotImplementedError(
                "reparam.free_params (multi-parameter INR) is not yet wired for "
                "domain decomposition (per-channel tile render/backward); run it "
                "on a single GPU / non-DD config.")

        def _leaf():
            if not use_reparam:
                return vp
            if dd_on:
                # Mirror the multisource DD path (_dd_render_tile): render ONLY
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
        targets.bind_ownership(
            np.arange(targets.n_union) if own is None else own, dev)
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
            g = leaf.grad
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
                # Joint N-channel reparam backward (single GPU): each freed
                # parameter's own leaf carries dL/d(that parameter); map them all
                # onto the shared trunk in one chunked pass. DD is guarded out
                # above; illumination precond is not supported on this path.
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
                # Tile-grad path (mirrors multisource _dd_backward_tile): g
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
                print(f"[freqsel] s{si} it {it:4d} pool {it % sched.n_pools:2d}"
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
                        m[:, :ny_, :nx_].cpu().numpy())
                if use_reparam and (spec.reparam.save_net
                                    or os.environ.get("SWEEP_SAVE_REPARAM_NET") == "1"):
                    torch.save(net.state_dict(),
                               task_dir / f"reparam_net_iter{gi + 1:04d}.pt")
                np.savez(task_dir / "curves.npz",
                         losses=np.array(losses), iter_s=np.array(times),
                         peak_gb=np.array(peaks))

        # keep a handle to the final grid-vp so the caller can render output
        self._freqsel_last_vp = vp
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
        params + Adam state kept). A single-stage run is bit-identical to the
        original single-band path.
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
                # from a spurious SIGABRT (observed at 2-16Hz with 12 c2f
                # levels). 1800 s default, env-overridable.
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
        vp0_native = np.load(_vp_ref.path).astype(np.float32)
        if vp0_native.ndim != 3:
            raise ValueError("freqsel path is 3-D (use a thin-slab volume "
                             f"for 2-D tests); init shape {vp0_native.shape}")
        native_dh = float(spec.grid.dh)

        task_dir.mkdir(parents=True, exist_ok=True)
        net = optimizer = None
        losses, times, peaks = [], [], []
        chk_first = None
        ny_ = nx_ = None
        for si, stage in enumerate(stages):
            fspec = getattr(stage, "frequency", None) or fspec_global
            if fspec is None:
                raise ValueError(
                    "frequency_selection: stage has no frequency sub-spec "
                    "and source_encoding.frequency is unset")
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

        artifacts, summary = {}, {}
        if rank == 0:
            if use_reparam:
                with torch.no_grad():
                    m = net.render(chunk_rows=chunk_rows).detach()
            else:
                m = self._freqsel_last_vp.detach()
            final = m[:, :ny_, :nx_].cpu().numpy()
            np.save(task_dir / "inverted_vp.npy", final)
            # Multi-parameter reparam: also dump every freed channel (z, …).
            if use_reparam and getattr(spec.reparam, "free_params", None):
                with torch.no_grad():
                    allf = net.render_all(chunk_rows=chunk_rows).detach()
                for _i, _fp in enumerate(spec.reparam.free_params):
                    np.save(task_dir / f"inverted_{_fp.name}.npy",
                            allf[_i, :, :ny_, :nx_].cpu().numpy())
            np.savez(task_dir / "curves.npz", losses=np.array(losses),
                     iter_s=np.array(times), peak_gb=np.array(peaks))
            summary = {
                "mode": "frequency_selection",
                "reparam": bool(use_reparam),
                "stages": len(stages),
                "nt": int(times and 0 or 0),
                "steady_check": float(chk_first)
                if chk_first is not None else 0.0,
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
            # multisource path; freqsel writes to task_dir root (alongside
            # inverted_vp.npy), not task_dir/output.
            if (use_reparam and net is not None
                    and (spec.reparam.save_net
                         or os.environ.get("SWEEP_SAVE_REPARAM_NET") == "1")):
                net_path = task_dir / "reparam_net.pt"
                torch.save(net.state_dict(), net_path)
                artifacts["reparam_net"] = str(net_path)
                print(f"[freqsel] saved reparam net -> {net_path}", flush=True)
            print(f"[freqsel] DONE ({len(stages)} stage(s)) "
                  f"mean(1-GCN) {losses[0]:.4f} -> {losses[-1]:.4f}",
                  flush=True)
        return artifacts, summary
