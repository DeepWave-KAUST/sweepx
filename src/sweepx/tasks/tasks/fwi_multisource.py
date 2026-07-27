"""Multisource (streaming CRG/plan) FWI task runner (mixin). Verbatim from runner.py."""
from pathlib import Path
from sweep_tasks.preproc.filter import bandpass_torch as _bandpass_torch_fft
from sweep_tasks.preproc.filter import (
    apply_receiver_smoothing as _apply_receiver_smoothing,
    receiver_smoothing_matrix as _receiver_smoothing_matrix,
)
import math
import numpy as np
import os
import time
from sweep_tasks.schemas import FWISpec
from sweep_tasks._helpers.bounds import _effective_bound
from sweep_tasks._helpers.dd import (
    _dd_backward_tile,
    _dd_config,
    _dd_render_tile,
    _dd_tile_bounds,
    _dd_wrap,
)
from sweep_tasks._helpers.illumination import _apply_illumination_precond
from sweep_tasks._helpers.loss import (
    _compute_loss,
    _diving_asinh_for_nodes,
    _diving_window_mask,
)
from sweep_tasks._helpers.metadata import _dump_run_metadata
from sweep_tasks._helpers.optimizer import (
    _build_optimizer,
    _build_reparam_optimizer,
)
from sweep_tasks._helpers.plan_apply import _normalize_fwi_init_models
from sweep_tasks._helpers.plotting import (
    _dump_diving_window_qc,
    _dump_receiver_rotation_qc,
    _plot_loss_curve,
)
from sweep_tasks._helpers.reparam import (
    _advance_hash_schedule,
    _build_reparam_net,
    _has_hash_schedule,
    _render_full_to_cpu_tiled,
)
from sweep_tasks._helpers.solver_build import (
    _build_solver,
    _cfl_check,
)
from sweep_tasks._helpers.stages import (
    _normalise_stage_list,
    _resample_vp_tensor,
)
from sweep_tasks._helpers.util import (
    _apply_seed,
    _resolve_device,
)
from sweep_tasks._helpers.wavelet_build import (
    _build_wavelet,
    _get_wavelet_source_delay_s,
)


class MultisourceRunnerMixin:
    def _run_fwi_multisource(self, spec: FWISpec, task_dir: Path):
        """3-D OBN FWI driver — unified plan + UTM rotation + source encoding.

        Implements the production OBN 1-GPU path on the unified plan stack:
        every iteration draws a shared-shots batch from a
        ``grouping='crg'`` :class:`sweep_io.seismic_plan.SeismicPlan`,
        encodes it as a ±1-signed supershot, runs one ``Acoustic3D``
        forward + adjoint with ``source_encoding=True``, and steps the
        optimizer. Per-shot multi-GPU CRG mode (without encoding) is not
        yet wired and will raise a clear error.

        Setup deviations from :meth:`_run_fwi` (the static-obs CSG path):

        * Geometry comes from :class:`FromPlanGeometry` configured with
          ``rotation_metadata`` + (optionally) ``dh_xyz_m`` /
          ``grid_origin_xyz_m`` / ``auto_origin_pad_cells``. UTM source /
          shot positions get rotated via
          :class:`sweep_io.geometry.RotatedFrame` into the model frame,
          then quantised to grid indices.
        * Obs is streamed per iter via
          :class:`sweep_io.seismic_plan.PlanReader.read_rows` (one SEG-Y
          read pass per iter). No giant ``(nshots, nrec, nt)`` tensor
          materialised at task start.
        * Source encoding flips ``signs ∈ {-1, +1}^B`` per iter; the
          wavelet becomes ``(1, B, nt)`` and obs becomes
          ``Σ_i sign_i · obs_i`` of shape ``(1, n_shared, nt)`` — one
          forward call per iter regardless of B.

        Activated via the ``obs.plan.sampling`` block — when that's set on
        a ``grouping='crg'`` plan the runner dispatches here; otherwise it
        runs the static-obs CSG loop in :meth:`_run_fwi`.
        """
        import torch

        from sweep_io.geometry import load_rotation_metadata
        from sweep_io.seismic_plan import (
            PlanReader,
            SeismicPlan,
            build_shotkey_to_nodes_index,
            precompute_group_unique_keys,
            sample_percrg_independent,
            sample_shared_shots_from_plan,
            sample_shared_shots_receiver_first,
        )
        from sweep_tasks.runtime import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        if spec.obs.plan is None or spec.obs.plan.sampling is None:
            raise ValueError(
                "_run_fwi_multisource requires obs.plan.sampling to be set "
                "(PlanSamplingConfig — see ObsPlanConfig.sampling)."
            )
        sampling_cfg = spec.obs.plan.sampling
        if Path(spec.geometry.plan_path) != Path(spec.obs.plan.plan_path):
            raise ValueError(
                "geometry.plan_path and obs.plan.plan_path must point at "
                f"the same file; got {spec.geometry.plan_path!r} vs "
                f"{spec.obs.plan.plan_path!r}."
            )

        _apply_seed(spec.seed)
        dd_on, dd_py, dd_px, dd_rc = _dd_config()
        dev = (_dist.resolve_dist_device(spec.device, dist_info.local_rank)
               if dd_on else _resolve_device(spec.device))

        encoding_spec = spec.source_encoding
        encoding_on = bool(encoding_spec is not None and encoding_spec.enabled)
        # B.2 (2026-05-22): per-shot multi-GPU CRG path. When
        # ``source_encoding.enabled=false`` (or absent), the forward loop
        # below loops over the B OBN groups as a per-shot batch
        # (``solver(..., source_encoding=False)`` with wavelet shape
        # ``(B_local, 1, nt)``) instead of collapsing them into a ±1
        # supershot. Under torchrun, each rank gets a contiguous slice of
        # the B batch — ``B_local = ceil(B / world_size)`` shots per rank
        # — and the vp gradient is all-reduce'd after backward.
        if encoding_on and dist_info.is_distributed and not dd_on:
            raise NotImplementedError(
                "Encoded supershot FWI is 1-GPU by construction. Run with "
                "source_encoding.enabled=false to use the per-shot multi-"
                "GPU path, or run on a single GPU (no torchrun)."
            )
        dd_mesh = None
        if dd_on:
            from sweep.parallel.mesh import MeshTopology
            if not encoding_on:
                raise ValueError(
                    "SWEEP_DD_ENABLE=1 requires source_encoding.enabled=true "
                    "(DD v1 = encoded supershot, shot_groups=1).")
            _w, _r = int(dist_info.world_size), int(dist_info.rank)
            if _w != dd_py * dd_px:
                raise ValueError(
                    f"DD world_size {_w} != SWEEP_DD_PY*PX {dd_py}*{dd_px}.")
            dd_mesh = MeshTopology(py=dd_py, px=dd_px, shot_groups=1,
                                   world_size=_w, rank=_r)
            print(f"[dd] ModelParallel ON: world={_w} py={dd_py} px={dd_px} "
                  f"render_chunk={dd_rc} (illum/smooth/seabed/cudagraph "
                  f"disabled in DD v1)", flush=True)
        if int(sampling_cfg.shared_shots_per_iter) == 0:
            raise ValueError(
                "obs.plan.sampling.shared_shots_per_iter > 0 is required "
                "for both the encoded supershot path and the per-shot path "
                "(every batch group needs a shared-receiver set)."
            )

        # Setup-phase timer: each major step prints its wall time so the
        # iter-loop profile isn't bottlenecked by setup costs we can't see.
        _t_setup0 = time.perf_counter()
        def _stage(label: str, t_start: float) -> float:
            now = time.perf_counter()
            print(f"[multisource] setup {label}: {now - t_start:.2f}s "
                  f"(cumulative {now - _t_setup0:.2f}s)")
            return now

        # --- 1) Load plan + rotation, apply min_coverage filter.
        _t = time.perf_counter()
        plan = SeismicPlan.load(spec.geometry.plan_path)
        _t = _stage(f"SeismicPlan.load (n_rows={plan.n_rows:,}, n_groups={plan.n_groups})", _t)
        if plan.grouping != "crg":
            raise ValueError(
                "_run_fwi_multisource requires a grouping='crg' SeismicPlan "
                f"(got {plan.grouping!r}). Use 'sweep-tasks build-plan "
                "--grouping crg --receiver-quantize-m ...' to build one."
            )
        if spec.geometry.rotation_metadata is None:
            raise ValueError(
                "Multisource FWI on a CRG plan requires "
                "geometry.rotation_metadata (UTM → model-frame transform). "
                "Set FromPlanGeometry.rotation_metadata to the dataset's "
                "rotation_metadata.json."
            )
        min_cov = int(sampling_cfg.min_coverage)
        if encoding_spec is not None:
            min_cov = max(min_cov, int(encoding_spec.min_coverage))
        # Per-CRG independent coverage: each iter gives every node its OWN
        # sub-sampled rows (no shared-shot intersection). Only meaningful on
        # the per-shot (non-encoded) path — the encoded supershot NEEDS a
        # shared receiver grid to sum ``Σ sign_i·obs_i``. See PerCRGBatch.
        per_crg_independent = (
            bool(getattr(sampling_cfg, "per_crg_independent", False))
            and not encoding_on
        )
        if per_crg_independent and dist_info.is_root:
            print("[multisource] per_crg_independent=ON — each node inverts its "
                  "own aperture (ragged rows, padded+masked; no shared "
                  "intersection).", flush=True)
        # NOTE: we do NOT pre-filter the plan by min_coverage at this point.
        # `sample_shared_shots_from_plan` accepts min_coverage as a sampling
        # parameter and applies it as a cheap per-group-row-counts mask
        # each iter (microseconds). Pre-filtering via plan.filter_groups
        # rewrites the entire multi-GB plan even when only a few
        # groups are dropped — costs minutes of setup on a large survey
        # and saves zero solver compute.
        if min_cov > 0:
            n_below = int((plan.per_group_row_counts() < min_cov).sum())
            print(f"[multisource] min_coverage={min_cov}: deferred to sampler "
                  f"(would drop {n_below}/{plan.n_groups} groups; pre-filtering "
                  "the plan is too expensive on a multi-GB CRG layout).")
        frame = load_rotation_metadata(spec.geometry.rotation_metadata)
        _t = _stage("load_rotation_metadata", _t)

        # --- 2) Init model + shape + dh.
        init_models = _normalize_fwi_init_models(spec)
        if init_models[0].path is None:
            raise ValueError(
                "Multisource FWI requires init_model.path (a 3-D vp "
                "npy); ModelRef.constant is not supported on this path."
            )
        init_vp_np = np.load(init_models[0].path).astype(np.float32)
        if init_vp_np.ndim != 3:
            raise ValueError(
                f"CRG-plan FWI expects a 3-D init_model (nz, ny, nx); "
                f"got shape {init_vp_np.shape}"
            )
        dh_xyz = spec.geometry.dh_xyz_m
        if dh_xyz is None:
            dh = float(spec.grid.dh)
            dh_xyz = (dh, dh, dh)
        dz_m, dy_m, dx_m = float(dh_xyz[0]), float(dh_xyz[1]), float(dh_xyz[2])

        # --- 3) Pre-project UTM → model frame xy.
        # ONE rotation pass per array. Rotation is a numpy matmul on every
        # point — was previously called 2-3 times (auto-origin, initial
        # projection, post-filter re-projection); each call took tens of seconds
        # on a large plan, even Lustre-warm, summing to minutes of dead
        # setup time. Now we project once, reuse via fancy-indexing.
        # MOVED UP (was post-crop): the auto-origin is now computed
        # before the model_plan crop so the crop can be origin-aware
        # (see the fix below). Same data, same single rotation pass —
        # just relocated.
        plan_model_xy = frame.to_model(plan.row_source_xyz[:, :2])
        _t = _stage(f"frame.to_model on plan.row_source_xy ({plan.n_rows:,} pts)", _t)
        src_model_xy = frame.to_model(plan.group_xyz[:, :2])
        _t = _stage(f"frame.to_model on plan.group_xy ({plan.n_groups} pts)", _t)

        # --- 3b) Grid origin (auto from rotated bbox if not given). The
        # init_vp_np grid uses THIS same origin (it's built with the same
        # auto-pad rule); the model_plan crop below converts user-space
        # x/y/z windows in model meters → cell indices via this origin.
        origin = spec.geometry.grid_origin_xyz_m
        pad_z, pad_y, pad_x = (int(p) for p in spec.geometry.auto_origin_pad_cells)
        if origin is None:
            x_min = float(min(src_model_xy[:, 0].min(), plan_model_xy[:, 0].min()))
            y_min = float(min(src_model_xy[:, 1].min(), plan_model_xy[:, 1].min()))
            # z-origin: the model top IS the free surface (sea surface), which
            # by convention is the first layer (index 0 = 0 m datum). Do NOT
            # pad above it — there is nothing to model above the sea surface,
            # and a z-pad would shift the model_plan z-crop downward, silently
            # chopping the top of the water column and dropping the free
            # surface / receivers ~pad_z*dz below where they belong. Only the
            # lateral x/y axes get the PML buffer pad.
            origin = (
                0.0,
                y_min - pad_y * dy_m,
                x_min - pad_x * dx_m,
            )
            print(f"[multisource] auto grid_origin_xyz_m = "
                  f"({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}) "
                  f"(z=0 at sea surface; x/y padded by auto_origin_pad_cells)")
            _t = _stage("auto grid_origin (z=0 surface; x/y bbox.min - pad)", _t)
        init_origin_z = float(origin[0])
        init_origin_y = float(origin[1])
        init_origin_x = float(origin[2])

        # --- 2b) Optional ModelPlan: crop vp to inversion window. The
        # crop is ORIGIN-AWARE: ``x_lo = floor((x_window - origin_x) / dh)``
        # so the data slice corresponds to the requested model-meter
        # window. Earlier versions computed ``x_lo = floor(x_window / dh)``
        # (no origin shift) — with a negative auto-origin (the edge pad) this
        # shifted the actual grid coverage left by the pad,
        # silently chopping the OBN nodes on the right edge of every run.
        nz_pre, ny_pre, nx_pre = init_vp_np.shape
        z_lo, z_hi = 0, nz_pre
        y_lo, y_hi = 0, ny_pre
        x_lo, x_hi = 0, nx_pre
        crop_origin_offset = (0.0, 0.0, 0.0)
        if spec.model_plan is not None:
            mp = spec.model_plan
            z_lo = int(np.floor((mp.z_window_m[0] - init_origin_z) / dz_m)) if mp.z_window_m else 0
            z_hi = int(np.ceil((mp.z_window_m[1] - init_origin_z) / dz_m)) + 1 if mp.z_window_m else nz_pre
            y_lo = int(np.floor((mp.y_window_m[0] - init_origin_y) / dy_m)) if mp.y_window_m else 0
            y_hi = int(np.ceil((mp.y_window_m[1] - init_origin_y) / dy_m)) + 1 if mp.y_window_m else ny_pre
            x_lo = int(np.floor((mp.x_window_m[0] - init_origin_x) / dx_m)) if mp.x_window_m else 0
            x_hi = int(np.ceil((mp.x_window_m[1] - init_origin_x) / dx_m)) + 1 if mp.x_window_m else nx_pre
            z_lo = max(0, z_lo); z_hi = min(nz_pre, z_hi)
            y_lo = max(0, y_lo); y_hi = min(ny_pre, y_hi)
            x_lo = max(0, x_lo); x_hi = min(nx_pre, x_hi)
            if z_hi <= z_lo or y_hi <= y_lo or x_hi <= x_lo:
                raise ValueError(
                    f"model_plan window collapsed: "
                    f"z=[{z_lo},{z_hi}) y=[{y_lo},{y_hi}) x=[{x_lo},{x_hi})"
                )
            init_vp_np = init_vp_np[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].copy()
            crop_origin_offset = (z_lo * dz_m, y_lo * dy_m, x_lo * dx_m)
            print(f"[multisource] model_plan crop: vp shape "
                  f"({nz_pre},{ny_pre},{nx_pre}) -> {init_vp_np.shape}, "
                  f"crop_origin_offset += ({crop_origin_offset[0]:.1f}, "
                  f"{crop_origin_offset[1]:.1f}, {crop_origin_offset[2]:.1f}) m "
                  f"(model-x window covered: "
                  f"[{init_origin_x + crop_origin_offset[2]:.1f}, "
                  f"{init_origin_x + crop_origin_offset[2] + init_vp_np.shape[-1] * dx_m:.1f}) m)")
        shape = tuple(int(v) for v in init_vp_np.shape)
        if dd_on:
            # DD v1 needs uniform tiles: edge-replicate the HIGH x/y edge so
            # nx % px == 0 and ny % py == 0. Padding is appended past the survey
            # edge, so the origin and every source/receiver coord stay valid; the
            # extra cells carry no sources/receivers (PML-absorbed).
            _pz, _pny, _pnx = init_vp_np.shape
            _ny_pad, _nx_pad = (-_pny) % dd_py, (-_pnx) % dd_px
            if _ny_pad or _nx_pad:
                init_vp_np = np.pad(
                    init_vp_np, ((0, 0), (0, _ny_pad), (0, _nx_pad)), mode="edge")
                shape = tuple(int(v) for v in init_vp_np.shape)
                print(f"[dd] padded grid to uniform tiles: "
                      f"({_pz},{_pny},{_pnx}) -> {shape} (px={dd_px} py={dd_py})",
                      flush=True)
        nz, ny, nx = shape

        # Final cropped-frame origin (used by all grid-index projections).
        origin_z = init_origin_z + crop_origin_offset[0]
        origin_y = init_origin_y + crop_origin_offset[1]
        origin_x = init_origin_x + crop_origin_offset[2]

        # --- 4) Compute integer grid indices (no more rotations).
        plan_grid_x = np.rint((plan_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64)
        plan_grid_y = np.rint((plan_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64)
        plan_grid_z = np.rint(
            (plan.row_source_xyz[:, 2] - origin_z) / dz_m
        ).astype(np.int64)
        src_grid_x = np.rint((src_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64)
        src_grid_y = np.rint((src_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64)
        src_grid_z = np.rint(
            (plan.group_xyz[:, 2] - origin_z) / dz_m
        ).astype(np.int64)
        # Sweep's source/receiver coord convention is (x, y, z) along the
        # last axis (matches Acoustic3D's grid).
        plan_grid_xyz = np.stack([plan_grid_x, plan_grid_y, plan_grid_z], axis=-1)
        src_grid_xyz = np.stack([src_grid_x, src_grid_y, src_grid_z], axis=-1)
        _t = _stage("grid index projection (rint + stack)", _t)

        # When the grid is smaller than the rotated survey extent, drop
        # rows/groups whose source position falls outside.
        row_in = (
            (plan_grid_xyz[:, 0] >= 0) & (plan_grid_xyz[:, 0] < nx)
            & (plan_grid_xyz[:, 1] >= 0) & (plan_grid_xyz[:, 1] < ny)
            & (plan_grid_xyz[:, 2] >= 0) & (plan_grid_xyz[:, 2] < nz)
        )
        slot_in = (
            (src_grid_xyz[:, 0] >= 0) & (src_grid_xyz[:, 0] < nx)
            & (src_grid_xyz[:, 1] >= 0) & (src_grid_xyz[:, 1] < ny)
            & (src_grid_xyz[:, 2] >= 0) & (src_grid_xyz[:, 2] < nz)
        )
        _t = _stage(f"bounds check (row_in={row_in.sum()}/{row_in.size}, "
                    f"slot_in={slot_in.sum()}/{slot_in.size})", _t)

        # --- 4a) Setup-stage in-window filter (ALL modes). Drop ONCE, here,
        # every shot row whose receiver grid index is out of bounds AND every
        # row belonging to an out-of-grid source node, then drop nodes left
        # empty. Unified-plan equivalent of fwi_workflow Step-6b
        # ``filter_*_to_rotated_box``: the sampler (random OR receiver-first)
        # then only ever sees in-window shots/nodes, so no out-of-window shot
        # or trace can reach the solver — where the C kernel would silently
        # zero it (record_kernel range-guard), diluting the gradient with
        # zero-synthetic-vs-real-obs residuals. This REPLACES the old per-iter
        # ``row_in_keep``, which was computed, printed, and NEVER applied —
        # out-of-window receivers leaked straight through to the solver.
        if not (row_in.all() and slot_in.all()):
            # Combined row keep: in-window receiver AND its source node in-window.
            row_group = np.repeat(
                np.arange(plan.n_groups, dtype=np.int64),
                np.diff(plan.group_offsets),
            )
            row_keep = row_in & slot_in[row_group]
            n_oob_rows = int((~row_in).sum())
            n_oob_groups = int((~slot_in).sum())
            n_rows0, n_groups0 = int(plan.n_rows), int(plan.n_groups)
            plan = plan.filter_rows(row_keep).drop_empty_groups()
            # Re-derive every plan-indexed geometry array on the filtered plan
            # (group renumbering after drop_empty_groups makes the old arrays
            # stale). Same origin / dh / grid as steps 3-4 above.
            plan_model_xy = frame.to_model(plan.row_source_xyz[:, :2])
            src_model_xy = frame.to_model(plan.group_xyz[:, :2])
            plan_grid_xyz = np.stack([
                np.rint((plan_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64),
                np.rint((plan_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64),
                np.rint((plan.row_source_xyz[:, 2] - origin_z) / dz_m).astype(np.int64),
            ], axis=-1)
            src_grid_xyz = np.stack([
                np.rint((src_model_xy[:, 0] - origin_x) / dx_m).astype(np.int64),
                np.rint((src_model_xy[:, 1] - origin_y) / dy_m).astype(np.int64),
                np.rint((plan.group_xyz[:, 2] - origin_z) / dz_m).astype(np.int64),
            ], axis=-1)
            # Everything left is in-window by construction.
            row_in = np.ones(plan.n_rows, dtype=bool)
            slot_in = np.ones(plan.n_groups, dtype=bool)
            print(
                f"[multisource] setup in-window filter: dropped {n_oob_rows} "
                f"({n_oob_rows / max(n_rows0, 1):.1%}) OOB shot rows + "
                f"{n_oob_groups} OOB nodes -> plan {n_rows0}->{plan.n_rows} rows, "
                f"{n_groups0}->{plan.n_groups} nodes "
                f"(grid nz,ny,nx={nz},{ny},{nx})"
            )
            _t = _stage("setup in-window plan filter (filter_rows+drop_empty)", _t)
        eligible_groups = np.flatnonzero(slot_in).astype(np.int64)

        # --- 4a') Diving-wave window: match the pick database's per-node
        # moveout params onto the nodes ONCE, here — AFTER the in-window
        # filter, whose drop_empty_groups() renumbers groups (matching before
        # it would silently mis-assign every node's window). The window itself
        # is geometry-only and gets built per node inside the misfit loop.
        _diving_asinh_g = None
        if getattr(spec.loss, "diving_window_db", None):
            if not (dz_m == dy_m == dx_m):
                raise ValueError(
                    "loss.diving_window_db needs an isotropic grid: the window's "
                    "offset is |rec - node| * dh with a SCALAR dh, so anisotropic "
                    f"cells would mis-scale it; got dh_xyz_m=({dz_m}, {dy_m}, {dx_m})."
                )
            _diving_asinh_g, _n_hit = _diving_asinh_for_nodes(
                spec.loss.diving_window_db, src_model_xy)
            _n_miss = int(plan.n_groups) - _n_hit
            print(f"[multisource] diving window: matched {_n_hit}/{plan.n_groups} "
                  f"nodes to {Path(spec.loss.diving_window_db).name}")
            if _n_miss:
                print(f"[multisource] WARNING: {_n_miss} node(s) have no pick -> "
                      f"NOT windowed (full record enters the misfit)")
            if _n_hit == 0:
                raise ValueError(
                    f"loss.diving_window_db matched 0/{plan.n_groups} nodes within "
                    f"25 m. Is {spec.loss.diving_window_db} in the same rotated "
                    f"frame as geometry.rotation_metadata?"
                )
            _t = _stage("diving-window node match", _t)

        # --- 4b) QC: dump the receiver layout in BOTH UTM (raw) and model
        # frame (post-rotation, origin-shifted) so the user can confirm the
        # rotation_metadata.json gives the same survey layout as the
        # legacy pipeline. Side effect: writes
        # ``qc/receiver_layout_rotation.png`` + ``qc/receiver_layout.npz``.
        try:
            _dump_receiver_rotation_qc(
                task_dir=task_dir,
                recv_model_xy=src_model_xy,                # (n_groups, 2) rotated
                recv_utm_xy=plan.group_xyz[:, :2],         # (n_groups, 2) raw UTM
                recv_z=plan.group_xyz[:, 2],               # (n_groups,)  z (vertical, no rotation)
                grid_origin_xyz=(origin_x, origin_y, origin_z),
                grid_shape=(nx, ny, nz),
                dh_xyz=(dx_m, dy_m, dz_m),
            )
        except Exception as qerr:  # noqa: BLE001
            print(f"[multisource] receiver layout QC skipped: {qerr}")
        _t = _stage("receiver-layout QC dump", _t)

        # --- 5) Build solver, wavelet, vp tensor (+ optional reparam_net).
        effective_dt = float(spec.time.dt)
        effective_nt = int(spec.time.nt)
        _cfl_check(
            float(init_vp_np.max()), float(min(dx_m, dy_m, dz_m)), effective_dt,
        )
        solver = _build_solver(
            spec.physics, spec.backend, shape, dx_m, effective_dt, effective_nt, dev,
        )
        if dd_on:
            solver = _dd_wrap(solver, dd_mesh)
        _t = _stage(f"_build_solver ({spec.physics.equation}, shape={shape})", _t)
        wavelet_np = _build_wavelet(
            spec.wavelet, spec.time,
            override_dt=effective_dt, override_nt=effective_nt,
        )
        wavelet_t = torch.as_tensor(wavelet_np, dtype=torch.float32, device=dev)
        # SIREN-pipeline wavelet has a zero-prepad (``source_delay_s``)
        # at the start of the wavelet array — main bang sits at sample
        # ``round(source_delay_s / dt)``. SIREN training also pre-padded
        # ``observed`` by the same amount, so to reuse the SIREN wavelet
        # as-is we mirror that frame: shift the prefetched obs forward
        # in time by the same number of samples each iter. Without this,
        # syn and obs are off by ``source_delay_s`` (a systematic
        # cycle-skip once the delay is a sizeable part of a period).
        obs_prepad_s = _get_wavelet_source_delay_s(spec.wavelet)
        obs_prepad_samples = (
            int(round(obs_prepad_s / float(effective_dt)))
            if obs_prepad_s > 0.0 else 0
        )
        if obs_prepad_samples > 0:
            print(
                f"[multisource] wavelet source_delay_s={obs_prepad_s*1000:.1f} ms "
                f"→ obs left-shifted by {obs_prepad_samples} samples "
                f"(@ effective_dt={effective_dt*1000:.2f} ms) each iter to "
                f"align syn/obs in the SIREN frame"
            )
        # Multi-stage frequency continuation. ``_normalise_stage_list``
        # returns the stage list (single-stage fallback when spec.stages is
        # None, using spec.bandpass + spec.epochs). The encoded path
        # bandpasses BOTH the source wavelet and the per-iter obs from the
        # PRISTINE wavelet at each stage entry ("target=wavelet" regime: syn
        # stays naturally bandlimited, no per-iter syn filter). The reparam
        # net / optimizer carry over across stages — one plan load + one
        # solver build for the whole sweep, no per-band resume.
        stage_list = _normalise_stage_list(spec)
        for _si, _st in enumerate(stage_list):
            # Multi-resolution: per-stage ``dh_m`` IS supported (coarse grid for
            # low-freq bands → finer for high). Per-stage ``dt_s``/``nt`` are now
            # ALSO supported: on the stage boundary the wavelet is re-resampled
            # from the pristine copy and the obs prefetch is re-primed at the new
            # time axis (the prefetch worker reads effective_dt/effective_nt live).
            _unsup = [_f for _f in ("wavelet",)
                      if getattr(_st, _f, None) is not None]
            if abs(float(_st.lr_scale) - 1.0) > 1e-12:
                _unsup.append("lr_scale")
            if abs(float(getattr(_st, "inr_lr_scale", 1.0)) - 1.0) > 1e-12:
                _unsup.append("inr_lr_scale")
            if getattr(_st, "batch_size", None) is not None:
                _unsup.append("batch_size")
            if _unsup:
                raise NotImplementedError(
                    f"stage {_si}: the encoded OBN path supports per stage "
                    f"bandpass / epochs / optimizer_reset / dh_m / dt_s / nt; "
                    f"unsupported here: {_unsup}"
                )
        stage_epochs = [int(_st.epochs) for _st in stage_list]
        stage_starts = [int(s) for s in np.cumsum([0] + stage_epochs[:-1])]
        wavelet_orig = wavelet_t.detach().clone()  # pristine, pre-bandpass
        # Pristine wavelet (numpy) + its dt, kept immutable so a per-stage dt
        # change re-resamples from the ORIGINAL (no cumulative drift). Consumed
        # in the stage-entry block below when a stage sets dt_s/nt.
        _wav_pristine_np = wavelet_orig.detach().cpu().numpy().copy()
        _wav_orig_dt = float(effective_dt)
        from sweep_tasks.preproc.resample import resample_time as _resample_time_wav

        def _stage_bandpass(_st):
            return (_st.bandpass if _st.bandpass is not None
                    else getattr(spec, "bandpass", None))
        # bandpass_spec is (re)assigned per stage inside the loop; seed it
        # with stage 0 so the run-meta / wavelet build reflect band 1.
        bandpass_spec = _stage_bandpass(stage_list[0])
        if len(stage_list) > 1:
            _bands = [(b.lo_hz, b.hi_hz) if (b := _stage_bandpass(s)) is not None
                      else None for s in stage_list]
            print(f"[crg] multi-stage FWI: {len(stage_list)} stages, "
                  f"epochs={stage_epochs} (total {sum(stage_epochs)}), "
                  f"bandpass/stage={_bands}")
        # Lateral receiver smoothing (applied to obs AND syn before the loss).
        _recsm = getattr(spec, "receiver_smoothing", None)
        if _recsm is not None and not _recsm.enabled:
            _recsm = None
        if _recsm is not None:
            # The operator mixes receivers across the whole supershot, so it
            # cannot be evaluated tile-locally (DD) or on the ragged per-CRG
            # loop, which compute their misfit on receiver subsets.
            if dd_on:
                raise ValueError(
                    "receiver_smoothing is not supported with domain "
                    "decomposition: each tile owns a receiver subset, so the "
                    "smoothing stencil would be truncated at tile boundaries")
            _tgt = ("obs AND syn" if _recsm.target == "both"
                    else "obs ONLY (syn left unsmoothed)")
            print(f"[crg] receiver smoothing ON: sigma={_recsm.sigma_cells} cells "
                  f"({_recsm.sigma_cells * float(spec.grid.dh):.0f} m), "
                  f"cutoff={_recsm.cutoff_sigmas} sigma — applied to {_tgt}",
                  flush=True)

        vp_leaf = torch.from_numpy(init_vp_np).to(dev).requires_grad_(True)
        inv_by_name = {"vp": vp_leaf}
        reparam_net = None
        # Pre-build the SIREN water mask from a 2-D seabed-depth map
        # (preferred — geological data) before constructing the reparam
        # net. The full-extent npz is sliced with the SAME (y_lo:y_hi,
        # x_lo:x_hi) window that cropped init_vp_np above, so the
        # broadcast 3-D mask lines up voxel-for-voxel with vp_leaf.
        water_mask_override = None
        cropped_sd = None   # captured for per-stage water-mask rebuild (multi-res)
        if (spec.reparam is not None
                and bool(getattr(spec.reparam, "mask_water_layer", False))
                and getattr(spec.reparam, "seabed_depth_path", None) is not None):
            from sweep_tasks.bathymetry import (
                load_seabed_depth_npz, water_mask_from_seabed_depth,
            )
            full_sd = load_seabed_depth_npz(spec.reparam.seabed_depth_path)
            if full_sd.ndim != 2:
                raise ValueError(
                    f"reparam.seabed_depth_path: expected 2-D (ny, nx) "
                    f"seabed_depth for 3-D FWI; got shape {full_sd.shape}"
                )
            if (full_sd.shape[0] != ny_pre or full_sd.shape[1] != nx_pre):
                raise ValueError(
                    f"reparam.seabed_depth_path shape {full_sd.shape} "
                    f"!= init_vp pre-crop (ny, nx) ({ny_pre}, {nx_pre}); "
                    "regenerate the bathymetry from the same init_vp."
                )
            cropped_sd = full_sd[y_lo:y_hi, x_lo:x_hi]
            mask_np = water_mask_from_seabed_depth(
                cropped_sd, nz=int(init_vp_np.shape[0]), dh_z_m=float(dz_m),
            )
            if dd_on and tuple(mask_np.shape) != tuple(shape):
                # match the DD-padded base grid (edge-replicate high x/y)
                mask_np = np.pad(
                    mask_np, ((0, 0),
                              (0, shape[-2] - mask_np.shape[-2]),
                              (0, shape[-1] - mask_np.shape[-1])), mode="edge")
            water_mask_override = torch.from_numpy(mask_np)
            print(
                f"[crg] reparam water_mask from {spec.reparam.seabed_depth_path}: "
                f"shape={tuple(mask_np.shape)}, "
                f"voxels={int(mask_np.sum()):,}/{mask_np.size:,} "
                f"({100*mask_np.sum()/mask_np.size:.2f}%), "
                f"depth range [{cropped_sd.min():.1f}, {cropped_sd.max():.1f}] m"
            )
        if spec.reparam is not None:
            reparam_net = _build_reparam_net(
                spec.reparam, vp_leaf, _effective_bound(spec.model_bounds, "vp"),
                water_mask_override=water_mask_override,
            )
            with torch.no_grad():
                rendered = reparam_net().detach()
            inv_by_name["vp"] = rendered
            n_params = sum(p.numel() for p in reparam_net.parameters())
            print(f"[crg] reparam=velocity_inr  net_params={n_params:,}  "
                  f"lr={spec.reparam.lr:.3e}")

        _t = _stage(f"build wavelet + bandpass + vp_leaf + reparam_net "
                    f"({'SIREN' if reparam_net is not None else 'no reparam'})", _t)

        # ---- Multi-resolution per-stage grid support -----------------------
        # Capture the dh-INDEPENDENT base state so a stage that sets ``dh_m``
        # can re-project geometry + resample vp/water-mask + rebuild the solver
        # at its own spacing (coarse for low-freq bands → fine for high), cutting
        # low-frequency compute by ~(dh_base/dh_stage)^3 cells. dt/nt stay fixed
        # (a dt valid at the finest grid is conservatively valid coarser), so the
        # obs/wavelet time axis never re-samples. Geometry re-projects from the
        # model-frame metres (dh-independent); the C kernel's OOB guard absorbs
        # the handful of edge rows that round just out of a coarser grid.
        _mr_origin = (float(origin_x), float(origin_y), float(origin_z))
        _mr_base_shape = tuple(int(v) for v in shape)
        _mr_base_dh = float(dx_m)
        _mr_pristine_vp = torch.from_numpy(np.ascontiguousarray(init_vp_np)).to(dev)
        _mr_plan_model_xy = plan_model_xy
        _mr_src_model_xy = src_model_xy
        _mr_row_z = np.ascontiguousarray(plan.row_source_xyz[:, 2]).astype(np.float64)
        _mr_grp_z = np.ascontiguousarray(plan.group_xyz[:, 2]).astype(np.float64)
        _mr_cropped_sd = cropped_sd
        # When the water mask is derived INTERNALLY (seabed_depth_path=null →
        # mask = init_vp == water_vp), there is no 2-D seabed_depth to resample
        # per stage; deriving the mask from the trilinear-resampled base would
        # fail the exact-equality test. Recover an effective 2-D seabed depth
        # from the base water mask (water column is contiguous from z=0:
        # seabed_depth[x,y] = n_water_cells * dz_base) so the per-stage
        # ``water_mask_from_seabed_depth`` rebuild works for both cases.
        if (_mr_cropped_sd is None and spec.reparam is not None
                and bool(getattr(spec.reparam, "mask_water_layer", False))):
            _wvp = float(getattr(spec.reparam, "water_vp_m_s", 1500.0))
            _base_wmask = (init_vp_np == _wvp)            # (nz, ny, nx)
            if bool(_base_wmask.any()):
                _mr_cropped_sd = (_base_wmask.sum(axis=0).astype(np.float64)
                                  * float(dz_m))          # (ny, nx) seabed depth (m)
                print(f"[crg] multi-res: derived effective seabed_depth from "
                      f"base water mask (init_vp=={_wvp:.0f}) for per-stage mask "
                      f"rebuild ({int((_mr_cropped_sd > 0).sum())}/"
                      f"{_mr_cropped_sd.size} wet columns)")
        _cur_dh = float(dx_m)

        def _grid_for_dh(dh):
            ox, oy, oz = _mr_origin
            nz_b, ny_b, nx_b = _mr_base_shape
            r = _mr_base_dh / float(dh)
            sh = (max(1, int(round(nz_b * r))),
                  max(1, int(round(ny_b * r))),
                  max(1, int(round(nx_b * r))))
            pg = np.stack([
                np.rint((_mr_plan_model_xy[:, 0] - ox) / dh).astype(np.int64),
                np.rint((_mr_plan_model_xy[:, 1] - oy) / dh).astype(np.int64),
                np.rint((_mr_row_z - oz) / dh).astype(np.int64),
            ], axis=-1)
            sg = np.stack([
                np.rint((_mr_src_model_xy[:, 0] - ox) / dh).astype(np.int64),
                np.rint((_mr_src_model_xy[:, 1] - oy) / dh).astype(np.int64),
                np.rint((_mr_grp_z - oz) / dh).astype(np.int64),
            ], axis=-1)
            return sh, pg, sg

        def _apply_stage_dh(dh, si):
            """Switch the run to grid spacing ``dh`` (m) for the current stage."""
            nonlocal shape, plan_grid_xyz, src_grid_xyz, solver, vp_leaf, _cur_dh
            sh, pg, sg = _grid_for_dh(dh)
            plan_grid_xyz, src_grid_xyz = pg, sg
            nz_s, ny_s, nx_s = sh
            new_base = _resample_vp_tensor(_mr_pristine_vp, sh).detach().to(dev)
            new_mask = None
            if _mr_cropped_sd is not None:
                import torch.nn.functional as _F
                from sweep_tasks.bathymetry import water_mask_from_seabed_depth
                sd_t = torch.from_numpy(
                    np.ascontiguousarray(_mr_cropped_sd))[None, None].float()
                sd_s = _F.interpolate(
                    sd_t, size=(ny_s, nx_s), mode="bilinear",
                    align_corners=False)[0, 0].numpy()
                new_mask = torch.from_numpy(
                    water_mask_from_seabed_depth(sd_s, nz=nz_s, dh_z_m=float(dh))
                ).to(dev)
            if dd_on:
                # DD v1 needs uniform tiles.  The base pad (at grid.dh) does NOT
                # survive a per-stage dh_m resample (e.g. 25 m -> 18.75 m can
                # leave ny % py != 0), so edge-replicate the HIGH y/x edge of the
                # RESAMPLED stage grid here so ny%py==0 and nx%px==0.  Pad is past
                # the survey edge -> origin + every src/rec coord stay valid.
                import torch.nn.functional as _F
                _yp, _xp = (-ny_s) % dd_py, (-nx_s) % dd_px
                if _yp or _xp:
                    # 3-D (nz, ny, nx) is (C, H, W) for F.pad -> size-4 replicate
                    # pads the last two dims (ny, nx), leaving nz untouched.
                    new_base = _F.pad(new_base, (0, _xp, 0, _yp), mode="replicate")
                    if new_mask is not None:
                        new_mask = _F.pad(
                            new_mask.float(), (0, _xp, 0, _yp),
                            mode="replicate").to(new_mask.dtype)
                    print(f"[dd] stage padded grid to uniform tiles: "
                          f"({nz_s},{ny_s},{nx_s}) -> "
                          f"({nz_s},{ny_s + _yp},{nx_s + _xp}) "
                          f"(px={dd_px} py={dd_py})", flush=True)
                    sh = (nz_s, ny_s + _yp, nx_s + _xp)
                    nz_s, ny_s, nx_s = sh
            shape = sh
            if reparam_net is not None:
                # update_base_velocity resamples coords + (optionally) the water
                # mask to the new shape; per-iter ``reparam_net()`` then renders
                # at the new grid, so no inv_by_name swap is needed here.
                reparam_net.update_base_velocity(new_base, water_mask=new_mask)
            else:
                vp_leaf = new_base.requires_grad_(True)
                inv_by_name["vp"] = vp_leaf
            # Free the PREVIOUS stage's solver GPU buffers before building the new
            # one. Each DD stage holds tile boundary rings + wavefields + record
            # buffers; if the old ModelParallel/PropTorch isn't dropped first they
            # accumulate across stages (a fine stage OOM'd while still carrying
            # the coarser stages' residue, though it alone needs far less).
            solver = None
            import gc as _gc
            _gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            solver = _build_solver(
                spec.physics, spec.backend, sh, float(dh),
                effective_dt, effective_nt, dev,
            )
            if dd_on:
                solver = _dd_wrap(solver, dd_mesh)
            _cur_dh = float(dh)
            print(f"[crg] STAGE {si + 1} grid -> dh={dh:.1f} m shape={sh} "
                  f"(base {_mr_base_dh:.1f} m / {_mr_base_shape})", flush=True)

        # --- 6) Optimizer.
        if reparam_net is not None:
            optimizer = _build_reparam_optimizer(
                spec.optimizer, reparam_net.parameters(), float(spec.reparam.lr),
            )
        else:
            optimizer = _build_optimizer(spec.optimizer, inv_by_name, ["vp"])
        _t = _stage("optimizer build", _t)

        # --- 7) Plan reader + RNG for the per-iter sampler.
        # PlanReader serves SEG-Y trace bytes through MultiFileSEGYReader
        # under the hood. cache_all=False keeps memory bounded (about a GB
        # even on a large survey) at the cost of decoding bytes per call;
        # cache_all=True is fine for ≤ ~50 GB plans that fit in RAM.
        # Trace cache + coalesce gap: legacy OBN-supershot IO optimisations
        # that absorb cold-Lustre re-reads (hot traces in RAM) and merge
        # adjacent intra-file ``pread``s into one big read. Without these
        # wait_io dominated every iteration on a large survey; with them it is a fraction.
        _trace_stride_bytes = int(
            plan.trace_size_per_file[0]
        ) if plan.trace_size_per_file.size else (
            240 + int(plan.samples_per_trace) * 4
        )
        # 4× trace stride: coalesces any two traces within 4 strides of
        # each other in the same file (typical for shared-shot sampling
        # where the CDF of intra-file gaps clusters near the stride).
        _coalesce_gap = max(0, 4 * _trace_stride_bytes)
        reader = PlanReader(
            # On-demand (cache_all=False) random reads on wekafs/Lustre: use
            # os.pread (mmap=False) so each read releases the GIL and the
            # DataLoader workers / DDP ranks read in parallel instead of
            # GIL-serialising on mmap page faults — the dominant wait_io cost
            # once compute is small (multi-GPU DDP, few shots/rank). Eager
            # cache_all=True does one bulk sequential read where mmap is fine.
            plan, mmap=bool(spec.obs.plan.cache_all),
            cache_all=bool(spec.obs.plan.cache_all),
            trace_cache_bytes=int(getattr(sampling_cfg, "trace_cache_bytes", 0)),
            coalesce_gap=_coalesce_gap,
        )
        if bool(spec.obs.plan.cache_all):
            print(f"[multisource] PlanReader: cache_all=True "
                  f"(eagerly loaded {plan.n_rows} traces into RAM)")
        else:
            _tcb = int(getattr(sampling_cfg, "trace_cache_bytes", 0))
            if _tcb < 0:
                cache_desc = "per-trace LRU, unbounded"
            elif _tcb > 0:
                cache_desc = f"per-trace LRU, budget {_tcb / 1e9:.2f} GB"
            else:
                cache_desc = "DISABLED — wait_io may dominate on cold-Lustre"
            print(f"[multisource] PlanReader: trace_cache={cache_desc}, "
                  f"coalesce_gap={_coalesce_gap} bytes "
                  f"(= 4× stride {_trace_stride_bytes})")
        # ``sign_rng`` is only used by the encoded supershot path. In the
        # per-shot path it stays None and the forward loop simply skips
        # the sign sampling.
        if encoding_on:
            sign_rng = np.random.default_rng(encoding_spec.sign_seed)
        else:
            sign_rng = None
        sample_rng = np.random.default_rng(spec.seed)
        _t = _stage(f"PlanReader init (cache_all={bool(spec.obs.plan.cache_all)})", _t)

        # Pre-compute sorted-unique (sx, sy) keys for every CRG group once
        # at setup. The shared-shot sampler then skips ``np.unique`` per
        # group per iter (saves O(B × N log N) Python work × 1000 iters).
        # On a large survey: seconds ONCE here save seconds per iter in the sampler.
        group_unique_keys = precompute_group_unique_keys(plan)
        _t = _stage(
            f"precompute_group_unique_keys ({plan.n_groups} groups)", _t,
        )

        # Receiver-first reverse index (shot_key -> covering nodes), built
        # ONCE when sampling.receiver_first is on. It lets the per-iter
        # target shot sweep the whole survey (pick a target, then the nodes
        # that recorded it) instead of the random-group intersection
        # collapsing toward the survey centre on partial-coverage OBN data.
        shotkey_to_nodes = None
        shotkey_keys_arr = None
        if bool(getattr(sampling_cfg, "receiver_first", False)):
            # min_cov (= max(sampling, source_encoding) coverage) and
            # eligible_groups (= the in-grid slot_in nodes) must match the
            # random-sampler path exactly, else receiver-first surfaces nodes
            # the encoded path would reject (out-of-grid source / below the
            # effective coverage threshold).
            shotkey_to_nodes = build_shotkey_to_nodes_index(
                plan, min_coverage=int(min_cov),
                eligible_groups=eligible_groups,
                verbose=bool(dist_info.is_root),
            )
            shotkey_keys_arr = np.fromiter(
                shotkey_to_nodes.keys(), dtype=np.int64,
                count=len(shotkey_to_nodes),
            )
            _t = _stage(
                f"build_shotkey_to_nodes_index "
                f"({len(shotkey_to_nodes)} shot keys)", _t,
            )

        # Survey-wide sub-sample of every physical shot's (sx, sy) — used
        # as the faint background in the per-iter QC map. The full
        # set kills matplotlib; ~20k random hits give a clear footprint.
        # NOTE: the QC map is drawn in MODEL frame (post-rotation,
        # origin-shifted), not UTM — that's the frame the inversion
        # actually operates in. We capture the bg-row indices now and
        # do the rotation lookup once via the already-computed
        # plan_model_xy / src_model_xy arrays (no extra rotation pass).
        _SURVEY_MAP_MAX_SHOTS = 20_000
        if int(plan.n_rows) > _SURVEY_MAP_MAX_SHOTS:
            _rng_map = np.random.default_rng(0)
            _bg_idx = _rng_map.choice(
                int(plan.n_rows), size=_SURVEY_MAP_MAX_SHOTS, replace=False,
            )
        else:
            _bg_idx = np.arange(int(plan.n_rows), dtype=np.int64)
        all_shots_model_bg = plan_model_xy[_bg_idx].astype(np.float64)
        all_groups_model_bg = src_model_xy.astype(np.float64)
        # Inversion-active extent (model-frame meters) — what the solver
        # actually sees. ymin/xmin = grid origin (post-crop, post-pad);
        # max = origin + n*dh. Drawn as a dashed rectangle on the QC map
        # so the user immediately sees which picked OBN / used shots
        # land INSIDE the active inversion window.
        inversion_extent_xy_m = (
            float(origin_x), float(origin_x + nx * dx_m),
            float(origin_y), float(origin_y + ny * dy_m),
        )
        _t = _stage(
            f"survey-map background sample ({all_shots_model_bg.shape[0]} "
            f"shots / {all_groups_model_bg.shape[0]} OBN nodes; model frame; "
            f"grid {nx}×{ny} cells "
            f"= {nx*dx_m/1000:.1f}×{ny*dy_m/1000:.1f} km)", _t,
        )

        # Well-log QC positions: 6 pseudo-wells on a 3×2 (x × y) grid
        # spaced across the cropped inversion extent. (x_km, y_km) are
        # in MODEL frame meters; converted to grid (iy, ix) via the
        # current origin / dh. Wells outside the grid render as
        # "OUT OF GRID" stubs (the helper handles it cleanly).
        _well_x_m = np.linspace(
            float(origin_x) + 0.15 * nx * dx_m,
            float(origin_x) + 0.85 * nx * dx_m, 3,
        )
        _well_y_m = np.linspace(
            float(origin_y) + 0.25 * ny * dy_m,
            float(origin_y) + 0.75 * ny * dy_m, 2,
        )
        well_xy_m = np.array(
            [(x, y) for y in _well_y_m for x in _well_x_m],
            dtype=np.float64,
        )
        well_grid_idx = np.stack([
            np.rint((well_xy_m[:, 1] - origin_y) / dy_m).astype(np.int64),
            np.rint((well_xy_m[:, 0] - origin_x) / dx_m).astype(np.int64),
        ], axis=-1)
        well_labels = [
            f"x={xy[0]/1000:.1f} y={xy[1]/1000:.1f} km"
            for xy in well_xy_m
        ]
        _t = _stage(
            f"well-log QC wells ({len(well_labels)} pseudo-wells: "
            f"{', '.join(well_labels)})", _t,
        )

        # Cross-iter SEG-Y prefetcher. Each iter's batch sampling + SEG-Y
        # read runs in a background thread so the GPU can compute on iter
        # N while we fetch iter N+1's traces. Without this, random batches
        # past the OS pagecache working set blow up wait_io from ~0.5s to
        # tens of seconds per iter (a survey larger than the available RAM).
        #
        # Inside the loader we also parallelise the per-slot SEG-Y reads
        # across an inner thread pool (``num_workers`` from the config) —
        # the SEG-Y reader releases the GIL during mmap reads so threads
        # see real concurrency on NVMe storage.
        from concurrent.futures import ThreadPoolExecutor

        n_io_workers = max(1, int(sampling_cfg.num_workers))
        # Outer pool: cross-iter pre-load (1 worker is enough — we only
        # need one prefetched batch at a time).
        prefetch_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sharedshot-prefetch",
        )
        # Inner pool: per-group SEG-Y reads inside one iter's load.
        io_pool = (
            ThreadPoolExecutor(max_workers=n_io_workers,
                                thread_name_prefix="sharedshot-io")
            if n_io_workers > 1 else None
        )

        # Per-iter timing accumulator for the prefetch worker so we can
        # break down wait_io into (sample / read / resample / trim) — the
        # main loop's wait_io only tells us the OUTER prefetcher cost.
        # Each entry is one iter's dict; reset by main loop after read.
        prefetch_timings: list[dict] = []

        def _load_iter_payload(rng_state):
            """Sample shared-shots + read all per-group traces. Returns
            ``(batch, traces_per_group)`` where ``traces_per_group`` is a
            ``(B, n_shared, nt_segy)`` ``float32`` numpy array.

            ``rng_state`` seeds the sampler for determinism — the caller
            (the main loop) advances the master ``sample_rng`` state so
            re-runs with the same seed produce the same iter sequence.
            """
            tstats = {"sample": 0.0, "alloc": 0.0, "read": 0.0,
                      "resample": 0.0, "trim": 0.0}
            t_sample0 = time.perf_counter()
            local_rng = np.random.default_rng(rng_state)
            # Deterministic group ENUMERATION (SWEEP_ENUM_GROUPS=1): treat
            # ``rng_state`` as the group ordinal and process exactly that
            # ONE CRG (B=1) with its OWN shots. Iterating ordinals 0..N-1
            # over N=epochs dumps the init-model sum/mean of every CRG's
            # gradient — a true full-survey gradient, vs the random
            # shared-shot supershot draws.
            if os.environ.get("SWEEP_ENUM_GROUPS") == "1":
                _elig_all = (eligible_groups if eligible_groups is not None
                             else np.arange(int(plan.n_groups), dtype=np.int64))
                _gi = int(rng_state) % int(_elig_all.size)
                _elig_iter = _elig_all[_gi:_gi + 1]
                _bs_iter = 1
            else:
                _elig_iter = eligible_groups
                _bs_iter = int(spec.batchsize)
            _percrg_iter = bool(per_crg_independent and _bs_iter > 1)
            if _percrg_iter:
                # Per-CRG independent: B random nodes, each keeps its OWN
                # sub-sampled rows (no shared-shot intersection). Ragged.
                b = sample_percrg_independent(
                    plan, local_rng,
                    batch_size=_bs_iter,
                    source_lines_per_group=int(sampling_cfg.source_lines_per_group),
                    max_traces_per_sourceline=int(sampling_cfg.max_traces_per_sourceline),
                    min_coverage=int(min_cov),
                    eligible_groups=_elig_iter,
                )
            elif shotkey_to_nodes is not None and _bs_iter > 1:
                # Receiver-first: pick a target shot then the nodes that
                # recorded it, so the per-iter supershot sweeps the whole
                # survey. Falls back internally to the random sampler if no
                # target reaches batch_size within the retry budget.
                b = sample_shared_shots_receiver_first(
                    plan, local_rng,
                    batch_size=_bs_iter,
                    source_lines_per_group=int(sampling_cfg.source_lines_per_group),
                    max_traces_per_sourceline=int(sampling_cfg.max_traces_per_sourceline),
                    min_coverage=int(min_cov),
                    max_retries=int(sampling_cfg.receiver_first_max_retries),
                    eligible_groups=eligible_groups,
                    shotkey_to_nodes=shotkey_to_nodes,
                    shotkey_keys_arr=shotkey_keys_arr,
                    precomputed_group_unique_keys=group_unique_keys,
                )
            else:
                b = sample_shared_shots_from_plan(
                    plan, local_rng,
                    batch_size=_bs_iter,
                    source_lines_per_group=int(sampling_cfg.source_lines_per_group),
                    max_traces_per_sourceline=int(sampling_cfg.max_traces_per_sourceline),
                    # min_coverage + eligible_groups applied per-iter at sample
                    # time (instead of pre-filtering the plan via filter_groups
                    # / filter_rows) — see _run_fwi_multisource setup.
                    min_coverage=int(min_cov),
                    eligible_groups=_elig_iter,
                    precomputed_group_unique_keys=group_unique_keys,
                )
            tstats["sample"] = time.perf_counter() - t_sample0
            t_alloc0 = time.perf_counter()
            B = int(b.group_indices.size)

            if _percrg_iter:
                # Ragged read → per-node grid-cell dedup → zero-pad to a dense
                # (B, max_nrec, nt) batch. Each node keeps its own receiver
                # geometry (``recv_rows_padded``); the padded tail is flagged
                # in ``valid_mask`` and masked out of the loss. Padding the
                # receiver rows with row 0 keeps every padded receiver on a
                # valid grid cell (its residual is zeroed by the mask anyway).
                _obs_list: list = [None] * B
                _rows_list: list = [None] * B

                def _read_dedup(i):
                    # Dedup rows to unique receiver cells BEFORE reading, so
                    # full-coverage nodes (many raw rows, several shots per cell)
                    # only pay the SEG-Y read for the ~unique-cell subset.
                    rows_i = b.rows_per_group[i]
                    cells_i = plan_grid_xyz[rows_i]               # (nrec_i, 3)
                    _, keep = np.unique(cells_i, axis=0, return_index=True)
                    keep = np.sort(keep)
                    rows_kept = rows_i[keep]
                    _rows_list[i] = rows_kept
                    _obs_list[i] = reader.read_rows(rows_kept)    # read deduped only

                tstats["alloc"] = time.perf_counter() - t_alloc0
                t_read0 = time.perf_counter()
                if io_pool is None:
                    for i in range(B):
                        _read_dedup(i)
                else:
                    list(io_pool.map(_read_dedup, range(B)))
                _counts = np.array([r.size for r in _rows_list], dtype=np.int64)
                n_shared = int(_counts.max())
                if dist_info.is_root:
                    print(f"[per-crg] nrec (distinct cells): min={int(_counts.min())} "
                          f"max={n_shared} median={int(np.median(_counts))} "
                          f"total={int(_counts.sum())} | padded obs "
                          f"{24 * n_shared * plan.samples_per_trace * 4 / 2**30:.1f} GB",
                          flush=True)
                t_np = np.zeros(
                    (B, n_shared, plan.samples_per_trace), dtype=np.float32,
                )
                _vmask = np.zeros((B, n_shared), dtype=bool)
                _rrows = np.zeros((B, n_shared), dtype=np.int64)
                for i in range(B):
                    ni = int(_counts[i])
                    t_np[i, :ni] = _obs_list[i]
                    _vmask[i, :ni] = True
                    _rrows[i, :ni] = _rows_list[i]
                    if ni < n_shared:
                        _rrows[i, ni:] = _rows_list[i][0]
                b.rows_per_group = _rows_list      # dedup'd (ragged)
                b.valid_mask = _vmask
                b.recv_rows_padded = _rrows
                b.n_shared = n_shared
                tstats["read"] = time.perf_counter() - t_read0
            else:
                n_shared = int(b.n_shared)
                t_np = np.empty(
                    (B, n_shared, plan.samples_per_trace), dtype=np.float32,
                )
                tstats["alloc"] = time.perf_counter() - t_alloc0

                def _read_one(i):
                    # b.rows_per_group[i] holds ABSOLUTE plan-row indices —
                    # PlanReader.read_rows takes those directly.
                    t_np[i] = reader.read_rows(b.rows_per_group[i])

                t_read0 = time.perf_counter()
                if io_pool is None:
                    for i in range(B):
                        _read_one(i)
                else:
                    list(io_pool.map(_read_one, range(B)))
                tstats["read"] = time.perf_counter() - t_read0

            t_resample0 = time.perf_counter()
            # Resample obs time axis on the prefetcher thread too so the
            # main loop's wait_io covers resample as well.
            if abs(plan.dt_s - effective_dt) > 1.0e-12:
                from sweep_tasks.preproc.resample import resample_time
                t_np = resample_time(t_np, plan.dt_s, effective_dt, axis=-1)
            tstats["resample"] = time.perf_counter() - t_resample0

            t_trim0 = time.perf_counter()
            # Trim / zero-pad to match effective_nt — unconditionally,
            # since SEG-Y trace length can differ from spec.time.nt even
            # when dt matches (field SEG-Y often stores a longer record than the
            # YAML asks for). Previously this trim was nested
            # inside the dt-mismatch branch and missed nt-only mismatches.
            cur_nt = t_np.shape[-1]
            if cur_nt > effective_nt:
                t_np = t_np[..., :effective_nt]
            elif cur_nt < effective_nt:
                pad = [(0, 0)] * t_np.ndim
                pad[-1] = (0, effective_nt - cur_nt)
                t_np = np.pad(t_np, pad)
            t_np = t_np.astype(np.float32, copy=False)
            tstats["trim"] = time.perf_counter() - t_trim0
            tstats["B"] = B
            tstats["n_shared"] = n_shared
            prefetch_timings.append(tstats)
            return b, t_np

        def _next_rng_state():
            """Draw + return a fresh seed for the next iter's sampler,
            keeping ``sample_rng`` as the single source of randomness."""
            return int(sample_rng.integers(0, 2**31 - 1))

        def _iter_seed(iter_idx):
            """Sampler arg for iter ``iter_idx``: the group ordinal in
            enumerate mode (SWEEP_ENUM_GROUPS=1), else a fresh random seed."""
            if os.environ.get("SWEEP_ENUM_GROUPS") == "1":
                return int(iter_idx)
            return _next_rng_state()

        # --- 7b) Optional priors: TVPrior + SeabedFreezeMask.
        tv_prior = None
        smooth_spec = getattr(spec, "smooth_regularization", None)
        if smooth_spec is not None and float(smooth_spec.weight) > 0.0:
            from sweep_nn import TVPrior

            tv_prior = TVPrior(
                order=smooth_spec.order,
                x_weight=float(smooth_spec.x_weight),
                y_weight=float(smooth_spec.y_weight),
                z_weight=float(smooth_spec.z_weight),
                velocity_scale_m_s=float(smooth_spec.velocity_scale_m_s),
            )
        seabed_mask = None
        freeze_spec = getattr(spec, "freeze_water_layer", None)
        if freeze_spec is not None and freeze_spec.enabled:
            from sweep_nn import SeabedFreezeMask

            sb = np.load(freeze_spec.seabed_depth_path)
            if isinstance(sb, np.lib.npyio.NpzFile):
                sb = sb["seabed_depth"]
            seabed_mask = SeabedFreezeMask(
                np.asarray(sb), dz_m=dz_m,
                buffer_cells=int(freeze_spec.buffer_cells),
            )

        # --- 8) Outputs / QC dirs (rank-0 / single-rank).
        out_dir = task_dir / "output"
        out_dir.mkdir(exist_ok=True)
        snapshots_dir = out_dir / "epochs"
        snapshots_dir.mkdir(exist_ok=True)
        qc_dir = task_dir / "qc"
        qc_enabled = spec.qc is not None and spec.qc.every_n_epochs > 0

        # --- 9) Per-iter loop (global epoch counter across all stages).
        total_epochs = int(sum(stage_epochs))
        losses: list[float] = []
        # Optional per-phase profile (SWEEP_TASKS_MSPROF=1): cuda-synced timing
        # of the non-solver per-iter work — dedup / obs-encode+bandpass /
        # reparam render — to locate the GPU-idle gap seen in util traces.
        # Adds syncs, so only when explicitly enabled.
        _msprof = os.environ.get("SWEEP_TASKS_MSPROF") == "1"

        def _msync():
            if _msprof and dev.type == "cuda":
                torch.cuda.synchronize(dev)
        illum_spec = getattr(spec, "illumination_precondition", None)
        illum_on = bool(illum_spec is not None and illum_spec.enabled)
        if illum_on and getattr(spec.backend, "impl", "eager") == "eager":
            print("[illum] WARN: eager backend does not populate solver "
                  "illumination buffers; precondition is a no-op.")
        # perf/acoustic-bwd-skip-illum (3d6fe97): c-backend illumination is now
        # opt-in (default off). Re-enable on the solver when precond is active so
        # the backend populates source/receiver_illumination. No-op on old cores.
        if illum_on and getattr(spec.backend, "impl", "eager") == "c":
            try:
                solver.compute_illumination = True
            except Exception:
                pass
        loss_fn = lambda syn, obs: _compute_loss(syn, obs, spec.loss)
        def _csync():
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        _t = _stage("priors + output dirs + loss_fn ready (setup DONE)", _t)

        # Dump the resolved config + comprehensive runtime metadata at
        # task_dir root. Captures everything a future reader needs to
        # interpret this run: full pydantic-expanded YAML, host / GPU
        # / package versions / git SHAs, plus derived setup quantities
        # (shape, origin, n_groups, water_mask voxel count, obs prepad
        # samples, sampler config, ...). Best-effort — wrapped so any
        # I/O hiccup never blocks the run.
        try:
            _runtime_extras = {
                "device": str(dev),
                "epochs": int(total_epochs),
                "batchsize": int(spec.batchsize),
                "effective_dt_s": float(effective_dt),
                "effective_nt": int(effective_nt),
                "grid_shape_zyx": list(shape),
                "dh_xyz_m": [float(dz_m), float(dy_m), float(dx_m)],
                "origin_xyz_m": [
                    float(origin_z), float(origin_y), float(origin_x),
                ],
                "crop_origin_offset_m": list(crop_origin_offset),
                "init_vp_min_mean_max": [
                    float(init_vp_np.min()),
                    float(init_vp_np.mean()),
                    float(init_vp_np.max()),
                ],
                "plan_n_rows": int(plan.n_rows),
                "plan_n_groups": int(plan.n_groups),
                "plan_samples_per_trace": int(plan.samples_per_trace),
                "eligible_groups": int(eligible_groups.size),
                "min_coverage": int(min_cov),
                "n_picked_per_iter_B": int(spec.batchsize),
                "obs_prepad_s": float(obs_prepad_s),
                "obs_prepad_samples": int(obs_prepad_samples),
                "wavelet_kind": getattr(spec.wavelet, "kind", None),
                "loss_kind": getattr(spec.loss, "kind", None),
                "reparam_kind": (
                    getattr(spec.reparam, "kind", None)
                    if spec.reparam is not None else None
                ),
                "reparam_net_params": (
                    int(sum(p.numel() for p in reparam_net.parameters()))
                    if reparam_net is not None else None
                ),
                "reparam_lr": (
                    float(spec.reparam.lr) if spec.reparam is not None else None
                ),
                "reparam_vp_std": (
                    float(spec.reparam.vp_std) if spec.reparam is not None else None
                ),
                "reparam_vp_mean": (
                    float(spec.reparam.vp_mean) if spec.reparam is not None else None
                ),
                "reparam_mask_water_layer": (
                    bool(getattr(spec.reparam, "mask_water_layer", False))
                    if spec.reparam is not None else False
                ),
                "reparam_water_vp_m_s": (
                    float(getattr(spec.reparam, "water_vp_m_s", 1500.0))
                    if spec.reparam is not None else None
                ),
                "reparam_seabed_depth_path": (
                    str(getattr(spec.reparam, "seabed_depth_path", None))
                    if (spec.reparam is not None
                        and getattr(spec.reparam, "seabed_depth_path", None)
                        is not None)
                    else None
                ),
                "water_mask_voxels_true": (
                    int(water_mask_override.sum().item())
                    if water_mask_override is not None else None
                ),
                "water_mask_voxels_total": (
                    int(water_mask_override.numel())
                    if water_mask_override is not None else None
                ),
                "bandpass_hz": (
                    [float(bandpass_spec.lo_hz),
                     float(bandpass_spec.hi_hz),
                     int(bandpass_spec.order)]
                    if bandpass_spec is not None else None
                ),
                "sampling": {
                    "shared_shots_per_iter": int(sampling_cfg.shared_shots_per_iter),
                    "source_lines_per_group": int(sampling_cfg.source_lines_per_group),
                    "max_traces_per_sourceline": int(sampling_cfg.max_traces_per_sourceline),
                    "min_coverage": int(sampling_cfg.min_coverage),
                    "num_workers": int(sampling_cfg.num_workers),
                    "prefetch_factor": int(sampling_cfg.prefetch_factor),
                    "trace_cache_bytes": int(
                        getattr(sampling_cfg, "trace_cache_bytes", 0)
                    ),
                    "dedup_mode": str(getattr(sampling_cfg, "dedup_mode", "none")),
                },
                "qc_every_n_epochs": (
                    int(spec.qc.every_n_epochs) if spec.qc is not None else None
                ),
                "show_every": int(spec.show_every),
                "setup_wall_s": float(time.perf_counter() - _t_setup0),
            }
            _dump_run_metadata(spec, task_dir, extras=_runtime_extras)
            print(f"[run-meta] wrote {task_dir/'config_resolved.yaml'} "
                  f"+ {task_dir/'run_meta.json'}")
        except Exception as meta_err:  # noqa: BLE001
            print(f"[run-meta] dump skipped: {meta_err}")

        # Prime the prefetcher with iter 0's load so the first iter's
        # ``wait_io`` is also overlapped (with setup work above, ideally).
        next_future = prefetch_pool.submit(_load_iter_payload, _iter_seed(0))
        for epoch in range(total_epochs):
            # Stage entry: re-bandpass the PRISTINE wavelet + switch the obs
            # bandpass spec when crossing into a new frequency-continuation
            # stage. Net/optimizer state carries over (reset only on request).
            if epoch in stage_starts:
                _si = stage_starts.index(epoch)
                _st = stage_list[_si]
                bandpass_spec = _stage_bandpass(_st)
                # --- per-stage dt/nt: update the time axis BEFORE the wavelet
                #     bandpass + solver rebuild. The prefetch worker reads
                #     effective_dt/effective_nt live, so re-priming the in-flight
                #     future (below) is all the obs side needs. ---
                _stage_dt = getattr(_st, "dt_s", None)
                _stage_nt = getattr(_st, "nt", None)
                _time_changed = False
                if _stage_dt is not None and abs(float(_stage_dt) - effective_dt) > 1e-12:
                    effective_dt = float(_stage_dt); _time_changed = True
                if _stage_nt is not None and int(_stage_nt) != effective_nt:
                    effective_nt = int(_stage_nt); _time_changed = True
                if _time_changed:
                    # re-resample the PRISTINE wavelet to the new dt (the bandpass
                    # just below consumes wavelet_orig) + recompute the obs
                    # source-delay prepad at the new dt.
                    _wav_np = _resample_time_wav(_wav_pristine_np, _wav_orig_dt,
                                                 effective_dt, axis=-1)
                    # pad/trim the resampled wavelet to effective_nt so the solver
                    # runs exactly effective_nt steps → syn length == obs length
                    # (obs is trimmed to effective_nt in the prefetch worker;
                    # resample rounding else leaves them off by ~1 sample,
                    # because effective_nt comes from the record duration while
                    # the resampler rounds nt_old * dt_old / dt_new).
                    _cur = _wav_np.shape[-1]
                    if _cur > effective_nt:
                        _wav_np = _wav_np[..., :effective_nt]
                    elif _cur < effective_nt:
                        _wav_np = np.pad(
                            _wav_np,
                            [(0, 0)] * (_wav_np.ndim - 1) + [(0, effective_nt - _cur)])
                    wavelet_orig = torch.as_tensor(
                        _wav_np, dtype=torch.float32, device=dev)
                    obs_prepad_samples = (
                        int(round(obs_prepad_s / float(effective_dt)))
                        if obs_prepad_s > 0.0 else 0)
                    print(f"[crg] stage {_si + 1}: dt -> {effective_dt * 1000:.3f} ms "
                          f"nt -> {effective_nt} "
                          f"(record {effective_dt * effective_nt:.2f} s)", flush=True)
                wavelet_t = (_bandpass_torch_fft(
                    wavelet_orig, lo=float(bandpass_spec.lo_hz),
                    hi=float(bandpass_spec.hi_hz), dt=float(effective_dt),
                    order=int(bandpass_spec.order), axis=-1).detach()
                    if bandpass_spec is not None else wavelet_orig)
                if bool(getattr(_st, "optimizer_reset", False)) and _si > 0:
                    optimizer.state.clear()
                _bp_txt = (f"{bandpass_spec.lo_hz}-{bandpass_spec.hi_hz}Hz"
                           if bandpass_spec is not None else "none")
                print(f"[crg] === STAGE {_si + 1}/{len(stage_list)} "
                      f"epoch[{epoch}:{epoch + int(_st.epochs)}] "
                      f"bandpass={_bp_txt} ===", flush=True)
                # Multi-resolution: switch grid spacing if this stage sets dh_m
                # (geometry re-projection + vp/water-mask resample + solver
                # rebuild). A per-stage dt/nt change ALSO needs the solver rebuilt,
                # so trigger on either. Reparam net params + Adam state carry over.
                _stage_dh = getattr(_st, "dh_m", None)
                _dh_changed = (_stage_dh is not None
                               and abs(float(_stage_dh) - _cur_dh) > 1e-9)
                if _dh_changed or _time_changed:
                    _apply_stage_dh(
                        float(_stage_dh) if _stage_dh is not None else _cur_dh, _si)
                if _time_changed and epoch > 0:
                    # the in-flight prefetch for THIS iter was loaded at the OLD
                    # dt/nt; redo it at the new time axis (same seed → same shots).
                    next_future = prefetch_pool.submit(
                        _load_iter_payload, _iter_seed(epoch))
            # Coarse-to-fine hash: advance the encoder level mask by whole-run
            # progress each epoch (base_levels -> final_levels over [warmup,
            # ramp_end], epoch fraction). The multisource path NEVER drove this, so
            # the hash was frozen at base_levels (=2) the whole run and the fine
            # levels never activated. Mirrors the _run_fwi / freqsel c2f driving.
            _c2f_cfg = getattr(getattr(getattr(spec, "reparam", None),
                                       "hash", None), "c2f", None)
            if (reparam_net is not None and _c2f_cfg is not None
                    and getattr(_c2f_cfg, "enabled", False)):
                if _has_hash_schedule(reparam_net):
                    _act = _advance_hash_schedule(
                        reparam_net, epoch / max(1, total_epochs - 1),
                        _c2f_cfg, optimizer)
                    if _act is not None and dist_info.is_root and (
                            epoch < 3 or epoch in stage_starts or epoch % 20 == 0):
                        print(f"[crg] c2f epoch {epoch}: active levels "
                              f"{_act[0]:.2f}/{_act[1]}", flush=True)
            t_iter = time.perf_counter()
            # Wait for the prefetched iter's payload (already in flight).
            t = time.perf_counter()
            batch, traces_per_group = next_future.result()
            t_wait = time.perf_counter() - t
            # Kick off iter N+1's load BEFORE doing iter N's GPU work so
            # the I/O overlaps with the solver fwd/bwd.
            if epoch + 1 < total_epochs:
                next_future = prefetch_pool.submit(
                    _load_iter_payload, _iter_seed(epoch + 1),
                )
            B = int(batch.group_indices.size)
            n_shared = int(batch.n_shared)
            # Per-CRG independent batch: each node has its OWN receiver
            # geometry (padded to n_shared) + a validity mask on the padded
            # tail. ``valid_mask``/``recv_rows_padded`` are set by the
            # prefetch worker; absent (None) on the shared-shot path.
            _percrg = getattr(batch, "valid_mask", None) is not None
            t = time.perf_counter()
            # Per-CRG uses an on-demand per-node solve loop, so keep the full
            # (B, max_nrec, nt) obs on the HOST — only one node's slice is
            # moved to the GPU at a time (all traces at once would not fit).
            obs_t = torch.as_tensor(
                traces_per_group, dtype=torch.float32,
                device=("cpu" if _percrg else dev),
            )
            # SIREN-frame alignment: shift obs forward in time by the
            # wavelet's ``source_delay_s`` (computed once at setup) so
            # the main wavelet bang at sample ``obs_prepad_samples`` in
            # syn lines up with the actual event in obs. ``torch.roll``
            # is cyclic, then we zero the wrapped tail to be safe
            # (legacy ``_run_fwi_3d_crg.py:1715-1719`` behaviour).
            if obs_prepad_samples > 0:
                obs_t = torch.roll(obs_t, shifts=obs_prepad_samples, dims=-1)
                obs_t[..., :obs_prepad_samples] = 0.0
            t_h2d = time.perf_counter() - t
            # The prefetcher absorbed the SEG-Y read AND the dt resample;
            # ``t_wait`` is the blocking portion (~0 when compute > io).
            t_sample = 0.0  # sampling now happens inside the prefetch worker
            t_io = t_wait + t_h2d
            t_resample = 0.0  # resample now happens inside the prefetch worker

            # Grid-indexed sources / receivers for this iter.
            sources_grid = src_grid_xyz[batch.group_indices]  # (B, 3)
            _diving_asinh_slot = (None if _diving_asinh_g is None
                                  else _diving_asinh_g[batch.group_indices])  # (B, 3)
            if _diving_asinh_slot is not None and not (_percrg and not dd_on):
                # Fail loudly: the window is applied ONLY in the per-CRG
                # on-demand misfit loop. Anywhere else it is a silent no-op and
                # a "diving-wave" inversion would quietly fit the FULL record —
                # reflections, multiples and all. Note the dispatch below reads
                # ``if dd_on: ... elif _percrg:``, so DD WINS even when
                # per_crg_independent is true: testing `_percrg` alone is not
                # enough to know the window will actually be reached.
                raise ValueError(
                    "loss.diving_window_db is applied only in the per-CRG on-demand "
                    "misfit loop, but this run takes the "
                    f"{'DD (dd_on wins over per_crg_independent)' if dd_on else 'shared-receiver'}"
                    " misfit branch, where the window would be silently ignored."
                )
            if _percrg:
                # No shared receiver grid — each node's geometry is built
                # per-shot below from batch.recv_rows_padded. rows0 (for QC)
                # is node 0's own (padded) rows.
                recv_grid = None
                rows0_used = batch.recv_rows_padded[0]
            else:
                recv_grid = plan_grid_xyz[batch.rows_per_group[0]]  # (n_shared, 3)
                # Track the post-dedup plan-row indices in lock-step with
                # obs_t / recv_grid so downstream QC can join back to the
                # plan (file_id, sx_utm, etc.) for sorting / labelling.
                rows0_used = batch.rows_per_group[0]

            # Optional per-iter dedupe: collapse duplicate (gx,gy,gz) cells.
            # ``"first"`` keeps the first hit; ``"nearest"`` keeps the
            # receiver whose pre-quantisation model-xy was closest to the
            # cell center. Apply the same row mask to obs traces across
            # all groups so the encoded supershot stays receiver-consistent.
            dedup_mode = getattr(sampling_cfg, "dedup_mode", "none")
            _msync(); _t_de = time.perf_counter()
            if (not _percrg) and dedup_mode != "none" and n_shared > 1:
                if dedup_mode == "first":
                    _, keep_idx = np.unique(
                        recv_grid, axis=0, return_index=True,
                    )
                    keep_idx = np.sort(keep_idx)
                else:  # "nearest"
                    # Compute model-frame xy of each shared shot (already
                    # available via rows_per_group[0] + frame.to_model).
                    rows0 = batch.rows_per_group[0]
                    mxy = frame.to_model(plan.row_source_xyz[rows0, :2])
                    mz = plan.row_source_xyz[rows0, 2]
                    cell_center_x = recv_grid[:, 0] * dx_m + origin_x
                    cell_center_y = recv_grid[:, 1] * dy_m + origin_y
                    cell_center_z = recv_grid[:, 2] * dz_m + origin_z
                    d2 = ((mxy[:, 0] - cell_center_x) ** 2
                          + (mxy[:, 1] - cell_center_y) ** 2
                          + (mz - cell_center_z) ** 2)
                    # Bucket rows by their (gx,gy,gz) cell key.
                    key = (
                        (recv_grid[:, 2].astype(np.int64) * (1 << 22))
                        | (recv_grid[:, 1].astype(np.int64) * (1 << 11))
                        | recv_grid[:, 0].astype(np.int64)
                    )
                    # Vectorised per-cell argmin(d2): lexsort by (d2, key) so
                    # each cell's rows are contiguous (primary=key) and ascending
                    # in d2 (secondary), then keep the FIRST row of each cell =
                    # the nearest (min-d2) one. Replaces a Python per-cell loop
                    # that ran on the main thread and left the GPU idle every
                    # iteration on dense OBN supershots.
                    order = np.lexsort((d2, key))
                    key_s = key[order]
                    first_of_cell = np.concatenate(
                        [[True], key_s[1:] != key_s[:-1]]
                    )
                    keep_idx = np.sort(order[first_of_cell].astype(np.int64))
                if keep_idx.size < n_shared:
                    obs_t = obs_t[:, keep_idx, :].contiguous()
                    recv_grid = recv_grid[keep_idx]
                    rows0_used = rows0_used[keep_idx]
                    n_shared = int(keep_idx.size)
            _msync(); t_dedup = time.perf_counter() - _t_de
            _msync(); _t_ob = time.perf_counter()
            valid_mask_local = None  # set in the per-shot per-CRG branch below
            if encoding_on:
                # ----- encoded supershot path (1 GPU, B=1) -----
                # source-encoding (sweep IO contract geophyai 24e91c9): sources
                # (1, nsrc, 3), receivers batch 1, wavelet (nsrc, nt) [per-source signed].
                sources_super = sources_grid[None, :, :].astype(np.int64)
                receivers_super = recv_grid[None, :, :].astype(np.int64)
                signs_np = sign_rng.choice([-1.0, 1.0], size=B).astype(np.float32)
                signs_t = torch.as_tensor(signs_np, device=dev)
                wavelet_super = (wavelet_t[None, :] * signs_t[:, None])              # (B, nt)
                obs_super = (obs_t * signs_t[:, None, None]).sum(dim=0, keepdim=True)
                # Bandpass the encoded obs supershot to match the (already
                # bandpassed) wavelet. Linearity: filtering each obs_i then
                # summing == filtering the sum (so equivalent to per-slot
                # pre-filtering, with one filter call per iter).
                if bandpass_spec is not None:
                    obs_super = _bandpass_torch_fft(
                        obs_super, lo=float(bandpass_spec.lo_hz),
                        hi=float(bandpass_spec.hi_hz),
                        dt=float(effective_dt),
                        order=int(bandpass_spec.order),
                        axis=-1,
                    )
                # local slice metadata for the encoded path: all on rank 0.
                local_start, local_end, dummy_slice = 0, B, False
            else:
                # ----- per-shot path (multi-GPU via DDP, B_local per rank) -----
                # Slice the B-batch contiguously across ranks; the rest of
                # the loop runs ONLY on the local slice. Gradients (and
                # illuminations) are all-reduce'd after backward so each
                # rank sees the full-batch update before optimizer.step.
                if dist_info.is_distributed:
                    rank = int(dist_info.rank)
                    world_size = int(dist_info.world_size)
                else:
                    rank, world_size = 0, 1
                per_rank = int(math.ceil(B / world_size))
                local_start = rank * per_rank
                local_end = min(local_start + per_rank, B)
                if local_end <= local_start:
                    # This rank drew an empty slice (B < world_size). Run
                    # ONE dummy shot with a near-zero loss multiplier so the
                    # backward + all_reduce participation stays consistent
                    # across ranks (collective ops must be called every iter
                    # by every rank). ``dummy_slice`` flag suppresses any
                    # contribution to the synced gradient.
                    local_start, local_end = 0, 1
                    dummy_slice = True
                else:
                    dummy_slice = False
                B_local = local_end - local_start
                sources_local = sources_grid[local_start:local_end].astype(np.int64)
                # sweep IO contract (geophyai 24e91c9): per-shot multi-shot uses
                # 2-D sources (nshots, ndim) + 2-D per-shot wavelet (nshots, nt);
                # 3-D (1, nsrc, ndim) is reserved for source-encoding mode.
                sources_super = sources_local                                      # (B_local, 3)
                if _percrg:
                    # Per-node receiver geometry (each node's own aperture),
                    # padded to n_shared; padded tail masked out of the loss.
                    receivers_super = plan_grid_xyz[
                        batch.recv_rows_padded[local_start:local_end]
                    ].astype(np.int64)                                             # (B_local, n_shared, 3)
                    valid_mask_local = torch.as_tensor(
                        batch.valid_mask[local_start:local_end], device=dev,
                    )                                                              # (B_local, n_shared) bool
                else:
                    receivers_super = np.broadcast_to(
                        recv_grid[None, :, :], (B_local, n_shared, 3),
                    ).astype(np.int64).copy()                                      # (B_local, n_shared, 3)
                wavelet_super = (
                    wavelet_t[None, :].expand(B_local, -1).contiguous()
                )                                                                  # (B_local, nt)
                obs_super = obs_t[local_start:local_end].contiguous()              # (B_local, n_shared, nt)
                if bandpass_spec is not None:
                    obs_super = _bandpass_torch_fft(
                        obs_super, lo=float(bandpass_spec.lo_hz),
                        hi=float(bandpass_spec.hi_hz),
                        dt=float(effective_dt),
                        order=int(bandpass_spec.order),
                        axis=-1,
                    )

            _msync(); t_obsbp = time.perf_counter() - _t_ob
            optimizer.zero_grad()
            _msync(); _t_re = time.perf_counter()
            if reparam_net is None:
                models = [vp_leaf]
                v_leaf_for_illum = vp_leaf
            elif dd_on:
                # DD Level-B: render ONLY this rank's tile window so the reparam
                # memory divides across tiles (validated in runs/dd_ifwi_smoke).
                _dd_bounds = _dd_tile_bounds(solver)
                base_leaf = _dd_render_tile(reparam_net, _dd_bounds, dd_rc)
                models = [base_leaf]
                v_leaf_for_illum = base_leaf
            elif spec.reparam.backward_mode == "single_step":
                models = [reparam_net()]
                v_leaf_for_illum = None  # net params; illum precond no-op
            else:
                # Chunk the forward render (bit-identical, pointwise) so the
                # single-GPU hash-render peak stays bounded — same knob the DD
                # tile render uses (SWEEP_DD_RENDER_CHUNK -> dd_rc, default 8).
                with torch.no_grad():
                    base_leaf = reparam_net.render(chunk_rows=dd_rc).detach().clone()
                base_leaf = base_leaf.requires_grad_(True)
                models = [base_leaf]
                v_leaf_for_illum = base_leaf
            _msync(); t_render = time.perf_counter() - _t_re

            _memprof = os.environ.get("SWEEP_MEM_PROFILE") == "1"
            if _memprof and epoch == 0 and dev.type == "cuda":
                torch.cuda.synchronize(dev)
                _mem_pre_fwd = torch.cuda.memory_allocated(dev)
            t = time.perf_counter()
            if dd_on:
                # Per-iter shared-shot sampling varies each tile's owned
                # source/receiver COUNT, but ModelParallel._set_geometry now
                # reallocates only the two count-sized buffers (record ~ nrec,
                # grad-wavelet ~ nsrc) and reuses the model-sized wavefield +
                # boundary buffers, so the geometry is swapped in place — no
                # forced re-capture. (The old force-recapture-every-iter
                # reallocated every buffer + the boundary ring each iter,
                # leaking ~one adjoint-wavefield set/iter and paying a throwaway
                # fwd+bwd.) First iter of each stage still captures via the fresh
                # per-stage solver; later iters hit the _set_geometry fast path.
                # ModelParallel infers the encoded supershot from shapes; the
                # source_encoding kwarg was removed from the core API.
                syn = solver(wavelet_super, sources_super, receivers_super,
                             models=models)
            elif _percrg:
                # On-demand per-node loop: move each node's obs to the GPU,
                # solve, backward, free — so only ONE node's obs+syn
                # is resident, not all B (tens of GB at full coverage). Each node's
                # backward accumulates into base_leaf.grad; global_norm counts
                # valid elements across the whole batch (set before the loop).
                _nrec_j = valid_mask_local.sum(dim=1).to(torch.int64).cpu().numpy()
                _nt_p = int(obs_t.shape[-1])
                loss_t = torch.zeros((), device=dev)
                syn = None
                _B_loc = local_end - local_start

                def _win_j(_j, _recv_j):
                    """Diving-wave window for node _j -> (1, nt, nij, 1) or None.
                    CRG reciprocity: sources_super[_j] IS the node, _recv_j are
                    the survey shots, which is exactly what the mask expects.

                    Must pass _cur_dh, NOT dx_m: a stage with ``dh_m`` re-projects
                    src_grid_xyz/plan_grid_xyz onto the stage grid (_apply_stage_dh
                    -> _grid_for_dh) and moves _cur_dh with them, but dx_m stays
                    pinned to the base spacing. Feeding dx_m would scale every
                    offset by _cur_dh/dx_m — on a 75 -> 37.5 m refinement that is
                    2x, which puts the window entirely off the diving wave, with
                    no crash and no shape error."""
                    if _diving_asinh_slot is None:
                        return None
                    return _diving_window_mask(
                        sources_super[_j:_j + 1][:, None, :], _recv_j,
                        _diving_asinh_slot[local_start + _j:local_start + _j + 1],
                        _nt_p, effective_dt, _cur_dh, spec.loss, dev,
                        frame_delay_s=obs_prepad_samples * effective_dt,
                    )

                _n_samp = float(int(valid_mask_local.sum().item()) * _nt_p) or 1.0
                if _diving_asinh_slot is None or spec.loss.kind == "trace_cosine":
                    # trace_cosine's "pointwise" tensor is the PER-TRACE value
                    # broadcast across time, so windowing does NOT shrink the sum
                    # (each surviving trace still contributes (1-cos)*nt) and
                    # sum/(traces*nt) stays the per-trace mean. Normalising it by
                    # the kept-sample count instead would inflate the gradient by
                    # 1/kept_fraction — ~3x here, and batch-dependent (the kept
                    # fraction varies 27-34% per node), i.e. a drifting effective lr.
                    global_norm = _n_samp
                else:
                    # Pointwise misfits DO shrink with the window (~70% of the
                    # summed residual is muted), so normalise by the KEPT sample
                    # weight to keep the gradient scale — and the tuned lr —
                    # comparable to an unwindowed run. Needs a geometry-only
                    # pre-pass because backward() runs per node below and so needs
                    # the norm up front; recomputing each mask is far cheaper than
                    # holding all B of them (tens of MB each at a production
                    # nt x nrec in float32), which is this loop's whole purpose.
                    _gn = 0.0
                    for _j in range(_B_loc):
                        _w0 = _win_j(_j, receivers_super[_j, :int(_nrec_j[_j])][None])
                        _gn += float(_w0.sum().item())
                        del _w0
                    global_norm = _gn or 1.0
                # DDP: global_norm MUST be the GLOBAL kept-sample count. Each rank
                # normalises its own nodes' loss by this before backward, and the
                # per-shot path all_reduce-SUMs vp_leaf.grad across ranks (~line
                # 1847). With a LOCAL norm (this rank's ~B/world_size nodes) that
                # SUM inflates the gradient by ~world_size — an 8x effective lr on
                # 8 GPUs that silently diverges a long run. all_reduce-ing the norm
                # makes the summed, globally-normalised gradient equal the 1-GPU run.
                if dist_info.is_distributed:
                    import torch.distributed as _td
                    _gtn = torch.tensor(float(global_norm), device=dev)
                    _td.all_reduce(_gtn, op=_td.ReduceOp.SUM)
                    global_norm = float(_gtn.item()) or 1.0
                for _j in range(_B_loc):
                    _nij = int(_nrec_j[_j])
                    _recv_j = receivers_super[_j, :_nij][None]                  # (1,nij,3)
                    _obs_j = obs_t[local_start + _j, :_nij].to(dev)[None]       # (1,nij,nt)
                    if bandpass_spec is not None:
                        # obs MUST be band-limited HERE. The stage bandpass upstream
                        # writes into ``obs_super``, but this loop re-slices the RAW
                        # host ``obs_t`` — so that filter is a DEAD STORE on the
                        # per-CRG path and obs reached the misfit unfiltered. Measured
                        # on a low-frequency field config: a tiny fraction of obs energy
                        # in band, most of the rest far above it, while syn
                        # (band-limited via the wavelet, ~line 1346) had ~80% in
                        # band. Two signals in
                        # disjoint bands are near-orthogonal, which pinned trace_cosine
                        # at ~1.0 (cos 0.015); filtering here lifts cos to 0.52-0.74.
                        # Done per node on-device after the H2D copy: obs_t is
                        # deliberately kept on the host (a full batch would not fit
                        # on-device), so filtering it whole would undo that.
                        _obs_j = _bandpass_torch_fft(
                            _obs_j, lo=float(bandpass_spec.lo_hz),
                            hi=float(bandpass_spec.hi_hz), dt=float(effective_dt),
                            order=int(bandpass_spec.order), axis=-1)
                        if obs_prepad_samples > 0:
                            # Re-zero the prepad head: the zero-phase filter smears
                            # energy back across the roll's zeroed samples.
                            _obs_j = _obs_j.clone()
                            _obs_j[..., :obs_prepad_samples] = 0.0
                    _syn_j = solver(
                        wavelet_super[_j:_j + 1], sources_super[_j:_j + 1],
                        _recv_j, models=models, source_encoding=False,
                    )
                    _obs_4d = _obs_j.permute(0, 2, 1).unsqueeze(-1)
                    _w = _win_j(_j, _recv_j)
                    if _w is None:
                        _loss_j = loss_fn(_syn_j, _obs_4d).sum()
                    else:
                        # MUTE-THEN-MISFIT: window syn AND obs before the misfit,
                        # so the misfit only ever sees the diving-wave window.
                        _loss_j = loss_fn(_syn_j * _w, _obs_4d * _w).sum()
                        # QC the data the misfit ACTUALLY sees. Dumped here, at
                        # the loss call site, so it is the very tensors that go
                        # into loss_fn, rather than a side reconstruction that
                        # could agree with the picks while the real misfit sees
                        # something else. That distinction is not academic: this
                        # dump is what exposed obs arriving unfiltered.
                        if (_j == 0 and dist_info.is_root
                                and epoch % spec.show_every == 0):
                            try:
                                _rec_j = receivers_super[_j, :_nij]
                                _off_j = (np.linalg.norm(
                                    _rec_j - sources_super[_j], axis=1) * _cur_dh)
                                # Pass real positions: the QC sorts by acquisition
                                # position, not offset. The window is a function of
                                # |offset| and so is azimuthally symmetric, which
                                # offset-sorting would hide.
                                _dump_diving_window_qc(
                                    _obs_4d, _syn_j, _w, _off_j, effective_dt,
                                    epoch, int(batch.group_indices[local_start + _j]),
                                    qc_dir,
                                    rec_xy=_rec_j[:, :2] * _cur_dh,
                                    node_xy=sources_super[_j][:2] * _cur_dh)
                            except Exception as _e:      # QC must never kill a run
                                print(f"[multisource] diving-window QC failed: {_e}")
                    (_loss_j / global_norm).backward()
                    loss_t = loss_t + _loss_j.detach()
                    if _j == _B_loc - 1:
                        syn = _syn_j.detach()
                        obs_super = _obs_j
            else:
                syn = solver(
                    wavelet_super, sources_super, receivers_super,
                    models=models, source_encoding=encoding_on,
                )
            _csync()
            t_fwd = time.perf_counter() - t
            if _memprof and epoch == 0 and dev.type == "cuda":
                _mem_post_fwd = torch.cuda.memory_allocated(dev)
            # Adapt obs to syn's canonical layout (n, nt, nrec, 1).
            # ``obs_super`` from the CRG prefetcher is (n, nrec, nt), so we
            # always permute + unsqueeze. After geophyai 21041c5 both
            # backends emit syn as 4-D canonical, so this is unconditional.
            # Per-CRG did its loss+backward inside the per-node loop above.
            obs_match = (None if _percrg
                         else obs_super.permute(0, 2, 1).unsqueeze(-1).contiguous())
            # Lateral receiver smoothing: the SAME linear operator on obs and
            # syn (see ReceiverSmoothingSpec). Built per iter because the
            # shared-shot set is resampled every iteration.
            if _recsm is not None and _percrg:
                raise ValueError(
                    "receiver_smoothing is not supported on the per-CRG "
                    "(ragged batch) path: its misfit is accumulated node by "
                    "node over padded receiver subsets, so the smoothing "
                    "stencil would straddle the padding. Use the encoded "
                    "supershot path (source_encoding.enabled=true).")
            if _recsm is not None and obs_match is not None:
                _rxy = torch.as_tensor(
                    np.asarray(receivers_super)[0, :, :2], device=dev,
                    dtype=torch.float32)
                _W = _receiver_smoothing_matrix(
                    _rxy, float(_recsm.sigma_cells),
                    cutoff_sigmas=float(_recsm.cutoff_sigmas), device=dev)
                if _recsm.target == "both":
                    if syn.dim() == 3:
                        syn = syn.permute(0, 2, 1).unsqueeze(-1).contiguous()
                    syn = _apply_receiver_smoothing(syn, _W)
                obs_match = _apply_receiver_smoothing(obs_match, _W)
            t = time.perf_counter()
            if _percrg:
                pass  # loss + backward already done in the on-demand loop
            elif dd_on:
                # ModelParallel returns a 3-D record (ns, nrec, nt); make it
                # canonical 4-D (ns, nt, nrec, 1) to match the single-domain
                # backends and obs_match (already 4-D) before the loss.
                if syn.dim() == 3:
                    syn = syn.permute(0, 2, 1).unsqueeze(-1).contiguous()
                # DD: each tile records only its OWN receivers (ordered by
                # solver._own_rec_idx); subset obs to match syn, then all_reduce
                # the misfit sum AND the element count so the global-normalised
                # gradient equals the single-GPU run. (all_reduce-sum backward is
                # identity for the local term -> correct per-tile gradient.)
                _own = solver._own_rec_idx
                if _own:
                    obs_tile = obs_match[:, :, _own, :]
                    loss_t = loss_fn(syn, obs_tile).sum()
                    _local_numel = float(syn.numel())
                else:
                    # This tile owns NO receivers for this (reseeded) supershot:
                    # the DD forward emitted a dummy placeholder receiver so the
                    # solver stays happy (syn shape (1, nt, 1, 1)), but obs has 0.
                    # Contribute 0 loss / 0 count -- the tile's gradient comes from
                    # neighbour adjoint wavefield via halo, not a local misfit.
                    # Keep syn in the graph (x0) and still all_reduce so the
                    # collective backward stays in sync across ranks.
                    loss_t = syn.sum() * 0.0
                    _local_numel = 0.0
                import torch.distributed as _td
                _tn = torch.tensor(_local_numel, device=dev)
                _td.all_reduce(_tn, op=_td.ReduceOp.SUM)
                global_norm = float(_tn.item()) or 1.0
                _td.all_reduce(loss_t, op=_td.ReduceOp.SUM)
                (loss_t / global_norm).backward()
            else:
                if valid_mask_local is not None:
                    # Per-CRG ragged batch: zero the padded-receiver tail in
                    # BOTH syn and obs so it drops out of the (L2/L1/Huber)
                    # misfit, and normalise by the VALID element count only.
                    _m = valid_mask_local[:, None, :, None].to(syn.dtype)      # (n,1,nrec,1)
                    loss_t = loss_fn(syn * _m, obs_match * _m).sum()
                    global_norm = float(syn.numel()) * max(
                        float(valid_mask_local.float().mean()), 1.0e-6)
                else:
                    loss_t = loss_fn(syn, obs_match).sum()
                    global_norm = float(syn.numel())
                # Dummy-slice ranks contribute zero loss / zero grad so the
                # all_reduce stays consistent across ranks; non-dummy ranks
                # use their full per-rank loss.
                loss_scale = 0.0 if (not encoding_on and dummy_slice) else 1.0
                (loss_t / global_norm * loss_scale).backward()
            _csync()
            t_bwd = time.perf_counter() - t

            # DDP gradient sync (per-shot path only). Each rank's
            # vp_leaf.grad holds B_local shots' partial contribution; sum
            # across ranks reproduces the full-batch gradient. The encoded
            # path stays 1-GPU and skips this entirely.
            if (not encoding_on
                    and dist_info.is_distributed
                    and v_leaf_for_illum is not None
                    and v_leaf_for_illum.grad is not None):
                import torch.distributed as _td
                _td.all_reduce(v_leaf_for_illum.grad, op=_td.ReduceOp.SUM)
                # loss_t is this rank's node-subset sum; SUM it so the logged
                # loss_t/global_norm is the true global per-trace mean, not the
                # ~1/world_size fragment each rank would otherwise report.
                _td.all_reduce(loss_t, op=_td.ReduceOp.SUM)
                # Also sync illuminations stored on the solver for the
                # precond step below (per-rank partial sums → global sum).
                for _attr in ("source_illumination", "receiver_illumination"):
                    _ill = getattr(solver, _attr, None)
                    if isinstance(_ill, torch.Tensor):
                        _td.all_reduce(_ill, op=_td.ReduceOp.SUM)
            losses.append(float(loss_t.detach().cpu()) / global_norm)

            # Raw dL/dvp dump when illum precond is OFF (the illum branch below
            # already dumps the pre-precond grad when on). SWEEP_DUMP_GRAD=1,
            # epoch 0, root only. base_leaf.grad still intact here (the reparam
            # backward that consumes it runs after the illum block).
            if (os.environ.get("SWEEP_DUMP_GRAD") == "1" and not illum_on
                    and dist_info.is_root and epoch == 0
                    and v_leaf_for_illum is not None
                    and v_leaf_for_illum.grad is not None):
                import numpy as _np
                _dd = os.environ.get("SWEEP_DUMP_DIR", "/tmp")
                _np.save(_dd + "/grad_raw.npy",
                         v_leaf_for_illum.grad.detach().cpu().numpy())
                print(f"[dump] raw dL/dvp -> {_dd}/grad_raw.npy "
                      f"shape={tuple(v_leaf_for_illum.grad.shape)}", flush=True)

            # --- Illumination precond (single_step + grid; or two_pass leaf).
            # DD v1: disabled (ModelParallel exposes no illum tensors; v_leaf is
            # a tile). See _dd_config / the [dd] setup banner.
            if illum_on and not dd_on:
                sill = getattr(solver, "source_illumination", None)
                rill = getattr(solver, "receiver_illumination", None)
                if reparam_net is None or v_leaf_for_illum is not None:
                    target_grad = v_leaf_for_illum.grad if v_leaf_for_illum is not None else None
                    # DUMP grad before/after illum precond. SWEEP_DUMP_GRAD=1
                    # writes the iter-0 raw/precond/illum once. Additionally
                    # SWEEP_DUMP_GRAD_NMEAN>0 accumulates the RAW grad (and
                    # S*R illum) over EVERY iter and writes a running MEAN.
                    # With the model frozen (lr=0) this mean is the full-batch
                    # gradient estimate: E over the per-iter shared-shot draws
                    # = the all-shots gradient the stochastic mini-batches
                    # sample (the supershot geometry forbids all groups at
                    # once, so we average instead).
                    _DG = (os.environ.get("SWEEP_DUMP_GRAD") == "1"
                           and dist_info.is_root and target_grad is not None)
                    _nmean = int(os.environ.get("SWEEP_DUMP_GRAD_NMEAN", "0") or 0)
                    if _DG:
                        import numpy as _np
                        _dd = os.environ.get("SWEEP_DUMP_DIR", "/tmp")
                        _graw = target_grad.detach().cpu().numpy()
                        _illsr = ((sill * rill).detach().cpu().numpy()
                                  if (sill is not None and rill is not None)
                                  else None)
                        if epoch == 0:
                            _np.save(_dd + "/grad_raw.npy", _graw)
                            if _illsr is not None:
                                _np.save(_dd + "/illum_sr.npy", _illsr)
                        if _nmean > 0:
                            _acc = getattr(self, "_graddump_acc", None)
                            if _acc is None:
                                _acc = {"g": _np.zeros_like(_graw),
                                        "i": (None if _illsr is None
                                              else _np.zeros_like(_illsr)),
                                        "n": 0}
                                self._graddump_acc = _acc
                            _acc["g"] += _graw
                            if _illsr is not None and _acc["i"] is not None:
                                _acc["i"] += _illsr
                            _acc["n"] += 1
                            _np.save(_dd + "/grad_raw_mean.npy",
                                     _acc["g"] / _acc["n"])
                            if _acc["i"] is not None:
                                _np.save(_dd + "/illum_sr_mean.npy",
                                         _acc["i"] / _acc["n"])
                            print(f"[graddump] mean over n={_acc['n']} iters",
                                  flush=True)
                    _apply_illumination_precond(
                        target_grad, sill, rill,
                        eps=illum_spec.epsilon, exponent=illum_spec.exponent,
                        relative_epsilon=getattr(illum_spec, "relative_epsilon", None),
                    )
                    if _DG and epoch == 0:
                        _np.save(_dd + "/grad_precond.npy",
                                 target_grad.detach().cpu().numpy())

            # --- Smooth regularization (TVPrior on the current vp/leaf).
            # SYNCED profile (SWEEP_TASKS_TPROF=1): time smooth-reg separately
            # and sync before the reparam timer so t_reparam_bwd is PURE reparam
            # (otherwise the smooth-reg backward's async tail bleeds into it).
            _tprof = os.environ.get("SWEEP_TASKS_TPROF") == "1"
            if _tprof and dev.type == "cuda": torch.cuda.synchronize(dev)
            _t_reg0 = time.perf_counter()
            if tv_prior is not None and v_leaf_for_illum is not None and not dd_on:
                reg_loss = float(smooth_spec.weight) * tv_prior(v_leaf_for_illum)
                # Accumulate into v_leaf_for_illum.grad (.backward adds).
                reg_loss.backward(retain_graph=False)
            if _tprof and dev.type == "cuda": torch.cuda.synchronize(dev)
            t_smoothreg = time.perf_counter() - _t_reg0

            # --- Seabed-freeze mask on the leaf gradient (post illum +
            # post smooth-reg so any mask zeros take precedence).
            if seabed_mask is not None and v_leaf_for_illum is not None and not dd_on:
                seabed_mask.apply_to(v_leaf_for_illum.grad)

            # --- Reparam two-pass: push leaf grad through the net.
            # UNCONDITIONALLY drain any in-flight solver-adjoint async tail BEFORE
            # the reparam timer and attribute that drain to t_bwd (where it
            # belongs) — otherwise the tail bleeds into t_reparam_bwd and makes it
            # look like the SIREN backward (which is really ~0.05 s) when it is in
            # fact the solver adjoint (scales with B). t_drain is printed so the
            # split is auditable without SWEEP_TASKS_TPROF.
            _t_drain0 = time.perf_counter()
            _csync()
            t_drain = time.perf_counter() - _t_drain0
            t_bwd += t_drain
            t = time.perf_counter()
            if dd_on and reparam_net is not None:
                # DD Level-B: push the TILE velocity grad through the net params
                # (z-chunked windowed backward) then all_reduce net-param grads
                # across tiles. Mirrors the validated runs/dd_ifwi_smoke path.
                _dd_backward_tile(reparam_net, v_leaf_for_illum,
                                  _dd_tile_bounds(solver), dd_rc)
            elif (reparam_net is not None
                    and spec.reparam.backward_mode == "two_pass_full"):
                v_grad = v_leaf_for_illum.grad
                if v_grad is not None:
                    rendered = reparam_net()
                    rendered.backward(v_grad)
            elif (reparam_net is not None
                  and spec.reparam.backward_mode == "two_pass_chunked"):
                v_grad = v_leaf_for_illum.grad
                if v_grad is not None and os.environ.get("SWEEP_REPARAM_CUDAGRAPH") == "1" \
                        and hasattr(reparam_net, "backward_velocity_gradient_graphed") \
                        and dev.type == "cuda":
                    # Root-cure for the prefetch-GIL stall: replay a captured
                    # render+backward graph (1 host launch) instead of ~500
                    # launches that get GIL-starved by the SEG-Y prefetch
                    # threads. Numerically equivalent (cosine 0.9999999).
                    reparam_net.backward_velocity_gradient_graphed(v_grad)
                elif v_grad is not None:
                    reparam_net.backward_velocity_gradient(
                        v_grad, chunk_rows=int(spec.reparam.backward_chunk_rows),
                    )
            _csync()
            t_reparam_bwd = time.perf_counter() - t

            t = time.perf_counter()
            optimizer.step()
            _csync()
            t_opt = time.perf_counter() - t
            if _memprof and epoch == 0 and dev.type == "cuda":
                _nn_p = (sum(p.numel() * p.element_size() for p in reparam_net.parameters())
                         if reparam_net is not None else vp_leaf.numel() * vp_leaf.element_size())
                _nn_o = 0
                for _st in optimizer.state.values():
                    for _v in _st.values():
                        if torch.is_tensor(_v):
                            _nn_o += _v.numel() * _v.element_size()
                _tot = torch.cuda.memory_allocated(dev)
                _pk = torch.cuda.max_memory_allocated(dev)
                _res = torch.cuda.memory_reserved(dev)
                _solver_fwd = _mem_post_fwd - _mem_pre_fwd
                print(f"[mem_profile] epoch0 | total_alloc={_tot/1e9:.2f}G peak={_pk/1e9:.2f}G reserved={_res/1e9:.2f}G "
                      f"|| NN(network): params={_nn_p/1e9:.3f}G optim_states={_nn_o/1e9:.3f}G "
                      f"|| SOLVER fwd-alloc(wavefields+boundary)={_solver_fwd/1e9:.2f}G "
                      f"(net_params_count={sum(p.numel() for p in reparam_net.parameters()) if reparam_net is not None else 0})",
                      flush=True)
            if reparam_net is None:
                bound = _effective_bound(spec.model_bounds, "vp")
                if bound is not None:
                    vp_leaf.data.clamp_(min=bound.min, max=bound.max)
            else:
                # The global vp render is consumed ONLY by snapshots/QC/final-
                # save, so render it on THOSE epochs — not every iter. At 3-D
                # production scale the full-grid render is a large tensor (+
                # chunked compute); doing it every epoch kept a full-grid vp copy
                # resident through the next forward for no reason (visible in
                # SWEEP_MEM_CENSUS as an extra (nz,ny,nx) float32 bucket). The
                # readers below (snapshot_now / qc) share this exact predicate,
                # so inv_by_name["vp"] is always freshly rendered when read.
                _need_vp = (epoch % spec.show_every == 0
                            or epoch == total_epochs - 1
                            or (epoch + 1) in stage_starts
                            or (qc_enabled
                                and epoch % spec.qc.every_n_epochs == 0))
                # Root-only: the DD net is replicated, so rank 0's full-grid render
                # IS the global model. Rendering on all ranks was redundant AND a
                # full-lateral z-slab render OOM'd 32 GB V100 tiles at 2-16: the
                # solver working set already fills most of the card, and even
                # render(chunk_rows=1) needs more than the headroom. The
                # z+y-tiled CPU render bounds render overhead to about a GB and lands
                # inv_by_name["vp"] on CPU (its only consumers are snapshot .npy /
                # well-log QC, both .cpu().numpy()).
                if _need_vp and dist_info.is_root:
                    with torch.no_grad():
                        if dd_on:
                            inv_by_name["vp"] = _render_full_to_cpu_tiled(
                                reparam_net, cz=1, cy=294)
                        else:
                            inv_by_name["vp"] = reparam_net().detach()

            iter_s = time.perf_counter() - t_iter
            cache_str = ""
            if hasattr(reader, "trace_cache_stats"):
                cs = reader.trace_cache_stats
                if cs is not None:
                    total = cs["hits"] + cs["misses"]
                    hit_rate = cs["hits"] / total if total > 0 else 0.0
                    cache_str = (f"  cache:[hit_rate={hit_rate:.1%} "
                                 f"entries={cs['entries']:,} "
                                 f"bytes={cs['bytes']/1e9:.1f}GB]")
            # Prefetch-worker per-stage breakdown for THIS iter.
            # prefetch_timings[-1] = the iter we just consumed (FIFO).
            pf_str = ""
            if prefetch_timings:
                pf = prefetch_timings[-2] if len(prefetch_timings) >= 2 else prefetch_timings[-1]
                # NOTE: the timing belongs to the iter we just finished
                # waiting on (= iter N); the next-submitted future is iter
                # N+1. With prime_pool[0] = iter 0, the pool of timings
                # contains [iter 0, iter 1, ...] in order; len at this
                # point equals number of FINISHED prefetch loads.
                pf_str = (f"  prefetch:[sample={pf['sample']:.2f} "
                          f"alloc={pf['alloc']:.2f} read={pf['read']:.2f} "
                          f"resample={pf['resample']:.2f} trim={pf['trim']:.2f}]")
            ms_str = ""
            if _msprof:
                ms_str = (f"  MSPROF:[dedup={t_dedup:.2f} "
                          f"obs_encode_bandpass={t_obsbp:.2f} "
                          f"reparam_render={t_render:.2f}]")
            _memstr = ""
            if dev.type == "cuda":
                _memstr = (f" memGB={torch.cuda.memory_allocated(dev) / 2**30:.1f}"
                           f"/{torch.cuda.max_memory_allocated(dev) / 2**30:.1f}pk")
                torch.cuda.reset_peak_memory_stats(dev)   # per-epoch peak trend
                if os.environ.get("SWEEP_MEM_CENSUS") == "1" and dist_info.is_root:
                    # Leak hunt: bucket EVERY live CUDA tensor by (shape,dtype,
                    # requires_grad,has_grad_fn). A bucket whose n grows each
                    # epoch — or buckets={} steadily climbing (reseed-keyed
                    # per-iter shapes) — names the retained allocation.
                    import gc as _gc, collections as _co
                    _buck = _co.defaultdict(lambda: [0, 0])
                    for _o in _gc.get_objects():
                        try:
                            if torch.is_tensor(_o) and _o.is_cuda:
                                _k = (tuple(_o.shape), str(_o.dtype),
                                      bool(_o.requires_grad), _o.grad_fn is not None)
                                _buck[_k][0] += 1
                                _buck[_k][1] += _o.element_size() * _o.nelement()
                        except Exception:
                            pass
                    _tot_mb = sum(v[1] for v in _buck.values()) / 2 ** 20
                    print(f"[memcensus] epoch {epoch:04d} "
                          f"live_cuda={_tot_mb:.0f}MB buckets={len(_buck)}", flush=True)
                    for _k, (_c, _b) in sorted(
                            _buck.items(), key=lambda kv: -kv[1][1])[:12]:
                        print(f"[memcensus]   n={_c:4d} {_b / 2 ** 20:8.1f}MB "
                              f"shape={_k[0]} {_k[1]} rg={_k[2]} gf={_k[3]}",
                              flush=True)
            print(f"[multisource] epoch {epoch:04d} loss={losses[-1]:.6e} "
                  f"B={B} n_shared={n_shared} iter_s={iter_s:.2f}{_memstr}  "
                  f"[wait_io={t_wait:.2f} h2d={t_h2d:.2f} resample={t_resample:.2f} "
                  f"fwd={t_fwd:.2f} bwd={t_bwd:.2f} smoothreg={t_smoothreg:.2f} "
                  f"drain={t_drain:.2f} reparam_bwd={t_reparam_bwd:.2f} "
                  f"opt={t_opt:.2f}]{cache_str}{pf_str}{ms_str}", flush=True)

            # --- Snapshots + QC.
            # Also snapshot the LAST iter of each stage (epoch+1 crosses a stage
            # start) BEFORE the next stage resamples the grid, so each band's
            # final model is captured on its OWN native grid (matches srprod's
            # per-band vp_iter snapshots for apples-to-apples comparison).
            snapshot_now = (epoch % spec.show_every == 0
                            or epoch == total_epochs - 1
                            or (epoch + 1) in stage_starts)
            if snapshot_now and dist_info.is_root:   # DD: only rank 0 holds the rendered vp
                vp_now = inv_by_name["vp"]
                np.save(
                    snapshots_dir / f"vp_epoch_{epoch:04d}.npy",
                    vp_now.detach().cpu().numpy(),
                )
            if qc_enabled and (epoch % spec.qc.every_n_epochs == 0
                               or epoch == total_epochs - 1):
                state_for_qc = {
                    "inv_by_name": inv_by_name,
                    "dh": float(dx_m),
                    "dh_xyz": (float(dz_m), float(dy_m), float(dx_m)),
                    "dt": float(effective_dt),
                    "qc_initial_vp": torch.from_numpy(init_vp_np),
                }
                self._run_epoch_qc_safe(
                    spec=spec, state=state_for_qc, qc_dir=qc_dir,
                    epoch=epoch, dev=dev,
                )

                # Multisource supershot QC: obs/syn interleave gather,
                # amplitude spectrum, and the per-iter survey footprint
                # (picked OBN nodes + used physical shots in the FULL
                # acquisition context). Snapshot the CPU tensors from
                # this iter — obs_super / syn are still alive here. Use
                # rows0_used (post-dedup) so length matches obs/syn.
                # Gated on ``qc.supershot_panel`` (independent of the
                # legacy 2D ``shot_gather`` knob).
                # QC panel: encoded path always dumps from the single rank.
                # In per-shot DDP, only rank 0 dumps (and only its first
                # local source as a representative slice — the full-batch
                # acquisition footprint is still shown via
                # ``picked_group_model_xy`` from ``batch.group_indices``).
                _qc_dump_panel = (
                    bool(getattr(spec.qc, "supershot_panel", True))
                    # Per-CRG has no shared supershot geometry (each node its
                    # own ragged receivers); the supershot panel joins on a
                    # single rows0 vector, so skip it here.
                    and not _percrg
                    and (encoding_on
                         or not dist_info.is_distributed
                         or int(dist_info.rank) == 0)
                )
                if _qc_dump_panel:
                    try:
                        from sweep_tasks.qc import save_supershot_qc_panel
                        # All coords in MODEL frame (post-rotation, in meters)
                        # so the QC map matches the frame the inversion
                        # runs in, and the dashed grid-extent rectangle
                        # (xmin/xmax, ymin/ymax in model meters) is
                        # meaningful on the same axes.
                        used_shot_model_xy = plan_model_xy[
                            rows0_used
                        ].astype(np.float64)
                        picked_group_model_xy = src_model_xy[
                            batch.group_indices
                        ].astype(np.float64)
                        f_hi = (float(bandpass_spec.hi_hz)
                                if bandpass_spec is not None else None)
                        # Source-line id per physical-shot column — lets
                        # the plotter sort gather columns by line so the
                        # eye sees along-line moveout cleanly (sampler
                        # natural order is (sx, sy)-key, which scatters
                        # lines randomly). Secondary key = absolute plan-
                        # row index: within a source line, rows are
                        # stored in SEG-Y file order (= along-sail-line
                        # order) regardless of survey orientation.
                        # Sorting by model_x as secondary collapses to
                        # chaos for sail lines that run along the y axis
                        # (common once a survey is rotated into the model frame).
                        used_sourceline_ids = plan.row_file_id[
                            rows0_used
                        ].astype(np.int64)
                        # In per-shot mode, take the first local source so
                        # the panel's obs/syn shapes match the encoded
                        # path's (1, n_shared, *) convention.
                        _obs_for_panel = obs_super.detach().cpu()
                        _syn_for_panel = syn.detach().cpu()
                        if not encoding_on:
                            _obs_for_panel = _obs_for_panel[:1]
                            _syn_for_panel = _syn_for_panel[:1]
                        save_supershot_qc_panel(
                            _obs_for_panel,
                            _syn_for_panel,
                            picked_group_utm_xy=picked_group_model_xy,
                            used_shot_utm_xy=used_shot_model_xy,
                            all_groups_utm_xy=all_groups_model_bg,
                            all_shots_utm_xy=all_shots_model_bg,
                            sourceline_ids=used_sourceline_ids,
                            within_sourceline_sort_key=np.asarray(
                                rows0_used, dtype=np.int64,
                            ),
                            frame_label="model",
                            inversion_extent_xy_m=inversion_extent_xy_m,
                            dt=float(effective_dt),
                            out_path=qc_dir / "supershot" / f"iter_{epoch:04d}.png",
                            epoch=epoch,
                            f_hi_hz=f_hi,
                        )
                    except Exception as ss_err:  # noqa: BLE001
                        print(f"[multisource] supershot QC skipped: {ss_err}")

                # Well-log QC: vp(z) at the 6 pseudo-wells set up at
                # session start. Quick sanity probe on the water layer,
                # seabed jump, and shallow gradient — flat 1500 m/s
                # across the water column is the strong expected signal.
                if bool(getattr(spec.qc, "well_logs", True)):
                    try:
                        from sweep_tasks.qc import save_vp_well_logs_png
                        vp_now_np = inv_by_name["vp"].detach().cpu().numpy()
                        save_vp_well_logs_png(
                            vp_now_np, init_vp_np,
                            well_grid_idx=well_grid_idx,
                            well_labels=well_labels,
                            dz_m=float(dz_m),
                            out_path=qc_dir / "well_logs" / f"iter_{epoch:04d}.png",
                            epoch=epoch,
                            vmin=(_effective_bound(spec.model_bounds, "vp").min
                                  if _effective_bound(spec.model_bounds, "vp") else None),
                            vmax=(_effective_bound(spec.model_bounds, "vp").max
                                  if _effective_bound(spec.model_bounds, "vp") else None),
                        )
                    except Exception as wl_err:  # noqa: BLE001
                        print(f"[multisource] well-log QC skipped: {wl_err}")

                # Gradient ortho-slice QC: dump the per-voxel velocity
                # gradient (post illumination precond + smooth-reg +
                # seabed mask, just before SIREN's reparam_bwd push)
                # next to the vp ortho slices. Uses sweep_image cmap +
                # symmetric percentile clipping so the sign structure is
                # visible. Snapshot ``v_leaf_for_illum.grad`` if it
                # exists (reparam two-pass path); fall back to
                # ``vp_leaf.grad`` for the direct-velocity path.
                # Gated on ``qc.gradient_png`` (shared with the 2D path).
                if bool(getattr(spec.qc, "gradient_png", False)):
                    try:
                        from sweep_tasks.qc import save_gradient_ortho_slices_png
                        grad_now = None
                        if (v_leaf_for_illum is not None
                                and v_leaf_for_illum.grad is not None):
                            grad_now = (
                                v_leaf_for_illum.grad.detach().cpu().numpy()
                            )
                        elif vp_leaf.grad is not None:
                            grad_now = vp_leaf.grad.detach().cpu().numpy()
                        if grad_now is not None:
                            save_gradient_ortho_slices_png(
                                grad_now,
                                dh=(float(dz_m), float(dy_m), float(dx_m)),
                                out_path=qc_dir / "gradient" / f"iter_{epoch:04d}.png",
                                epoch=epoch,
                            )
                    except Exception as g_err:  # noqa: BLE001
                        print(f"[multisource] gradient QC skipped: {g_err}")

                # Loss-curve QC: refresh ``qc/loss_curve.png`` (overwrites
                # in place each QC epoch) + dump the raw ``losses[]`` to
                # ``output/loss.npy`` so the user can monitor convergence
                # without waiting for the final-outputs block. Cheap —
                # one matplotlib call on a few-hundred-point list.
                # Gated on ``qc.loss_curve`` (shared with the 2D path).
                if bool(getattr(spec.qc, "loss_curve", True)):
                    try:
                        if losses:
                            _plot_loss_curve(
                                losses,
                                qc_dir / "loss_curve.png",
                                title=(
                                    f"OBN multisource FWI loss "
                                    f"(epoch {epoch}/{total_epochs - 1})"
                                ),
                            )
                            np.save(
                                out_dir / "loss.npy",
                                np.array(losses, dtype=np.float64),
                            )
                    except Exception as lc_err:  # noqa: BLE001
                        print(f"[multisource] loss-curve QC skipped: {lc_err}")

        # --- 10) Final outputs.
        prefetch_pool.shutdown(wait=False, cancel_futures=True)
        if io_pool is not None:
            io_pool.shutdown(wait=False, cancel_futures=True)
        reader.close()
        artifacts: list[Path] = []
        if dist_info.is_root:   # DD: only rank 0 renders/holds the global vp + writes
            final_vp_path = out_dir / "inverted_vp.npy"
            np.save(final_vp_path, inv_by_name["vp"].detach().cpu().numpy())
            artifacts.append(final_vp_path)
            loss_path = out_dir / "loss.npy"
            np.save(loss_path, np.array(losses, dtype=np.float64))
            artifacts.append(loss_path)
            try:
                artifacts.append(_plot_loss_curve(losses, out_dir / "loss.png",
                                                  title="OBN multisource FWI Loss"))
            except Exception as plot_err:  # noqa: BLE001
                print(f"[multisource] loss plot skipped: {plot_err}")
            # opt-in: dump the reparam net weights so per-level hash features can be
            # rendered offline (reparam.save_net or SWEEP_SAVE_REPARAM_NET=1 env
            # override). Off by default (large file).
            if reparam_net is not None and (
                    spec.reparam.save_net
                    or os.environ.get("SWEEP_SAVE_REPARAM_NET") == "1"):
                net_path = out_dir / "reparam_net.pt"
                torch.save(reparam_net.state_dict(), net_path)
                artifacts.append(net_path)
                print(f"[multisource] saved reparam net -> {net_path}", flush=True)

        summary = {
            "epochs": total_epochs,
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "n_virtual_sources": plan.n_groups,
            "batchsize": int(spec.batchsize),
            "encoding": True,
            "shape": list(shape),
        }
        return artifacts, summary
