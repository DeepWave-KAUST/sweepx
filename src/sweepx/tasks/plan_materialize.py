"""Materialise a static single-source FWI dataset from a SeismicPlan.

This is the ``obs.plan`` (``sampling=None``) path: conventional FWI on
OBN/streamer field data, driven entirely from one YAML, depending only on
sweep-stack (sweep-io ``PlanReader`` + sweep-preproc resample). It produces a
rectangular ``(nshots, nrec, nt)`` dataset where each shot is a SINGLE source
recorded by a FIXED number of its own receivers — the standard
``_fwi_train_step`` loop then does per-shot gradient accumulation
(``train_shot_batchsize``), INR reparam, multi-stage bandpass and illumination
exactly as for synthetic / SEG-Y obs.

Grouping semantics (source = ``group_xyz`` in both cases):
  * CSG  — group = air-gun shot;  receivers = recording nodes
           (``row_receiver_xyz``).
  * CRG  — group = OBN node (reciprocal virtual source);
           receivers = the air-gun positions it recorded
           (``row_source_xyz``).

The geometry pipeline (rotation → model frame → auto grid-origin → model_plan
crop → grid-index projection → in-window filter) mirrors
``_run_fwi_multisource`` setup so a conventional run lands on the SAME grid as
the encoded run it is compared against.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _stamp(msg: str, t0: float, *, verbose: bool) -> float:
    now = time.perf_counter()
    if verbose:
        print(f"[materialize] {msg}: {now - t0:.2f}s", flush=True)
    return now


def _cache_key(spec, init_models, *, effective_dt, effective_nt, source_delay_s) -> str:
    """Stable hash of everything that determines the materialised dataset."""
    geom = spec.geometry
    ocfg = spec.obs.plan
    mp = spec.model_plan
    payload = {
        "plan": str(geom.plan_path),
        "rot": str(geom.rotation_metadata),
        "dh_xyz": list(geom.dh_xyz_m) if geom.dh_xyz_m else None,
        "origin": (list(geom.grid_origin_xyz_m) if geom.grid_origin_xyz_m else None),
        "pad": list(geom.auto_origin_pad_cells or (0, 0, 0)),
        "init_vp": str(next((m.path for m in init_models if m.name == "vp"
                             and m.path is not None), "const")),
        "nrec": ocfg.n_receivers_per_shot,
        "select": ocfg.receiver_select,
        "max_shots": ocfg.max_shots,
        "min_rec": ocfg.min_receivers,
        "seed": ocfg.materialize_seed,
        "z_win": list(mp.z_window_m) if (mp and mp.z_window_m) else None,
        "y_win": list(mp.y_window_m) if (mp and mp.y_window_m) else None,
        "x_win": list(mp.x_window_m) if (mp and mp.x_window_m) else None,
        "dt": float(effective_dt), "nt": int(effective_nt),
        "delay": float(source_delay_s),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def materialize_plan_dataset(
    spec,
    init_models,
    *,
    effective_dt: float,
    effective_nt: int,
    source_delay_s: float = 0.0,
    verbose: bool = True,
):
    """Build a static single-source dataset from ``spec.obs.plan`` / geometry.

    Returns a dict with:
      ``sources``        (nshots, 3) int64 grid indices (x, y, z)
      ``receivers``      (nshots, nrec, 3) int64 grid indices
      ``obs``            (nshots, nt, nrec, 1) float32 numpy (canonical layout)
      ``shape``          (nz, ny, nx) of the cropped model
      ``cropped_models`` {name: cropped float32 array}  (model_plan applied)
      ``native_dt``      float — the dt the obs tensor is sampled at
    """
    from sweep_io.geometry import load_rotation_metadata
    from sweep_io.seismic_plan import PlanReader, SeismicPlan
    from sweep_tasks.preproc.resample import resample_time

    geom = spec.geometry
    ocfg = spec.obs.plan
    t = time.perf_counter()

    # ---- 0) cache check (CRG/CSG materialise once, illum/noillum reuse) -----
    key = _cache_key(spec, init_models, effective_dt=effective_dt,
                     effective_nt=effective_nt, source_delay_s=source_delay_s)
    cache_dir = Path(geom.plan_path).expanduser().parent / ".materialize_cache" / key
    meta_f = cache_dir / "meta.json"
    if meta_f.is_file():
        try:
            meta = json.loads(meta_f.read_text())
            out = {
                "sources": np.load(cache_dir / "sources.npy"),
                "receivers": np.load(cache_dir / "receivers.npy"),
                # Full RAM load (sequential, fast) rather than mmap: avoids
                # per-iter Lustre paging on slow GPU nodes + the non-writable
                # torch.from_numpy warning.
                "obs": np.load(cache_dir / "obs.npy"),
                "shape": tuple(meta["shape"]),
                "cropped_models": {k: np.load(cache_dir / f"model_{k}.npy")
                                   for k in meta["model_names"]},
                "native_dt": float(meta["native_dt"]),
                "nshots": int(meta["nshots"]), "nrec": int(meta["nrec"]),
            }
            if verbose:
                print(f"[materialize] CACHE HIT {cache_dir} (nshots={out['nshots']} "
                      f"nrec={out['nrec']} shape={out['shape']})", flush=True)
            return out
        except Exception as err:  # noqa: BLE001
            if verbose:
                print(f"[materialize] cache read failed ({err}); rebuilding", flush=True)

    # ---- 1) plan + rotation ------------------------------------------------
    plan = SeismicPlan.load(geom.plan_path)
    t = _stamp(f"SeismicPlan.load (n_rows={plan.n_rows:,}, n_groups={plan.n_groups}, "
               f"grouping={plan.grouping})", t, verbose=verbose)
    frame = load_rotation_metadata(geom.rotation_metadata)

    is_crg = (plan.grouping == "crg")
    # source = group; receiver-bearing rows differ by grouping.
    src_xyz_raw = np.asarray(plan.group_xyz, dtype=np.float64)             # (G, 3)
    rec_xyz_raw = (np.asarray(plan.row_source_xyz, dtype=np.float64) if is_crg
                   else np.asarray(plan.row_receiver_xyz, dtype=np.float64))  # (R, 3)

    # ---- 2) model-frame xy -------------------------------------------------
    rec_model_xy = frame.to_model(rec_xyz_raw[:, :2])
    src_model_xy = frame.to_model(src_xyz_raw[:, :2])

    # ---- 3) load vp, auto-origin, model_plan crop --------------------------
    vp_ref = next((m for m in init_models if m.name == "vp"), init_models[0])
    if vp_ref.constant is not None:
        init_vp_np = np.full(tuple(vp_ref.shape), float(vp_ref.constant), dtype=np.float32)
    else:
        init_vp_np = np.load(vp_ref.path).astype(np.float32)

    dh_xyz = geom.dh_xyz_m or (float(spec.grid.dh),) * 3
    dz_m, dy_m, dx_m = (float(dh_xyz[0]), float(dh_xyz[1]), float(dh_xyz[2]))

    origin = geom.grid_origin_xyz_m
    pad = geom.auto_origin_pad_cells or (0, 0, 0)
    pad_z, pad_y, pad_x = int(pad[0]), int(pad[1]), int(pad[2])
    if origin is None:
        x_min = float(min(src_model_xy[:, 0].min(), rec_model_xy[:, 0].min()))
        y_min = float(min(src_model_xy[:, 1].min(), rec_model_xy[:, 1].min()))
        # z=0 is the sea-surface free-surface datum; never pad above it.
        origin = (0.0, y_min - pad_y * dy_m, x_min - pad_x * dx_m)
    init_origin_z, init_origin_y, init_origin_x = (float(origin[0]), float(origin[1]),
                                                   float(origin[2]))

    nz_pre, ny_pre, nx_pre = init_vp_np.shape
    z_lo, z_hi = 0, nz_pre
    y_lo, y_hi = 0, ny_pre
    x_lo, x_hi = 0, nx_pre
    crop_off = (0.0, 0.0, 0.0)
    if spec.model_plan is not None:
        mp = spec.model_plan
        if mp.z_window_m:
            z_lo = int(np.floor((mp.z_window_m[0] - init_origin_z) / dz_m))
            z_hi = int(np.ceil((mp.z_window_m[1] - init_origin_z) / dz_m)) + 1
        if mp.y_window_m:
            y_lo = int(np.floor((mp.y_window_m[0] - init_origin_y) / dy_m))
            y_hi = int(np.ceil((mp.y_window_m[1] - init_origin_y) / dy_m)) + 1
        if mp.x_window_m:
            x_lo = int(np.floor((mp.x_window_m[0] - init_origin_x) / dx_m))
            x_hi = int(np.ceil((mp.x_window_m[1] - init_origin_x) / dx_m)) + 1
        z_lo, z_hi = max(0, z_lo), min(nz_pre, z_hi)
        y_lo, y_hi = max(0, y_lo), min(ny_pre, y_hi)
        x_lo, x_hi = max(0, x_lo), min(nx_pre, x_hi)
        if z_hi <= z_lo or y_hi <= y_lo or x_hi <= x_lo:
            raise ValueError(f"materialize: empty model_plan crop "
                             f"z[{z_lo}:{z_hi}] y[{y_lo}:{y_hi}] x[{x_lo}:{x_hi}]")
        crop_off = (z_lo * dz_m, y_lo * dy_m, x_lo * dx_m)

    origin_z = init_origin_z + crop_off[0]
    origin_y = init_origin_y + crop_off[1]
    origin_x = init_origin_x + crop_off[2]

    # Crop every inverted/aux model the same way.
    cropped_models: dict = {}
    for m in init_models:
        if m.constant is not None:
            arr = np.full(tuple(m.shape), float(m.constant), dtype=np.float32)
        else:
            arr = np.load(m.path).astype(np.float32)
        cropped_models[m.name] = arr[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].copy()
    init_vp_np = cropped_models["vp"] if "vp" in cropped_models else init_vp_np[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    nz, ny, nx = init_vp_np.shape

    def _project(model_xy, z_raw):
        return np.stack([
            np.rint((model_xy[:, 0] - origin_x) / dx_m).astype(np.int64),
            np.rint((model_xy[:, 1] - origin_y) / dy_m).astype(np.int64),
            np.rint((z_raw - origin_z) / dz_m).astype(np.int64),
        ], axis=-1)

    rec_grid = _project(rec_model_xy, rec_xyz_raw[:, 2])
    src_grid = _project(src_model_xy, src_xyz_raw[:, 2])
    t = _stamp(f"grid projection + crop -> shape (nz,ny,nx)=({nz},{ny},{nx})", t, verbose=verbose)

    # ---- 4) in-window filter (row receiver AND its group source in-grid) ---
    def _in(g):
        return ((g[:, 0] >= 0) & (g[:, 0] < nx)
                & (g[:, 1] >= 0) & (g[:, 1] < ny)
                & (g[:, 2] >= 0) & (g[:, 2] < nz))

    row_in = _in(rec_grid)
    slot_in = _in(src_grid)
    if not (row_in.all() and slot_in.all()):
        row_group = np.repeat(np.arange(plan.n_groups, dtype=np.int64),
                              np.diff(plan.group_offsets))
        row_keep = row_in & slot_in[row_group]
        n_oob_rows = int((~row_in).sum())
        n_oob_groups = int((~slot_in).sum())
        plan = plan.filter_rows(row_keep).drop_empty_groups()
        # Re-derive every plan-indexed array on the renumbered plan.
        src_xyz_raw = np.asarray(plan.group_xyz, dtype=np.float64)
        rec_xyz_raw = (np.asarray(plan.row_source_xyz, dtype=np.float64) if is_crg
                       else np.asarray(plan.row_receiver_xyz, dtype=np.float64))
        rec_model_xy = frame.to_model(rec_xyz_raw[:, :2])
        src_model_xy = frame.to_model(src_xyz_raw[:, :2])
        rec_grid = _project(rec_model_xy, rec_xyz_raw[:, 2])
        src_grid = _project(src_model_xy, src_xyz_raw[:, 2])
        if verbose:
            print(f"[materialize] in-window filter: dropped {n_oob_rows:,} OOB rows + "
                  f"{n_oob_groups} OOB groups -> n_rows={plan.n_rows:,} n_groups={plan.n_groups}",
                  flush=True)
    t = _stamp("in-window filter", t, verbose=verbose)

    # ---- 5) per-group fixed-receiver selection -----------------------------
    nrec = ocfg.n_receivers_per_shot
    rng = np.random.default_rng(int(ocfg.materialize_seed))
    sel_mode = ocfg.receiver_select
    min_rec = int(ocfg.min_receivers)

    group_ids = np.arange(plan.n_groups, dtype=np.int64)
    if ocfg.max_shots is not None and plan.n_groups > int(ocfg.max_shots):
        group_ids = np.sort(rng.choice(plan.n_groups, size=int(ocfg.max_shots),
                                       replace=False))

    sel_rows_per_shot: list[np.ndarray] = []   # absolute plan-row indices
    kept_src_grid: list[np.ndarray] = []
    n_dropped = 0
    for g in group_ids:
        sl = plan.group_slice(int(g))
        rows = np.arange(sl.start, sl.stop, dtype=np.int64)
        if rows.size == 0:
            n_dropped += 1
            continue
        rg = rec_grid[rows]                                # (n_g, 3)
        # Grid-cell dedup: keep one row per unique receiver cell (collapses
        # duplicate quantised positions; for 'nearest' keep the one closest
        # to the cell centre, else keep the first).
        cell_key = (rg[:, 2].astype(np.int64) * (1 << 42)
                    | rg[:, 1].astype(np.int64) * (1 << 21)
                    | rg[:, 0].astype(np.int64))
        if sel_mode == "nearest":
            cx = rg[:, 0] * dx_m + origin_x
            cy = rg[:, 1] * dy_m + origin_y
            cz = rg[:, 2] * dz_m + origin_z
            d2 = ((rec_model_xy[rows, 0] - cx) ** 2
                  + (rec_model_xy[rows, 1] - cy) ** 2
                  + (rec_xyz_raw[rows, 2] - cz) ** 2)
            order = np.lexsort((d2, cell_key))
        else:
            order = np.argsort(cell_key, kind="stable")
        ks = cell_key[order]
        first = np.concatenate([[True], ks[1:] != ks[:-1]])
        uniq_rows = rows[order[first]]
        if uniq_rows.size < max(min_rec, 1):
            n_dropped += 1
            continue
        if nrec is not None:
            if uniq_rows.size < nrec:
                n_dropped += 1
                continue
            if uniq_rows.size > nrec:
                pick = rng.choice(uniq_rows.size, size=int(nrec), replace=False)
                uniq_rows = np.sort(uniq_rows[pick])
        sel_rows_per_shot.append(uniq_rows)
        kept_src_grid.append(src_grid[int(g)])

    if not sel_rows_per_shot:
        raise ValueError("materialize: no shots survived receiver selection "
                         f"(n_receivers_per_shot={nrec}, min_receivers={min_rec}).")
    if nrec is None:
        counts = {a.size for a in sel_rows_per_shot}
        if len(counts) != 1:
            raise ValueError("materialize: receiver_select='all'/n_receivers_per_shot=None "
                             f"requires a uniform receiver count; got {sorted(counts)}. "
                             "Set obs.plan.n_receivers_per_shot.")
        nrec = sel_rows_per_shot[0].size

    nshots = len(sel_rows_per_shot)
    sources = np.stack(kept_src_grid, axis=0).astype(np.int64)             # (nshots, 3)
    receivers = np.stack([rec_grid[r] for r in sel_rows_per_shot], axis=0).astype(np.int64)
    if verbose:
        print(f"[materialize] selected nshots={nshots} x nrec={nrec} "
              f"(dropped {n_dropped} groups; grouping={plan.grouping})", flush=True)
    t = _stamp("receiver selection", t, verbose=verbose)

    # ---- 6) read obs (lazy PlanReader, threaded) ---------------------------
    # Coalesce nearby same-file traces into one pread (huge win over the
    # per-trace random-seek path); a small trace cache enables that path.
    _trace_bytes = int(plan.samples_per_trace) * 4 + 240
    reader = PlanReader(plan, mmap=True, cache_all=bool(ocfg.cache_all),
                        trace_cache_bytes=(512 << 20),
                        coalesce_gap=8 * _trace_bytes)
    native_dt = float(plan.dt_s)
    need_resample = abs(native_dt - float(effective_dt)) > 1.0e-12
    obs = np.empty((nshots, nrec, int(effective_nt)), dtype=np.float32)

    # Global (file_id, offset)-sorted read. CRG gathers scatter each node's
    # rows across many source-line files at non-adjacent offsets, so reading
    # per-gather = pure random seeks (hundreds of traces/s). Sorting ALL
    # selected rows by (file_id, offset) lets the PlanReader coalescer turn
    # them into a near-sequential file sweep (thousands/s), then we scatter
    # each trace back to its (shot, receiver) slot.
    shot_lens = np.array([r.size for r in sel_rows_per_shot], dtype=np.int64)
    all_rows = np.concatenate(sel_rows_per_shot)
    shot_of = np.repeat(np.arange(nshots, dtype=np.int64), shot_lens)
    pos_of = np.concatenate([np.arange(int(n), dtype=np.int64) for n in shot_lens])
    order = np.lexsort((plan.row_trace_offset[all_rows],
                        plan.row_file_id[all_rows]))
    rows_sorted = all_rows[order]
    CHUNK = 200_000
    n_io = max(1, min(32, 2 * (os.cpu_count() or 8)))

    def _read_chunk(c0):
        c1 = min(c0 + CHUNK, rows_sorted.size)
        tr = reader.read_rows(rows_sorted[c0:c1])                # (k, nt_native), sorted
        if need_resample:
            tr = resample_time(tr, native_dt, float(effective_dt), axis=-1)
        cur = tr.shape[-1]
        if cur >= effective_nt:
            tr = tr[:, :effective_nt]
        else:
            tr = np.pad(tr, [(0, 0), (0, int(effective_nt) - cur)])
        idx = order[c0:c1]
        obs[shot_of[idx], pos_of[idx]] = tr.astype(np.float32, copy=False)

    chunk_starts = list(range(0, rows_sorted.size, CHUNK))
    with ThreadPoolExecutor(max_workers=min(n_io, max(1, len(chunk_starts)))) as pool:
        list(pool.map(_read_chunk, chunk_starts))
    t = _stamp(f"read obs ({nshots} gathers, {all_rows.size:,} traces, "
               f"file/offset-sorted, {len(chunk_starts)} chunks)", t, verbose=verbose)

    # ---- 7) source-delay shift (align obs to SIREN-pipeline wavelet) -------
    prepad = int(round(source_delay_s / float(effective_dt))) if source_delay_s > 0 else 0
    if prepad > 0:
        obs = np.roll(obs, shift=prepad, axis=-1)
        obs[..., :prepad] = 0.0
        if verbose:
            print(f"[materialize] source_delay {source_delay_s*1000:.1f} ms -> "
                  f"obs left-shifted {prepad} samples @ dt={effective_dt*1000:.2f} ms",
                  flush=True)

    # ---- 8) canonical layout (nshots, nt, nrec, 1) -------------------------
    obs_canon = np.ascontiguousarray(obs.transpose(0, 2, 1))[..., None]

    # ---- 9) atomic cache write (tmp dir + rename) --------------------------
    try:
        tmp = cache_dir.with_name(cache_dir.name + f".tmp{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "sources.npy", sources)
        np.save(tmp / "receivers.npy", receivers)
        np.save(tmp / "obs.npy", obs_canon)
        for k, v in cropped_models.items():
            np.save(tmp / f"model_{k}.npy", v)
        (tmp / "meta.json").write_text(json.dumps({
            "nshots": int(nshots), "nrec": int(nrec), "shape": [nz, ny, nx],
            "native_dt": float(effective_dt),
            "model_names": list(cropped_models.keys()),
        }))
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if not cache_dir.exists():
            os.rename(tmp, cache_dir)
            if verbose:
                print(f"[materialize] cached -> {cache_dir}", flush=True)
        else:  # another rank won the race
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as err:  # noqa: BLE001
        if verbose:
            print(f"[materialize] cache write failed ({err}); continuing", flush=True)

    return {
        "sources": sources,
        "receivers": receivers,
        "obs": obs_canon,
        "shape": (nz, ny, nx),
        "cropped_models": cropped_models,
        "native_dt": float(effective_dt),  # already resampled to solver dt
        "nshots": nshots,
        "nrec": int(nrec),
    }
