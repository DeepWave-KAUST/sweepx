"""SEG-Y / seismic-plan payload loaders. Verbatim from runner.py."""
import numpy as np
from pathlib import Path


def _load_segy_single_file_payload(
    path: Path,
    *,
    byte_map: dict | None,
    source_depth_m_override: float | None,
    receiver_depth_m_override: float | None,
    coord_scalar_override: float | None,
    shot_ids: list[int] | None = None,
    cache: dict | None = None,
):
    """Open + scan + read a single SEG-Y; cache the result keyed by ``path``.

    Returns a dict with keys ``physical_geometry``, ``obs`` (``(nshots, nrec,
    nt)`` float32), ``shot_ids`` (1-D int64). Both ``ObsSegyConfig`` and
    ``FromSegyGeometry`` flow through this helper, so when both reference the
    same file the runner reads it once.
    """
    from sweep_io.segy_index import (
        build_segy_index,
        IndexedShotGatherDataset,
    )

    key = str(Path(path).resolve())
    if cache is not None and key in cache:
        return cache[key]

    idx = build_segy_index(
        [path],
        byte_map=byte_map,
        source_depth_m_override=source_depth_m_override,
        receiver_depth_m_override=receiver_depth_m_override,
        coord_scalar_override=coord_scalar_override,
        num_workers=1,
    )
    sids = (
        np.asarray(shot_ids, dtype=np.int64)
        if shot_ids is not None else np.asarray(idx.shot_ids, dtype=np.int64)
    )
    # Materialise obs eagerly (single-file path -> simple).
    ds = IndexedShotGatherDataset(idx, shot_ids=sids)
    obs = np.stack([ds[i]["obs"] for i in range(len(ds))], axis=0)   # (nshots, nrec, nt)
    ds.close()

    pg = idx.to_physical_geometry(shot_ids=sids)
    result = {
        "physical_geometry": pg,
        "obs": obs,
        "shot_ids": sids,
        "n_samples": idx.n_samples,
        "dt_s": idx.dt_s,
    }
    if cache is not None:
        cache[key] = result
    return result


def _load_seismic_plan_payload(
    plan_path: Path,
    *,
    cache_all: bool = False,
    grid_ndim: int | None = None,
    cache: dict | None = None,
    geometry_only: bool = False,
):
    """Load a ``seismic_plan_v1`` cache as a payload dict (mirrors
    :func:`_load_segy_single_file_payload`).

    This helper serves the static-obs CSG path (``grouping="csg"`` only,
    every group's row count must be uniform — typical for rigid
    streamers). CRG-grouped plans never come through here; they take one of
    two other routes, both via ``obs.plan``:

    * ``obs.plan.sampling`` set → the per-iter plan-streaming sampler in
      :meth:`TaskRunner._run_fwi_plan_streaming`, which calls
      :func:`sweep_io.seismic_plan.sample_shared_shots_from_plan` +
      :meth:`PlanReader.read_rows` directly.
    * ``obs.plan.sampling`` unset → :func:`plan_materialize.
      materialize_plan_dataset`, which builds a static reciprocal dataset
      (node = virtual source, its air-gun positions = receivers).

    ``grid_ndim``: 2 → keep (x, z) only (drop y); 3 → keep (x, y, z).
    ``geometry_only``: when True, skip the eager ``PlanReader.read_all`` so
    the geometry resolver doesn't pull every trace into RAM (RTM's per-batch
    PlanReader loop will do its own reads later). The cache entry under
    ``geometry_only=True`` and ``geometry_only=False`` are distinct so a
    subsequent eager-obs lookup still triggers a full load.
    """
    from sweep_io.seismic_plan import SeismicPlan, PlanReader
    from sweep_io.geometry import PhysicalGeometry

    plan_path = Path(plan_path)
    key = ("seismic_plan", str(plan_path.resolve()), bool(cache_all),
           int(grid_ndim) if grid_ndim else None, bool(geometry_only))
    if cache is not None and key in cache:
        return cache[key]

    plan = SeismicPlan.load(plan_path)
    if plan.grouping != "csg":
        raise ValueError(
            f"_load_seismic_plan_payload (CSG static-obs path) requires "
            f"grouping='csg' (got {plan.grouping!r}). This route resolves "
            "GEOMETRY from the plan while obs comes from elsewhere, so it "
            "has no CRG reciprocity. A CRG plan has two supported routes, "
            "both driven by obs.plan pointing at the same plan: set "
            "obs.plan.sampling (PlanSamplingConfig) for the per-iter "
            "supershot pipeline (_run_fwi_plan_streaming), or leave sampling "
            "unset for a static reciprocal dataset (plan_materialize: node "
            "= virtual source, its air-gun positions = receivers)."
        )
    counts = plan.per_group_row_counts()
    if counts.size == 0:
        raise ValueError("from_plan: SeismicPlan has zero groups")
    if not (counts == counts[0]).all():
        raise ValueError(
            f"from_plan: non-uniform receiver count per shot "
            f"({int(counts.min())}..{int(counts.max())}). The runner's "
            "per-stage dedupe path needs a uniform pristine layout; rebuild "
            "the plan with a filter that yields constant nrec or use "
            "from_segy_headers (which has its own per-shot dedupe)."
        )
    nshots = int(plan.n_groups)
    nrec = int(counts[0])
    nt = int(plan.samples_per_trace)

    # row arrays are sorted by group already → straight reshape works.
    src_xyz = np.asarray(plan.group_xyz, dtype=np.float64)               # (nshots, 3)
    rec_xyz = np.asarray(plan.row_receiver_xyz, dtype=np.float64).reshape(nshots, nrec, 3)

    if grid_ndim == 2:
        # 2-D streamer: keep (x, z) only.
        src_xyz = src_xyz[:, [0, 2]]
        rec_xyz = rec_xyz[:, :, [0, 2]]
    elif grid_ndim is not None and grid_ndim != 3:
        raise ValueError(
            f"from_plan: grid_ndim must be 2 or 3, got {grid_ndim}"
        )

    pg = PhysicalGeometry(
        sources_xyz_m=src_xyz,
        receivers_xyz_m=rec_xyz,
        dt=float(plan.dt_s), nt=nt,
        meta={"source": "SeismicPlan", "plan_path": str(plan_path),
              "plan_grouping": plan.grouping,
              "plan_build_meta": dict(plan.build_meta)},
    )

    if geometry_only:
        # `_segy_geometry_to_grid_indices` inspects obs.shape[1] to decide
        # whether a receiver mask is needed; supply a zero-cost placeholder
        # with the right receiver-axis size so the geometry path runs without
        # materialising actual samples. ``_build_geometry_2d`` drops the
        # returned obs immediately.
        obs = np.zeros((nshots, nrec, 1), dtype=np.float32)
    else:
        # Read obs into a single (nshots, nrec, nt) f32 tensor via PlanReader.
        reader = PlanReader(plan, cache_all=cache_all)
        try:
            obs_flat = reader.read_all().astype(np.float32, copy=False)  # (n_rows, nt)
        finally:
            reader.close()
        obs = obs_flat.reshape(nshots, nrec, nt)

    result = {
        "physical_geometry": pg,
        "obs": obs,
        "shot_ids": np.asarray(plan.group_id, dtype=np.int64),
        "n_samples": nt,
        "dt_s": float(plan.dt_s),
    }
    if cache is not None:
        cache[key] = result
    return result


def _load_segy_index_payload(
    index_path: Path,
    *,
    shot_ids: list[int] | None = None,
    cache: dict | None = None,
    lazy: bool = False,
    coalesce_gap: int = 0,
):
    """Load a pre-built SEGYIndex and materialise per-shot obs.

    ``lazy=False`` (default): read every shot eagerly, in input order, into
    a single ``(nshots, nrec, nt)`` tensor.

    ``lazy=True``: same final layout, but each shot is read by a background
    :class:`sweep_io.prefetch.Prefetcher` worker — useful when SEG-Y access
    is slow (network mount, cold cache) so the GIL-releasing I/O overlaps
    with whatever the consumer is doing at task-start. True per-step
    streaming during training is a future task: the dataset object is built
    but the materialised tensor is still what the train loop indexes.
    """
    from sweep_io.segy_index import SEGYIndex, IndexedShotGatherDataset
    from sweep_io.prefetch import Prefetcher

    key = (str(Path(index_path).resolve()), tuple(shot_ids) if shot_ids else None,
           bool(lazy), int(coalesce_gap))
    if cache is not None and key in cache:
        return cache[key]

    idx = SEGYIndex.load(index_path)
    sids = (
        np.asarray(shot_ids, dtype=np.int64)
        if shot_ids is not None else np.asarray(idx.shot_ids, dtype=np.int64)
    )

    ds = IndexedShotGatherDataset(idx, shot_ids=sids, coalesce_gap=coalesce_gap)
    if lazy:
        # Background-thread reader; results stay in input order.
        with Prefetcher((ds[i]["obs"] for i in range(len(ds))), queue_depth=2) as pf:
            obs = np.stack(list(pf), axis=0)
    else:
        obs = np.stack([ds[i]["obs"] for i in range(len(ds))], axis=0)
    ds.close()

    pg = idx.to_physical_geometry(shot_ids=sids)
    result = {
        "physical_geometry": pg,
        "obs": obs,
        "shot_ids": sids,
        "n_samples": idx.n_samples,
        "dt_s": idx.dt_s,
    }
    if cache is not None:
        cache[key] = result
    return result
