"""Acquisition-geometry construction (line/explicit/from-file/plan). Verbatim from runner.py."""
import numpy as np

from sweep_tasks.schemas import LineSet
from sweep_tasks._helpers.data_loading import (
    _load_segy_index_payload,
    _load_segy_single_file_payload,
    _load_seismic_plan_payload,
)

def _line_array(line: LineSet, fallback_stop: int) -> "np.ndarray":
    stop = line.stop if line.stop is not None else fallback_stop
    if stop <= line.start:
        raise ValueError(
            f"LineSet stop ({stop}) must be greater than start ({line.start})."
        )
    xs = np.arange(line.start, stop, line.step, dtype=np.int64).reshape(-1, 1)
    zs = np.full_like(xs, line.depth)
    return np.concatenate([xs, zs], axis=1)


def _build_geometry(
    geometry,
    shape: tuple[int, ...],
    *,
    dh: float | None = None,
    segy_cache: dict | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Dispatch on geometry.kind and return (sources, receivers) numpy arrays.

    Output shapes are always sources=(nshots, ndim) and receivers=(nshots, nrec, ndim).
    ``ndim`` is 2 when ``shape`` is ``(nz, nx)`` and 3 when ``shape`` is
    ``(nz, ny, nx)``. ``explicit`` / ``from_file`` accept either; ``line``
    is 2-D-only; the SEG-Y kinds are 2-D-only because their header-derived
    geometry is built around ``(x, z)`` only.

    ``dh`` and ``segy_cache`` are only used by the SEG-Y-backed kinds
    (``from_segy_headers`` / ``from_segy_index``). The cache lets the runner
    share a single SEG-Y scan between the geometry resolver and the obs
    loader when both point at the same file / same index.
    """

    kind = getattr(geometry, "kind", None)
    grid_ndim = len(shape)
    if kind == "line":
        if grid_ndim != 2:
            raise ValueError(
                f"LineGeometry only supports 2-D grids (shape ndim=2); "
                f"got shape={shape}. Use ExplicitGeometry / FromFileGeometry "
                f"for 3-D."
            )
        nx = int(shape[-1])
        sources = _line_array(geometry.sources, nx)
        rec = _line_array(geometry.receivers, nx)
        receivers = rec[None, ...].repeat(sources.shape[0], axis=0)
        return sources, receivers
    if kind == "explicit":
        sources, receivers = _explicit_geometry_arrays(geometry)
        if sources.shape[1] != grid_ndim:
            raise ValueError(
                f"ExplicitGeometry ndim={sources.shape[1]} does not match grid "
                f"ndim={grid_ndim} (shape={shape})."
            )
        return sources, receivers
    if kind == "from_file":
        sources, receivers = _from_file_geometry_arrays(geometry)
        if sources.shape[1] != grid_ndim:
            raise ValueError(
                f"FromFileGeometry ndim={sources.shape[1]} does not match grid "
                f"ndim={grid_ndim} (shape={shape})."
            )
        return sources, receivers
    if kind == "from_segy_headers":
        if grid_ndim != 2:
            raise ValueError(
                f"from_segy_headers geometry is 2-D-only; got grid shape={shape}. "
                f"For 3-D OBN data use the CRG-plan geometry adapter instead."
            )
        if dh is None:
            raise ValueError("from_segy_headers requires `dh` (spec.grid.dh).")
        payload = _load_segy_single_file_payload(
            geometry.path,
            byte_map=geometry.byte_map,
            source_depth_m_override=geometry.source_depth_m_override,
            receiver_depth_m_override=geometry.receiver_depth_m_override,
            coord_scalar_override=geometry.coord_scalar_override,
            cache=segy_cache,
        )
        # Initial parse always keeps ALL receivers (dedupe=False internally).
        # Per-stage dedupe (when ``geometry.dedupe=True``) happens inside
        # ``_prepare_stage`` against the pristine 120-receiver geometry, so
        # finer stages can recover the full receiver count even if a coarser
        # stage drops some.
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=False, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    if kind == "from_segy_index":
        if grid_ndim != 2:
            raise ValueError(
                f"from_segy_index geometry is 2-D-only; got grid shape={shape}."
            )
        if dh is None:
            raise ValueError("from_segy_index requires `dh` (spec.grid.dh).")
        payload = _load_segy_index_payload(
            geometry.index_path, shot_ids=geometry.shot_ids,
            cache=segy_cache,
        )
        # Initial parse always keeps ALL receivers (dedupe=False internally).
        # Per-stage dedupe (when ``geometry.dedupe=True``) happens inside
        # ``_prepare_stage`` against the pristine 120-receiver geometry, so
        # finer stages can recover the full receiver count even if a coarser
        # stage drops some.
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=False, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    if kind == "from_plan":
        if dh is None:
            raise ValueError("from_plan requires `dh` (spec.grid.dh).")
        if grid_ndim not in (2, 3):
            raise ValueError(
                f"from_plan: grid ndim must be 2 or 3 (shape={shape})."
            )
        payload = _load_seismic_plan_payload(
            geometry.plan_path,
            cache_all=False,
            grid_ndim=grid_ndim,
            cache=segy_cache,
            geometry_only=True,        # don't read traces just to compute geometry
        )
        sources_idx, receivers_idx, _obs_aligned = _segy_geometry_to_grid_indices(
            payload, float(dh),
            dedupe=False, dedup_method=geometry.dedup_method,
        )
        return sources_idx, receivers_idx
    raise ValueError(f"Unknown geometry.kind '{kind}'.")


# Back-compat alias: callers that imported the old 2-D-only name keep working.
# The new dispatcher handles both 2-D and 3-D based on the ``shape`` arg.
_build_geometry_2d = _build_geometry


def _explicit_geometry_arrays(geometry) -> tuple["np.ndarray", "np.ndarray"]:
    sources = np.asarray(geometry.sources, dtype=np.int64)
    rec_raw = geometry.receivers
    first = rec_raw[0]
    is_per_shot = bool(first) and isinstance(first[0], list)
    if is_per_shot:
        receivers = np.asarray(rec_raw, dtype=np.int64)
        if receivers.shape[0] != sources.shape[0]:
            raise ValueError(
                f"Explicit per-shot receivers shape {receivers.shape} first axis must equal "
                f"nshots {sources.shape[0]}."
            )
    else:
        receivers_2d = np.asarray(rec_raw, dtype=np.int64)
        receivers = receivers_2d[None, ...].repeat(sources.shape[0], axis=0)
    return sources, receivers


def _segy_geometry_to_grid_indices(payload, dh: float, *, dedupe: bool, dedup_method: str):
    """Snap a SEG-Y-derived PhysicalGeometry to integer grid indices.

    Returns ``(sources_idx, receivers_idx, obs_aligned)`` — all rectangular
    ``(nshots, ...)`` tensors. Two modes:

    * **Uniform-mask path** (``dedupe=False`` or the dedup mask is identical
      across all shots): obs is sliced once with the shared keep_idx and the
      receivers_idx is returned post-slice. This is the fast path used by
      symmetric layouts (a stationary land array, streamer where source
      offsets are integer multiples of dh, etc.).

    * **Non-uniform-mask path** (``dedupe=True`` and shots disagree on which
      receivers survive — typical for streamer acquisition where the source
      shift per shot is not a multiple of dh): build per-shot keep_idx
      arrays, truncate to the minimum keep-count, and fancy-index obs
      per shot to a rectangular ``(nshots, n_common, nt)`` tensor.
    """
    pg = payload["physical_geometry"]
    obs = payload["obs"]
    gg, mask = pg.to_grid(dh=(dh, dh), dedupe=dedupe, dedup_method=dedup_method)
    uniform = bool(np.all(mask == mask[0:1]))
    if not dedupe or uniform:
        keep = np.flatnonzero(mask[0])
        if keep.size != obs.shape[1]:
            obs = np.ascontiguousarray(obs[:, keep, :])
            receivers_idx = gg.receivers[:, keep, :]
        else:
            receivers_idx = gg.receivers
        return gg.sources.astype(np.int64), receivers_idx.astype(np.int64), obs

    # Non-uniform path: per-shot keep_idx, truncate to common min.
    per_shot_keep = [np.flatnonzero(mask[s]) for s in range(mask.shape[0])]
    counts = np.asarray([k.size for k in per_shot_keep])
    n_common = int(counts.min())
    if n_common <= 0:
        raise ValueError("dedupe removed all receivers in at least one shot")
    truncated = np.stack(
        [k[:n_common].astype(np.int64) for k in per_shot_keep], axis=0
    )
    nshots = mask.shape[0]
    rec_out = np.empty((nshots, n_common, gg.receivers.shape[-1]), dtype=np.int64)
    obs_out = np.empty((nshots, n_common, obs.shape[-1]), dtype=obs.dtype)
    for s in range(nshots):
        rec_out[s] = gg.receivers[s, truncated[s], :].astype(np.int64)
        obs_out[s] = obs[s, truncated[s], :]
    return gg.sources.astype(np.int64), rec_out, np.ascontiguousarray(obs_out)


def _from_file_geometry_arrays(geometry) -> tuple["np.ndarray", "np.ndarray"]:
    sources = np.load(geometry.sources_file).astype(np.int64)
    receivers = np.load(geometry.receivers_file).astype(np.int64)
    if sources.ndim != 2:
        raise ValueError(
            f"sources_file array must be 2D (nshots, ndim), got shape {sources.shape}."
        )
    if receivers.ndim == 2:
        receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)
    elif receivers.ndim == 3:
        if receivers.shape[0] != sources.shape[0]:
            raise ValueError(
                f"receivers_file first axis {receivers.shape[0]} must equal nshots "
                f"{sources.shape[0]}."
            )
    else:
        raise ValueError(
            f"receivers_file array must be 2D or 3D, got shape {receivers.shape}."
        )
    return sources, receivers
