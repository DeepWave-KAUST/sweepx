"""Acquisition geometry: source / receiver positions and sampling parameters.

Two flavours coexist:

- :class:`Geometry` — positions stored as **grid indices**, tied to a
  specific ``dh``. This is what the ``sweep`` propagators consume.
- :class:`PhysicalGeometry` — positions in **meters**, grid-agnostic.
  Call :meth:`PhysicalGeometry.to_grid` to snap to a concrete ``dh``
  per FWI stage (with optional dedupe when multiple traces collide on
  the same grid cell).

The physical form is what you should hold across a multi-scale FWI run:
each stage may use a different ``dh``, so the grid form is rebuilt per
stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np


@dataclass
class Geometry:
    """Source / receiver acquisition geometry.

    Attributes
    ----------
    sources
        ``(nshots, ndim)`` grid indices. ``ndim=2`` for 2-D ``(x, z)``,
        ``ndim=3`` for 3-D ``(x, y, z)``.
    receivers
        ``(nshots, nreceivers, ndim)`` grid indices.
    dt
        Time sampling interval (s). Optional.
    nt
        Number of time samples. Optional.
    dh
        Spatial sampling per axis (same length as ``ndim``). Optional.
    meta
        Arbitrary user metadata that gets serialized verbatim.
    """

    sources: np.ndarray
    receivers: np.ndarray
    dt: float | None = None
    nt: int | None = None
    dh: Tuple[float, ...] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sources = np.asarray(self.sources)
        self.receivers = np.asarray(self.receivers)
        if self.sources.ndim != 2:
            raise ValueError(
                f"`sources` must be (nshots, ndim); got shape {self.sources.shape}"
            )
        if self.receivers.ndim != 3:
            raise ValueError(
                f"`receivers` must be (nshots, nreceivers, ndim); "
                f"got shape {self.receivers.shape}"
            )
        if self.sources.shape[0] != self.receivers.shape[0]:
            raise ValueError(
                f"nshots mismatch: sources has {self.sources.shape[0]}, "
                f"receivers has {self.receivers.shape[0]}"
            )
        if self.sources.shape[1] != self.receivers.shape[2]:
            raise ValueError(
                f"ndim mismatch: sources has ndim={self.sources.shape[1]}, "
                f"receivers has ndim={self.receivers.shape[2]}"
            )
        if self.dh is not None and len(self.dh) != self.sources.shape[1]:
            raise ValueError(
                f"`dh` must have length ndim={self.sources.shape[1]}; got {self.dh}"
            )

    # ------------------------------------------------------------------ derived
    @property
    def nshots(self) -> int:
        return int(self.sources.shape[0])

    @property
    def nreceivers(self) -> int:
        return int(self.receivers.shape[1])

    @property
    def ndim(self) -> int:
        return int(self.sources.shape[1])

    # ------------------------------------------------------------------ serdes
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = self.sources.tolist()
        d["receivers"] = self.receivers.tolist()
        if self.dh is not None:
            d["dh"] = list(self.dh)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Geometry":
        return cls(
            sources=np.asarray(d["sources"]),
            receivers=np.asarray(d["receivers"]),
            dt=d.get("dt"),
            nt=d.get("nt"),
            dh=tuple(d["dh"]) if d.get("dh") is not None else None,
            meta=d.get("meta", {}) or {},
        )

    def save(self, path: str | Path) -> None:
        """Save as JSON (always) or .npz (binary, more compact)."""
        path = Path(path)
        if path.suffix.lower() == ".npz":
            np.savez(
                path,
                sources=self.sources,
                receivers=self.receivers,
                _meta=json.dumps(
                    dict(dt=self.dt, nt=self.nt, dh=self.dh, meta=self.meta)
                ),
            )
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Geometry":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as f:
                meta = json.loads(str(f["_meta"]))
                return cls(
                    sources=f["sources"],
                    receivers=f["receivers"],
                    dt=meta.get("dt"),
                    nt=meta.get("nt"),
                    dh=tuple(meta["dh"]) if meta.get("dh") is not None else None,
                    meta=meta.get("meta", {}) or {},
                )
        return cls.from_dict(json.loads(path.read_text()))


# ============================================================================
# Physical-units geometry (grid-agnostic)
# ============================================================================
@dataclass
class PhysicalGeometry:
    """Source/receiver positions in **meters** — independent of any grid.

    For multi-scale FWI the same geometry is reused across many ``dh``;
    holding the physical form and calling :meth:`to_grid` per stage is the
    canonical pattern.

    Attributes
    ----------
    sources_xyz_m
        ``(nshots, ndim)`` source coordinates in meters. ``ndim=2`` is
        ``(x, z)``, ``ndim=3`` is ``(x, y, z)``.
    receivers_xyz_m
        ``(nshots, nreceivers, ndim)`` receiver coordinates in meters.
    dt
        Time sampling interval (s).
    nt
        Number of time samples.
    meta
        Optional metadata dict.
    """

    sources_xyz_m: np.ndarray
    receivers_xyz_m: np.ndarray
    dt: float
    nt: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sources_xyz_m = np.asarray(self.sources_xyz_m, dtype="float64")
        self.receivers_xyz_m = np.asarray(self.receivers_xyz_m, dtype="float64")
        if self.sources_xyz_m.ndim != 2:
            raise ValueError(
                f"`sources_xyz_m` must be (nshots, ndim); got shape {self.sources_xyz_m.shape}"
            )
        if self.receivers_xyz_m.ndim != 3:
            raise ValueError(
                f"`receivers_xyz_m` must be (nshots, nreceivers, ndim); "
                f"got shape {self.receivers_xyz_m.shape}"
            )
        if self.sources_xyz_m.shape[0] != self.receivers_xyz_m.shape[0]:
            raise ValueError(
                f"nshots mismatch: sources has {self.sources_xyz_m.shape[0]}, "
                f"receivers has {self.receivers_xyz_m.shape[0]}"
            )
        if self.sources_xyz_m.shape[1] != self.receivers_xyz_m.shape[2]:
            raise ValueError(
                f"ndim mismatch: sources has ndim={self.sources_xyz_m.shape[1]}, "
                f"receivers has ndim={self.receivers_xyz_m.shape[2]}"
            )

    # ------------------------------------------------------------------ derived
    @property
    def nshots(self) -> int:
        return int(self.sources_xyz_m.shape[0])

    @property
    def nreceivers(self) -> int:
        return int(self.receivers_xyz_m.shape[1])

    @property
    def ndim(self) -> int:
        return int(self.sources_xyz_m.shape[1])

    # ------------------------------------------------------------- main API
    def to_grid(
        self,
        dh: Tuple[float, ...] | float,
        *,
        origin_xyz_m: Tuple[float, ...] | None = None,
        dedupe: bool = True,
        dedup_method: Literal["nearest", "first"] = "nearest",
    ) -> Tuple["Geometry", np.ndarray]:
        """Snap to nearest grid cell for the given ``dh``.

        Parameters
        ----------
        dh
            Grid spacing in meters. Scalar broadcasts to all axes; otherwise
            a tuple of length ``ndim``.
        origin_xyz_m
            Coordinate origin (the position of grid index ``0``). Defaults
            to zero on every axis.
        dedupe
            When two or more receivers in the same shot snap to the same
            grid cell, drop the redundant ones. Sources are not deduped
            (each source already targets a distinct shot).
        dedup_method
            ``"nearest"`` — keep the receiver whose true position is
            closest to the grid-cell center.
            ``"first"`` — keep the first receiver in input order.

        Returns
        -------
        geom : Geometry
            Grid-index geometry with ``dh`` populated.
        keep_mask : np.ndarray
            Boolean array of shape ``(nshots, nreceivers)``. ``True`` where
            the receiver survived dedupe. Use this to slice observed-data
            tensors along the receiver axis so they line up with the
            surviving receivers per shot::

                gg, mask = pg.to_grid(dh=(25.0, 25.0))
                obs_aligned = [obs[s][..., mask[s], :] for s in range(pg.nshots)]
        """
        if np.isscalar(dh):
            dh_arr = np.full(self.ndim, float(dh), dtype="float64")
        else:
            dh_arr = np.asarray(dh, dtype="float64")
            if dh_arr.shape != (self.ndim,):
                raise ValueError(
                    f"`dh` must be a scalar or length-{self.ndim} sequence; got {dh}"
                )
        if np.any(dh_arr <= 0):
            raise ValueError(f"all `dh` entries must be > 0; got {dh}")
        origin = (
            np.zeros(self.ndim, dtype="float64")
            if origin_xyz_m is None
            else np.asarray(origin_xyz_m, dtype="float64")
        )
        if origin.shape != (self.ndim,):
            raise ValueError(
                f"`origin_xyz_m` must have length {self.ndim}; got {origin_xyz_m}"
            )

        sources_idx = np.round(
            (self.sources_xyz_m - origin) / dh_arr
        ).astype(np.int64)
        receivers_idx = np.round(
            (self.receivers_xyz_m - origin) / dh_arr
        ).astype(np.int64)

        keep_mask = np.ones((self.nshots, self.nreceivers), dtype=bool)
        if dedupe:
            for s in range(self.nshots):
                seen: dict[tuple, tuple[int, float]] = {}
                for i in range(self.nreceivers):
                    key = tuple(int(v) for v in receivers_idx[s, i])
                    cell_center = origin + receivers_idx[s, i].astype("float64") * dh_arr
                    d = float(np.linalg.norm(self.receivers_xyz_m[s, i] - cell_center))
                    if key not in seen:
                        seen[key] = (i, d)
                    else:
                        prev_i, prev_d = seen[key]
                        if dedup_method == "nearest" and d < prev_d:
                            keep_mask[s, prev_i] = False
                            seen[key] = (i, d)
                        else:  # "first" or "nearest" but new is farther
                            keep_mask[s, i] = False

        geom = Geometry(
            sources=sources_idx,
            receivers=receivers_idx,
            dt=self.dt,
            nt=self.nt,
            dh=tuple(float(v) for v in dh_arr),
            meta=dict(self.meta),
        )
        return geom, keep_mask
