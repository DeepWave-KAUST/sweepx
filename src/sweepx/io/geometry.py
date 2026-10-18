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

For surveys whose physical coordinates live in a rotated frame (e.g. OBN
acquisitions in UTM that must be projected onto a model-aligned XY frame
before the propagator's axis-aligned grid is built), use
:class:`RotatedFrame` + :meth:`PhysicalGeometry.apply_rotation`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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

    # --------------------------------------------------------- rotated frame
    def apply_rotation(self, frame: "RotatedFrame") -> "PhysicalGeometry":
        """Return a copy with the horizontal ``(x, y)`` coordinates rotated
        from UTM into the model frame defined by ``frame``.

        Only valid for ``ndim == 3``. The vertical ``z`` coordinate is
        passed through unchanged. The returned geometry shares ``dt`` /
        ``nt`` and carries a ``meta['rotation']`` payload recording the
        frame parameters so downstream consumers can round-trip back to
        UTM via :meth:`RotatedFrame.to_utm`.
        """
        if self.ndim != 3:
            raise ValueError(
                f"apply_rotation requires a 3-D PhysicalGeometry "
                f"(got ndim={self.ndim}). For 2-D geometry the rotation "
                "is a no-op or should be applied before instantiation."
            )
        src = np.asarray(self.sources_xyz_m, dtype="float64")
        rec = np.asarray(self.receivers_xyz_m, dtype="float64")
        src_xy = frame.to_model(src[..., :2])
        rec_xy = frame.to_model(rec[..., :2].reshape(-1, 2)).reshape(rec.shape[:-1] + (2,))
        src_out = np.concatenate([src_xy, src[..., 2:3]], axis=-1)
        rec_out = np.concatenate([rec_xy, rec[..., 2:3]], axis=-1)
        new_meta = dict(self.meta)
        new_meta["rotation"] = frame.to_dict()
        return PhysicalGeometry(
            sources_xyz_m=src_out,
            receivers_xyz_m=rec_out,
            dt=self.dt,
            nt=self.nt,
            meta=new_meta,
        )


# ============================================================================
# Line-azimuth estimation (PCA) — derive a rotation from scattered points
# ============================================================================
def _normalize_angle_rad(angle: float) -> float:
    """Normalize an angle to the half-open interval ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def principal_direction(points_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate the dominant 2-D line direction with a two-component PCA.

    Fits the principal axis of a scattered ``(x, y)`` point cloud (e.g. the
    shot positions of a 2-D streamer line, or the node positions of an OBN
    receiver line) via SVD and returns its azimuth.

    Parameters
    ----------
    points_xy
        ``(npoints, 2)`` coordinates. At least two distinct points required.

    Returns
    -------
    direction : np.ndarray
        ``(2,)`` unit vector along the dominant line direction.
    angle : float
        ``arctan2(dy, dx)`` of ``direction`` in radians.

    Notes
    -----
    The SVD sign of the leading singular vector is arbitrary, so the
    returned direction may point either way along the line (the azimuth is
    only defined modulo π). This mirrors
    ``fwi_workflow.geometry.acquisition.principal_direction`` exactly, so a
    frame fit here reproduces the legacy rotation bit-for-bit.
    """
    points = np.asarray(points_xy, dtype="float64")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"points_xy must have shape (npoints, 2); got {points.shape}"
        )
    if len(points) < 2:
        raise ValueError(
            "at least two points are required to estimate a line direction"
        )
    centered = points - np.mean(points, axis=0)
    if np.allclose(centered, 0.0):
        raise ValueError("cannot estimate a line direction from identical points")
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = np.asarray(vh[0], dtype="float64")
    direction /= np.linalg.norm(direction)
    angle = float(np.arctan2(direction[1], direction[0]))
    return direction, angle


# ============================================================================
# RotatedFrame — 2-D rotation around an origin in the horizontal plane
# ============================================================================
@dataclass(frozen=True)
class RotatedFrame:
    """UTM ↔ model-frame 2-D rotation.

    Maps ``model_xy = (utm_xy - origin_xy_utm) @ R^T + shifts`` where
    ``R`` is a 2×2 rotation matrix. The model frame is what the
    propagator's axis-aligned grid consumes; the UTM frame is what
    field-data SEG-Y headers (and the OBN survey metadata) live in.

    Sign convention matches the legacy ``rotation_metadata.json`` schema
    produced by ``fwi_workflow-dev``: ``rotation_matrix`` is the matrix
    that rotates a UTM displacement vector (relative to ``origin_xy_utm``)
    onto the model-frame axes. ``target_axis="x"`` means the inline
    azimuth points along model ``+x``; ``"y"`` means inline points along
    model ``+y``.

    Parameters
    ----------
    origin_xy_utm
        ``(2,)`` UTM origin that maps to model-frame ``(0, 0)`` before
        any inline / crossline shift.
    rotation_matrix
        ``(2, 2)`` rotation matrix. If only a rotation angle is known,
        construct with :meth:`from_rotation_deg`.
    target_axis
        ``"x"`` (default) or ``"y"`` — which model axis is the inline.
    inline_shift, crossline_shift
        Optional rigid offsets along the rotated inline / crossline axes
        (meters). Default ``0.0``.
    """

    origin_xy_utm: np.ndarray
    rotation_matrix: np.ndarray
    target_axis: Literal["x", "y"] = "x"
    inline_shift: float = 0.0
    crossline_shift: float = 0.0

    def __post_init__(self) -> None:
        # Promote inputs to canonical numpy form. Frozen dataclass means we
        # have to use object.__setattr__ to mutate the fields.
        origin = np.asarray(self.origin_xy_utm, dtype="float64")
        if origin.shape != (2,):
            raise ValueError(
                f"origin_xy_utm must have shape (2,); got {origin.shape}"
            )
        rot = np.asarray(self.rotation_matrix, dtype="float64")
        if rot.shape != (2, 2):
            raise ValueError(
                f"rotation_matrix must have shape (2, 2); got {rot.shape}"
            )
        if self.target_axis not in ("x", "y"):
            raise ValueError(
                f"target_axis must be 'x' or 'y'; got {self.target_axis!r}"
            )
        object.__setattr__(self, "origin_xy_utm", origin)
        object.__setattr__(self, "rotation_matrix", rot)

    # --------------------------------------------------------- constructors
    @classmethod
    def from_rotation_deg(
        cls,
        *,
        origin_xy_utm: np.ndarray | Tuple[float, float],
        rotation_deg: float,
        target_axis: Literal["x", "y"] = "x",
        inline_shift: float = 0.0,
        crossline_shift: float = 0.0,
    ) -> "RotatedFrame":
        """Build a frame from a single rotation angle (degrees, CCW)."""
        theta = float(np.deg2rad(rotation_deg))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]], dtype="float64")
        return cls(
            origin_xy_utm=np.asarray(origin_xy_utm, dtype="float64"),
            rotation_matrix=R,
            target_axis=target_axis,
            inline_shift=float(inline_shift),
            crossline_shift=float(crossline_shift),
        )

    @classmethod
    def from_metadata(
        cls, metadata: str | Path | Mapping[str, object],
    ) -> "RotatedFrame":
        """Build a frame from a ``rotation_metadata.json`` path or dict.

        Compatible with the legacy ``fwi_workflow-dev`` schema:
        ``origin_xy``, ``rotation_matrix``, optional ``inline_shift``,
        ``crossline_shift``, ``target_axis``.

        Missing shifts default to ``0.0``; missing ``target_axis``
        defaults to ``"x"``.
        """
        if isinstance(metadata, Mapping):
            meta = dict(metadata)
        else:
            path = Path(metadata)
            with path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        if "origin_xy" not in meta or "rotation_matrix" not in meta:
            raise KeyError(
                "rotation metadata must contain 'origin_xy' and "
                "'rotation_matrix' fields."
            )
        return cls(
            origin_xy_utm=np.asarray(meta["origin_xy"], dtype="float64"),
            rotation_matrix=np.asarray(meta["rotation_matrix"], dtype="float64"),
            target_axis=str(meta.get("target_axis", "x")),
            inline_shift=float(meta.get("inline_shift", 0.0)),
            crossline_shift=float(meta.get("crossline_shift", 0.0)),
        )

    @classmethod
    def fit(
        cls,
        points_xy: np.ndarray,
        *,
        target_axis: Literal["x", "y"] = "x",
        shift_inline_to_zero: bool = False,
        shift_crossline_to_zero: bool = False,
        shift_points_xy: np.ndarray | None = None,
    ) -> "RotatedFrame":
        """Derive a frame from scattered UTM points via a 2-D PCA line fit.

        This is the companion *constructor* to :meth:`from_rotation_deg` /
        :meth:`from_metadata`: those **consume** a known rotation, while
        ``fit`` **derives** one from the acquisition geometry. It estimates
        the dominant survey-line azimuth from ``points_xy`` with
        :func:`principal_direction` and builds the frame that rotates that
        azimuth onto the model ``target_axis``.

        Ports the rotation estimation of
        ``fwi_workflow.geometry.line2d.rotate_csg_index``: identical SVD
        principal direction, ``origin = mean(points)``, and
        ``[[cos, -sin], [sin, cos]]`` matrix convention. The resulting frame
        therefore serialises (via :meth:`to_dict`) to the same
        ``rotation_metadata.json`` schema and reproduces the legacy
        inline / crossline coordinates bit-for-bit.

        Parameters
        ----------
        points_xy
            ``(npoints, 2)`` UTM xy used to estimate the line azimuth (e.g.
            one point per shot for a 2-D streamer line).
        target_axis
            Model axis the inline direction is rotated onto: ``"x"``
            (default, 2-D streamer convention) or ``"y"``.
        shift_inline_to_zero
            Shift the inline coordinate so its minimum is ``0`` (line starts
            at the grid edge). Matches the legacy 2-D streamer default.
        shift_crossline_to_zero
            Shift the crossline coordinate so its minimum is ``0``. Combine
            with ``shift_inline_to_zero`` for the 3-D OBN first-quadrant
            layout.
        shift_points_xy
            Points used to compute the shift minima; defaults to
            ``points_xy``. Pass the union of sources + receivers to
            reproduce the legacy ``rotate_csg_index`` shift, which always
            minimised over both even when the rotation was fit on shots only.

        Returns
        -------
        RotatedFrame
            Frame whose :meth:`to_model` aligns the fitted line to
            ``target_axis`` and applies the requested shifts.
        """
        if target_axis not in ("x", "y"):
            raise ValueError(f"target_axis must be 'x' or 'y'; got {target_axis!r}")
        pts = np.asarray(points_xy, dtype="float64")
        _, line_angle = principal_direction(pts)
        target_angle = 0.0 if target_axis == "x" else np.pi / 2.0
        applied = _normalize_angle_rad(target_angle - line_angle)
        c, s = np.cos(applied), np.sin(applied)
        # Match the legacy ``rotation_metadata.json`` convention used by
        # ``to_model`` (``model = (utm - origin) @ R.T + shifts``). NOTE this
        # is the opposite-sign matrix to :meth:`from_rotation_deg`, which maps
        # its ``rotation_deg`` argument through ``[[c, s], [-s, c]]``.
        R = np.array([[c, -s], [s, c]], dtype="float64")
        origin = np.mean(pts, axis=0)

        inline_shift = 0.0
        crossline_shift = 0.0
        if shift_inline_to_zero or shift_crossline_to_zero:
            shift_pts = (
                pts if shift_points_xy is None
                else np.asarray(shift_points_xy, dtype="float64")
            )
            rotated = (shift_pts - origin[None, :]) @ R.T
            if target_axis == "x":
                inline_vals, crossline_vals = rotated[:, 0], rotated[:, 1]
            else:
                inline_vals, crossline_vals = rotated[:, 1], rotated[:, 0]
            if shift_inline_to_zero:
                inline_shift = -float(np.nanmin(inline_vals))
            if shift_crossline_to_zero:
                crossline_shift = -float(np.nanmin(crossline_vals))

        return cls(
            origin_xy_utm=origin,
            rotation_matrix=R,
            target_axis=target_axis,
            inline_shift=inline_shift,
            crossline_shift=crossline_shift,
        )

    # --------------------------------------------------------- transforms
    def to_model(self, xy_utm: np.ndarray) -> np.ndarray:
        """Project ``(..., 2)`` UTM xy to model-frame xy."""
        xy = np.asarray(xy_utm, dtype="float64")
        if xy.shape[-1] != 2:
            raise ValueError(
                f"to_model: last axis must be size 2 (xy); got shape {xy.shape}"
            )
        flat = xy.reshape(-1, 2) - self.origin_xy_utm[None, :]
        rotated = flat @ self.rotation_matrix.T
        # Add inline / crossline shifts along the chosen target axis.
        if self.target_axis == "y":
            rotated[:, 0] += self.crossline_shift
            rotated[:, 1] += self.inline_shift
        else:
            rotated[:, 0] += self.inline_shift
            rotated[:, 1] += self.crossline_shift
        return rotated.reshape(xy.shape)

    def to_utm(self, xy_model: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_model` — model-frame xy back to UTM xy."""
        xy = np.asarray(xy_model, dtype="float64")
        if xy.shape[-1] != 2:
            raise ValueError(
                f"to_utm: last axis must be size 2 (xy); got shape {xy.shape}"
            )
        flat = xy.reshape(-1, 2).copy()
        # Undo the shifts first.
        if self.target_axis == "y":
            flat[:, 0] -= self.crossline_shift
            flat[:, 1] -= self.inline_shift
        else:
            flat[:, 0] -= self.inline_shift
            flat[:, 1] -= self.crossline_shift
        # ``R`` is a rotation so ``R^-1 == R^T``; we applied ``@ R.T`` in
        # ``to_model`` so the inverse is ``@ R``.
        utm = flat @ self.rotation_matrix + self.origin_xy_utm[None, :]
        return utm.reshape(xy.shape)

    # --------------------------------------------------------- serdes
    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_xy": self.origin_xy_utm.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "target_axis": str(self.target_axis),
            "inline_shift": float(self.inline_shift),
            "crossline_shift": float(self.crossline_shift),
        }


def load_rotation_metadata(path: str | Path) -> RotatedFrame:
    """Build a :class:`RotatedFrame` from a JSON metadata file.

    Thin wrapper around :meth:`RotatedFrame.from_metadata` for the
    file-path call style; matches the legacy
    ``fwi_workflow.geometry.line2d.load_rotation_transform``.
    """
    return RotatedFrame.from_metadata(path)
