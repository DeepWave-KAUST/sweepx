"""Acquisition geometry: source / receiver positions and sampling parameters.

Positions are **grid indices**, not physical coordinates — this matches the
shape convention used by the `sweep` propagators. The optional `dh` field
records the spacing if you want to convert back to physical coordinates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Tuple

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
