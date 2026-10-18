"""Wavelet I/O — read the source signature from an ``.npz`` container.

Two schemas in the wild:

* **Direct-batch format** (legacy 1-D arrays from sweep wavelet-fit jobs):
  the array lives under the key ``"wavelet"`` and the time vector under
  ``"time_s"``. Both arrays are 1-D, same length.
* **SIREN-pipeline format**: the array lives under one of
  ``("optimized_siren_wavelet", "direct_causal_wavelet",
  "initial_causal_wavelet")`` with a scalar ``"dt_s"`` (s) and optional
  ``"source_delay_s"`` (s) that records the zero-prepad applied during
  SIREN training. Consumers should mirror that prepad on the observed
  data so syn and obs stay aligned (see :attr:`WaveletNPZ.source_delay_s`).

:func:`load_wavelet_npz` auto-detects the schema (via the preferred-key
fallback list, matching ``fwi_workflow-dev``'s 3-D CRG FWI runner) and
returns a unified :class:`WaveletNPZ` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


_DEFAULT_PREFER_KEYS: tuple[str, ...] = (
    "wavelet",
    "optimized_siren_wavelet",
    "direct_causal_wavelet",
    "initial_causal_wavelet",
)


@dataclass(frozen=True)
class WaveletNPZ:
    """Loaded source-wavelet payload.

    Attributes
    ----------
    samples
        1-D ``float32`` array of length ``nt`` with the wavelet amplitudes.
    dt_s
        Sample interval in seconds.
    source_delay_s
        Optional zero-prepad in front of the wavelet (s). The SIREN
        wavelet-fit pipeline writes this so downstream FWI can left-pad
        observed traces by the same amount; ``None`` when absent.
    key
        Which npz key the samples were read from (helpful for logs).
    """

    samples: np.ndarray
    dt_s: float
    source_delay_s: float | None = None
    key: str = ""


def load_wavelet_npz(
    path: str | Path,
    *,
    prefer_keys: Sequence[str] = _DEFAULT_PREFER_KEYS,
    explicit_key: str | None = None,
) -> WaveletNPZ:
    """Read a wavelet from an ``.npz`` file.

    The selected array key is, in order: ``explicit_key`` (if non-``None``);
    otherwise the first key in ``prefer_keys`` present in the npz. Raises
    ``KeyError`` if none match.

    The sample interval is taken from ``dt_s`` (scalar) when present;
    otherwise inferred from ``time_s`` via the median diff (matches the
    legacy ``fwi_workflow-dev`` behaviour). If neither is present a
    ``KeyError`` is raised — sweep-stack will not invent a ``dt``.

    Parameters
    ----------
    path
        Path to the ``.npz`` file.
    prefer_keys
        Ordered fallback list of npz keys for the wavelet array.
    explicit_key
        Force-select this key; bypasses ``prefer_keys``.
    """
    path = Path(path)
    npz = np.load(path, allow_pickle=False)
    try:
        files = set(npz.files)
        if explicit_key is not None:
            if explicit_key not in files:
                raise KeyError(
                    f"explicit_key {explicit_key!r} not in {sorted(files)} "
                    f"(file={path})"
                )
            key = explicit_key
        else:
            key = next((k for k in prefer_keys if k in files), None)
            if key is None:
                raise KeyError(
                    f"wavelet npz {path} has none of {tuple(prefer_keys)}; "
                    f"got {sorted(files)}"
                )
        samples = np.asarray(npz[key], dtype=np.float32).reshape(-1)
        if "dt_s" in files:
            dt_s = float(np.asarray(npz["dt_s"]).item())
        elif "time_s" in files:
            time_s = np.asarray(npz["time_s"], dtype=np.float64)
            if time_s.size < 2:
                raise KeyError(
                    f"wavelet npz {path} has 'time_s' but only "
                    f"{time_s.size} sample(s); cannot infer dt"
                )
            dt_s = float(np.median(np.diff(time_s)))
        else:
            raise KeyError(
                f"wavelet npz {path} has neither 'dt_s' nor 'time_s'; "
                "supply one or call with explicit_key + a sibling timing file."
            )
        if "source_delay_s" in files:
            source_delay_s: float | None = float(
                np.asarray(npz["source_delay_s"]).item()
            )
        else:
            source_delay_s = None
    finally:
        npz.close()
    return WaveletNPZ(
        samples=samples, dt_s=dt_s,
        source_delay_s=source_delay_s, key=key,
    )


__all__ = ["WaveletNPZ", "load_wavelet_npz"]
