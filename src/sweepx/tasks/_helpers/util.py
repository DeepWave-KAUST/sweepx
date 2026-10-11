"""Small runner utilities (time id, device, seed). Verbatim from runner.py."""
from datetime import datetime, timezone

import numpy as np
import torch

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_task_id(task_type: str, override: str | None) -> str:
    if override:
        return override
    return f"{task_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _resolve_device(device: str) -> "torch.device":
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --- Domain-decomposition (DD) mode helpers (env-gated; no-op when off) -------
# DD splits the solver's spatial domain across N GPUs (ModelParallel), with the
# velocity_inr reparam rendered PER TILE (render_window). Validated standalone
# in runs/dd_ifwi_smoke; wired into _run_fwi_plan_streaming behind SWEEP_DD_ENABLE.


def _apply_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)


def _is_cuda_dev(dev) -> bool:
    """``True`` iff CUDA is available **and** ``dev`` names a CUDA device.

    Accepts ``torch.device`` or string ("cuda" / "cuda:N" / "cpu"). Used
    by every site that branches between GPU and CPU code paths (obs
    pipeline, local-window solver cache, illumination, etc.) — keeping
    the check in one place avoids the historical ~5 different inline
    repetitions that drifted in subtle ways (some only checked
    ``torch.cuda.is_available()``, others only the device type).
    """
    import torch  # lazy: runner is callable from torch-less environments
    if not torch.cuda.is_available():
        return False
    if isinstance(dev, torch.device):
        return dev.type == "cuda"
    if isinstance(dev, str):
        return dev.startswith("cuda")
    return False


_BLOWUP_HINT = (
    "The simulation blew up: check the CFL condition (`time.dt` against "
    "`grid.dh` and the fastest velocity) and the models for NaN / Inf / zero "
    "values."
)


def _nonfinite_summary(arr: "np.ndarray", name: str, saved_to=None, *,
                       axis_names: tuple[str, ...] = ("shot",),
                       labels: dict | None = None,
                       chunk_elems: int = 1 << 22) -> str | None:
    """One sentence on the NaN / Inf in ``arr``, or ``None`` when it is finite.

    The scan walks ``arr`` one axis-0 slice at a time and in ``chunk_elems``
    pieces inside it, so the temporary boolean mask is at most ``chunk_elems``
    bytes however large a 3-D record is — never a full-size copy.

    ``axis_names`` names the leading axes for the message (the location of the
    first bad value is reported along each of them). ``labels`` maps an axis
    name to a sequence that turns an index into something readable — e.g. the
    snapshot time step at each snapshot index.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0 or not (np.issubdtype(arr.dtype, np.floating)
                             or np.issubdtype(arr.dtype, np.complexfloating)):
        return None
    n_bad = n_nan = 0
    first = None
    for i in range(arr.shape[0]):
        # A view when the slice is contiguous (the usual case for a tensor
        # brought to the host with .cpu().numpy()); one slice copied otherwise.
        flat = np.ascontiguousarray(arr[i]).reshape(-1)
        for lo in range(0, flat.size, chunk_elems):
            block = flat[lo:lo + chunk_elems]
            bad = ~np.isfinite(block)
            nb = int(np.count_nonzero(bad))
            if nb == 0:
                continue
            n_bad += nb
            n_nan += int(np.count_nonzero(np.isnan(block)))
            if first is None:
                first = (i,) + tuple(int(v) for v in np.unravel_index(
                    lo + int(np.argmax(bad)), arr.shape[1:]))
    if n_bad == 0:
        return None
    where = []
    for ax, idx in zip(axis_names, first):
        lab = (labels or {}).get(ax)
        named = lab is not None and idx < len(lab)
        where.append(f"{ax} {idx}" + (f" ({lab[idx]})" if named else ""))
    saved = f" Saved to {saved_to} for inspection." if saved_to else ""
    return (
        f"{name} contains {n_bad} non-finite value(s) out of {arr.size} "
        f"({n_nan} NaN, {n_bad - n_nan} Inf); the first is at "
        f"{', '.join(where)} (index {first} of shape {tuple(arr.shape)})."
        f"{saved}"
    )


def _raise_if_nonfinite(*summaries: str | None) -> None:
    """Raise ``ValueError`` naming every array :func:`_nonfinite_summary` flagged.

    Called on the host copies AFTER they were written to disk, so a blown-up
    run still leaves the arrays behind for inspection while the task ends
    ``failed`` instead of reporting success on garbage.
    """
    found = [s for s in summaries if s]
    if found:
        raise ValueError(" ".join(found) + " " + _BLOWUP_HINT)
