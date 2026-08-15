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
