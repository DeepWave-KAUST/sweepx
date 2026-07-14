"""Source/receiver illumination accumulation + preconditioning.

Extracted verbatim from ``runner.py`` (bit-identical). Used by the fwi /
multisource / freqsel task paths.
"""


def _accumulate_illumination(solver, sill_sum, rill_sum):
    """Snapshot a chunk's solver illumination tensors into running buffers.

    Returns the updated ``(sill_sum, rill_sum)`` pair. The propagator
    backend zeros its ``source_illumination`` / ``receiver_illumination``
    attrs at the start of every forward call, so each chunk's backward
    must be snapshot here BEFORE the next chunk's forward kicks in.

    Either buffer may start as ``None``; the helper detach-clones the
    first snapshot and adds in-place thereafter.
    """
    import torch

    sill = getattr(solver, "source_illumination", None)
    rill = getattr(solver, "receiver_illumination", None)
    if sill is None or rill is None:
        return sill_sum, rill_sum
    sill_d = sill.detach()
    rill_d = rill.detach()
    if sill_sum is None:
        sill_sum = sill_d.clone()
    else:
        sill_sum.add_(sill_d)
    if rill_sum is None:
        rill_sum = rill_d.clone()
    else:
        rill_sum.add_(rill_d)
    return sill_sum, rill_sum


def _apply_illumination_precond(
    grad_tensor, sill_sum, rill_sum, *, eps: float, exponent: float,
    relative_epsilon: float | None = None,
) -> None:
    """Divide ``grad_tensor`` in place by ``(S·R + eps)**exponent``.

    Shapes must match (the illumination buffers are allocated by the
    backend to match the unpadded velocity tensor). No-op when either
    buffer is ``None`` (no chunk produced illumination, e.g. all chunks
    were empty on this rank).

    ``relative_epsilon`` (when set) replaces the absolute ``eps`` with a
    water level ``relative_epsilon * max(S·R)`` recomputed here at every
    application, capping the maximum boost at ``relative_epsilon**-exponent``.
    S·R spans many decades on field data and its scale drifts with the
    stage band/residual, so a fixed absolute eps is either a no-op or a
    kill-switch; the relative form tracks the scale automatically.
    """
    import torch

    if sill_sum is None or rill_sum is None or grad_tensor is None:
        return
    sr_raw = sill_sum * rill_sum
    if relative_epsilon is not None:
        eps = float(relative_epsilon) * float(sr_raw.max())
        if eps <= 0.0:  # all-zero illumination -> nothing to precondition
            return
    sr = sr_raw + float(eps)
    if abs(exponent - 1.0) < 1.0e-12:
        scale = 1.0 / sr
    elif abs(exponent - 0.5) < 1.0e-12:
        scale = 1.0 / torch.sqrt(sr)
    else:
        scale = 1.0 / torch.pow(sr, float(exponent))
    grad_tensor.mul_(scale)
