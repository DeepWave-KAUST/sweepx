"""Model-bounds helpers: resolve the active ModelBounds entry and clamp
inverted tensors to it in place.

Extracted verbatim from ``runner.py`` (bit-identical).
"""


def _effective_bound(bounds_by_name, name):
    """Return the ModelBounds entry for ``name`` if active, else None.

    A bound is considered inactive when its ``enabled`` flag is False —
    callers that read ``min``/``max`` (clamp, reparam network, QC) all
    funnel through this helper so the YAML's ``enabled: false`` toggle
    is honoured uniformly. Backwards compatible: bounds without the
    ``enabled`` attribute (older schemas) are always active.
    """
    if not bounds_by_name:
        return None
    bound = bounds_by_name.get(name) if hasattr(bounds_by_name, "get") else None
    if bound is None:
        return None
    if not getattr(bound, "enabled", True):
        return None
    return bound


def _apply_bounds(inv_tensors_by_name, bounds_by_name, *, skip_names=()) -> None:
    """Clamp each named tensor to its model_bounds entry in place.

    ``skip_names`` (e.g. names handled by a reparam network) are passed
    through untouched — the network already enforces its own bounds via
    its render-time clamp.
    """
    if not bounds_by_name:
        return
    skip = set(skip_names)
    for name, t in inv_tensors_by_name.items():
        if name in skip:
            continue
        bound = _effective_bound(bounds_by_name, name)
        if bound is None:
            continue
        t.data.clamp_(min=bound.min, max=bound.max)
