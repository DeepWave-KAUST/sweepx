"""``list_equations`` — let the LLM discover which equations sweep supports.

The flat builders can't enumerate sweep's ~33 equations in their schema, and the
LLM must not guess equation names or their model requirements. This tool exposes
the real registry: each equation maps to the ordered list of physical-property
models it needs (Acoustic→[vp], AcousticVTI→[vp,epsilon,delta], ...).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sweep_agent.tools import register


def equation_model_names(cls: type) -> list[str]:
    """Ordered required model names for an equation class.

    Mirrors sweep_tasks.runner._model_names_for_equation: prefer MODEL_SPECS,
    fall back to a ``models`` class attribute. Order matters — the runner expects
    multi-model lists in MODEL_SPECS order.
    """
    specs = getattr(cls, "MODEL_SPECS", None)
    if specs:
        try:
            return [s.name for s in specs]
        except Exception:
            pass
    models_attr = getattr(cls, "models", None)
    if isinstance(models_attr, (list, tuple)):
        return [str(m) for m in models_attr]
    return []


def all_equations() -> dict[str, list[str]]:
    """Map every registered sweep equation name → its ordered model names."""
    import sweep.equations as eq_mod

    classes = eq_mod._equation_classes()
    return {name: equation_model_names(classes[name]) for name in sorted(classes)}


class ListEquationsParams(BaseModel):
    filter: str | None = Field(
        None,
        description="Case-insensitive substring filter on the equation name (e.g. 'vti', '3d', 'elastic'). None lists all.",
    )


@register(
    name="list_equations",
    description=(
        "List the wave equations sweep supports and the physical-property models each one requires "
        "(e.g. Acoustic→[vp]; AcousticVTI→[vp, epsilon, delta]; Elastic→[vp, vs, rho]). Call this "
        "BEFORE choosing an `equation` for build_forward_spec — never guess equation names or their "
        "model requirements. Use the optional substring `filter` to narrow (e.g. 'vti'). For any "
        "non-Acoustic equation, you must supply the extra models via build_forward_spec(extra_models=...)."
    ),
    params_model=ListEquationsParams,
)
def list_equations(args: ListEquationsParams) -> dict[str, Any]:
    try:
        eqs = all_equations()
    except Exception as exc:
        return {"error": f"could not import sweep.equations: {exc}"}
    if args.filter:
        f = args.filter.lower()
        eqs = {k: v for k, v in eqs.items() if f in k.lower()}
        if not eqs:
            return {"count": 0, "equations": {}, "note": f"no equation name matches '{args.filter}'."}
    return {
        "count": len(eqs),
        "equations": {k: {"models": v} for k, v in eqs.items()},
        "note": (
            "`models` is the ordered list of fields the equation needs. Pass vp via "
            "build_forward_spec(vp_path=...) and the rest via extra_models={name: path}."
        ),
    }
