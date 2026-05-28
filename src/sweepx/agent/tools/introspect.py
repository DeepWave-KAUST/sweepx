"""``list_equations`` — let the LLM discover which equations sweep supports.

The flat builders can't enumerate sweep's ~33 equations in their schema, and the
LLM must not guess equation names or their model requirements. This tool exposes
the real registry: each equation maps to the ordered list of physical-property
models it needs (Acoustic→[vp], AcousticVTI→[vp,epsilon,delta], ...).
"""

from __future__ import annotations

import typing
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


# ===========================================================================
# describe_task_schema — expose the full sweep_tasks parameter surface so the
# LLM can understand and fill ANY field of ANY task type, not just the flat
# build_forward_spec shortcuts.
# ===========================================================================

def _task_classes() -> dict[str, type]:
    from sweep_tasks import schemas

    return {
        "forward": schemas.ForwardSpec,
        "wavefield": schemas.WavefieldSpec,
        "fwi": schemas.FWISpec,
        "lsrtm": schemas.LSRTMSpec,
        "rtm": schemas.RTMSpec,
        "introspect": schemas.IntrospectSpec,
    }


def _type_name(ann: Any) -> str:
    """Human-readable name for a (possibly nested) type annotation."""
    if ann is None or ann is type(None):
        return "None"
    origin = typing.get_origin(ann)
    if origin is None:
        return getattr(ann, "__name__", str(ann).replace("typing.", ""))
    args = typing.get_args(ann)
    import types as _types

    if origin is typing.Union or (hasattr(_types, "UnionType") and origin is _types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        body = " | ".join(_type_name(a) for a in non_none)
        return body + ("?" if type(None) in args else "")
    if origin in (list, tuple, set):
        inner = ", ".join(_type_name(a) for a in args) if args else ""
        return f"{origin.__name__}[{inner}]"
    if origin is dict:
        return f"dict[{', '.join(_type_name(a) for a in args)}]" if args else "dict"
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"
    # Annotated[X, ...] → describe X
    if args:
        return _type_name(args[0])
    return getattr(origin, "__name__", str(origin))


def _safe_default(finfo: Any) -> Any:
    from pydantic_core import PydanticUndefined

    if getattr(finfo, "default", PydanticUndefined) is not PydanticUndefined and finfo.default is not None:
        d = finfo.default
    elif getattr(finfo, "default_factory", None) is not None:
        try:
            d = finfo.default_factory()
        except Exception:
            return "<factory>"
    else:
        return None
    if isinstance(d, (int, float, str, bool)):
        return d
    if isinstance(d, BaseModel):
        return f"<{type(d).__name__} defaults>"
    return str(d)


def _summarize_model(cls: type) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for fname, finfo in cls.model_fields.items():
        entry: dict[str, Any] = {
            "type": _type_name(finfo.annotation),
            "required": finfo.is_required(),
        }
        if finfo.description:
            entry["description"] = finfo.description
        if not finfo.is_required():
            entry["default"] = _safe_default(finfo)
        fields[fname] = entry
    return fields


def _models_in_annotation(ann: Any) -> list[type]:
    """All BaseModel subclasses referenced by an annotation (Optional/Union/list/Annotated)."""
    found: list[type] = []
    seen: set[int] = set()

    def walk(a: Any) -> None:
        if a is None or a is type(None):
            return
        if isinstance(a, type) and issubclass(a, BaseModel):
            if id(a) not in seen:
                seen.add(id(a))
                found.append(a)
            return
        for sub in typing.get_args(a):
            walk(sub)

    walk(ann)
    return found


class DescribeTaskSchemaParams(BaseModel):
    task_type: str = Field(
        ...,
        description="One of: forward, wavefield, fwi, lsrtm, rtm, introspect.",
    )
    section: str | None = Field(
        None,
        description=(
            "Optional top-level field name to expand (e.g. 'optimizer', 'loss', 'geometry', "
            "'wavelet', 'physics', 'backend', 'obs', 'stages', 'data_plan'). Expands nested / "
            "discriminated-union sub-schemas so you can see their fields and 'kind' variants."
        ),
    )


@register(
    name="describe_task_schema",
    description=(
        "Describe the full parameter surface of any sweep task type (forward / wavefield / fwi / "
        "lsrtm / rtm / introspect): every field with its type, whether required, default, and doc. "
        "Call with just task_type for the top-level field list, then pass `section` to expand a "
        "nested field (e.g. section='optimizer' shows the adam/sgd/lbfgs variants). Use this to "
        "understand and fill ANY parameter before calling build_spec — never guess field names."
    ),
    params_model=DescribeTaskSchemaParams,
)
def describe_task_schema(args: DescribeTaskSchemaParams) -> dict[str, Any]:
    try:
        classes = _task_classes()
    except Exception as exc:
        return {"error": f"could not import sweep_tasks.schemas: {exc}"}
    cls = classes.get(args.task_type)
    if cls is None:
        return {"error": f"unknown task_type '{args.task_type}'. choices: {sorted(classes)}"}

    if not args.section:
        return {
            "task_type": args.task_type,
            "fields": _summarize_model(cls),
            "note": (
                "Pass section=<field> to expand a nested field (e.g. 'optimizer','loss','geometry',"
                "'wavelet','physics','backend','obs','stages'). Then build the task with build_spec."
            ),
        }

    if args.section not in cls.model_fields:
        return {
            "error": f"'{args.section}' is not a field of {args.task_type}. "
            f"fields: {sorted(cls.model_fields)}"
        }
    ann = cls.model_fields[args.section].annotation
    models = _models_in_annotation(ann)
    if not models:
        return {
            "task_type": args.task_type,
            "section": args.section,
            "type": _type_name(ann),
            "note": "leaf field (primitive / literal); no nested schema to expand.",
        }
    return {
        "task_type": args.task_type,
        "section": args.section,
        "type": _type_name(ann),
        "variants": {m.__name__: _summarize_model(m) for m in models},
        "note": "discriminated unions pick a variant via its 'kind' field; set that in your spec dict.",
    }
