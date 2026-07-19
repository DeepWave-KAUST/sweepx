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


# Preferred "plain" equations when a model set is ambiguous (e.g. vp+vs+rho is
# shared by Elastic, DASElastic, ElasticAPM, ...). Earlier = more canonical.
_CANONICAL_EQUATIONS = (
    "Acoustic", "Elastic", "AcousticVTI", "AcousticTTI",
    "Acoustic3D", "Elastic3D", "AcousticVTI3D", "ElasticTTI",
)


def equations_for_models(extra_models: set[str] | list[str], ndim: int | None = None) -> list[str]:
    """Equations whose NON-primary models exactly equal ``extra_models``, ranked
    so the canonical / dimension-appropriate choice comes first.

    The primary model (the equation's first input — ``vp``, or ``vp0`` for
    elastic-TTI) always comes from ``vp_path``; ``extra_models`` are everything
    after it. So we match on ``models[1:]``, which works whether the primary is
    named ``vp`` or ``vp0``. Used to (a) auto-pick an equation from the files a
    user supplied and (b) suggest the right equation when the LLM left a
    multi-parameter model set on a plain-acoustic equation. Specialized variants
    (DAS / Visco / APM / Curvilinear / staggered / 1st-order) are deprioritized;
    when ``ndim`` is given, equations matching that dimensionality are preferred.
    """
    extra = set(extra_models)
    matches = [e for e, m in all_equations().items() if len(m) >= 1 and set(m[1:]) == extra]

    def rank(e: str) -> tuple:
        is3d = "3D" in e
        dim_mismatch = 0 if ndim is None else int((ndim == 3) != is3d)
        canon = _CANONICAL_EQUATIONS.index(e) if e in _CANONICAL_EQUATIONS else len(_CANONICAL_EQUATIONS)
        specialized = int(any(t in e for t in ("DAS", "Visco", "APM", "Curvilinear", "SG", "1st")))
        return (dim_mismatch, canon, specialized, len(e), e)

    return sorted(matches, key=rank)


def equation_default_fields(name: str) -> tuple[list[str], list[str]]:
    """(default_source_fields, default_receiver_fields) for an equation, statically.

    Reads the property getters on an uninitialised instance — works because these
    getters return fixed field lists (Acoustic→['h1'], Elastic→['sxx','szz'] /
    ['vx','vz'], AcousticVTIDefault3D→['sH','sV'] / ['vz'], ...). Returns ([], [])
    when unavailable.
    """
    import sweep.equations as eq_mod

    cls = eq_mod._equation_classes().get(name)
    if cls is None:
        return [], []

    def _get(attr: str) -> list[str]:
        prop = getattr(cls, attr, None)
        if isinstance(prop, property):
            try:
                return list(prop.fget(cls.__new__(cls)))
            except Exception:
                return []
        if isinstance(prop, (list, tuple)):
            return [str(x) for x in prop]
        return []

    return _get("default_source_fields"), _get("default_receiver_fields")


def equation_default_pml(name: str) -> str | None:
    """The equation's `default_pml_type` class attribute (Acoustic→'cpmlr',
    Elastic→'cpmls', ...). 'cpmls' yields 8 PML profiles (staggered grid) which
    Elastic's step needs; the schema's plain default 'cpmlr' (6) would crash it."""
    import sweep.equations as eq_mod

    cls = eq_mod._equation_classes().get(name)
    if cls is None:
        return None
    v = getattr(cls, "default_pml_type", None)
    return v if isinstance(v, str) else None


def equation_method(name: str) -> str:
    """First docstring line of an equation — its numerical method / reference.

    Lets the LLM distinguish anisotropic variants, e.g. AcousticVTIDuveneck
    ('First-order ... standard staggered grid') vs AcousticVTIAlkhalifah
    ('... Alkhalifah eta') vs AcousticVTI ('... Liang 2022')."""
    import sweep.equations as eq_mod

    cls = eq_mod._equation_classes().get(name)
    if cls is None:
        return ""
    return (cls.__doc__ or "").strip().split("\n")[0][:100]


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
    equations: dict[str, Any] = {}
    for k, models in eqs.items():
        sf, rf = equation_default_fields(k)
        equations[k] = {
            "models": models,
            "method": equation_method(k),
            "default_source": sf,
            "default_receiver": rf,
        }
    return {
        "count": len(equations),
        "equations": equations,
        "note": (
            "`method` is the equation's numerical formulation / reference — use it to choose among "
            "anisotropic variants (e.g. VTI: Duveneck 1st-order vs Alkhalifah eta vs Liang 2022). "
            "`models` is the ordered field list (pass vp via the builder, the rest via extra_models). "
            "`default_source`/`default_receiver` are auto-selected by build_forward_spec/build_fwi_spec, "
            "so you normally don't set source_type/receiver_type yourself."
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
