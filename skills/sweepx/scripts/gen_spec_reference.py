#!/usr/bin/env python
"""Generate ``references/spec-schema.md`` from the task-spec schema itself.

Walks every task type in ``sweepx.tasks.TASK_TYPES`` and every pydantic model
reachable from it, and writes one table per model: field, type, default (or
"required") and the one-line note found next to the field in the source.

    python scripts/gen_spec_reference.py                 # writes ../references/spec-schema.md
    python scripts/gen_spec_reference.py -o other.md     # or anywhere else

Re-run it after upgrading sweepx; the output names the version it was built from.
"""
from __future__ import annotations

import argparse
import enum
import inspect
import re
import types
import typing
from pathlib import Path

from pydantic import BaseModel

import sweepx
from sweepx.tasks import TASK_TYPES
from sweepx.tasks import yaml_io

# Sentences about internal history (other repos, commits, task tickets) are
# dropped from notes: they mean nothing to a user.
_DENY = re.compile(r"\w-dev\b|\bcommit [0-9a-f]{6,}|\b[0-9a-f]{7}\b|/FATT|Gap \d|TASK \d+|"
                   r"Phase.?1|sweep-tasks behavior", re.I)


def _first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z`(\"'])", text)
    keep = [p for p in parts if not _DENY.search(p)]
    out = keep[0] if keep else ""
    if len(out) > limit:
        out = out[: limit - 1].rsplit(" ", 1)[0] + " ..."
    return out.replace("|", "\\|")


def _doc_summary(cls) -> str:
    # Only a docstring the class itself defines (not pydantic's BaseModel one).
    doc = inspect.cleandoc(cls.__dict__.get("__doc__") or "")
    para = doc.split("\n\n", 1)[0]
    return _first_sentence(para, limit=400)


def _field_notes(cls) -> dict[str, str]:
    """Trailing ``# comment`` of a field line, else the comment block above it."""
    try:
        src = inspect.getsource(cls).splitlines()
    except (OSError, TypeError):
        return {}
    notes: dict[str, str] = {}
    block: list[str] = []
    pat = re.compile(r"^\s{4}(\w+)\s*:\s*[^=#]+?(?:=.*?)?(?:#\s*(.*))?$")
    for line in src[1:]:
        s = line.strip()
        if s.startswith("#"):
            block.append(s.lstrip("#").strip())
            continue
        m = pat.match(line)
        if m and not line.startswith("     ") and m.group(1) in cls.model_fields:
            trailing = (m.group(2) or "").strip()
            notes[m.group(1)] = _first_sentence(trailing or " ".join(block))
        if s and not s.startswith("#"):
            block = []
    return notes


# Notes checked against the runner (runs or code) where the source comment is wrong or silent.
# (class, field) -> note. Keep each one tied to a reproducible check.
ERRATA = {
    ("ObsSpec", "npy_path"): (
        "Verified: must be `(nshots, nt, nrec, 1)` float32, the layout of a forward "
        "run's `output/record.npy`. Only axis 0 is checked; an `(nshots, nrec, nt)` "
        "array is broadcast against the synthetic and the run 'succeeds' on garbage."),
    ("FromSegyGeometry", "byte_map"): (
        "0-based byte offsets inside the 240-byte trace header, keys from "
        "`sweepx.io.segy_index.SEGY_REV1_BYTES` (`sx`, `sy`, `rx`, `ry`, `shot`, "
        "`coord_scalar`, `source_depth`, `receiver_depth`, ...). Unknown keys (e.g. `gx`) "
        "are ignored silently; a 1-based value (`sx: 73`) reads the wrong bytes."),
    ("ObsSegyConfig", "byte_map"): "Same convention as `FromSegyGeometry.byte_map`.",
    ("StageSpec", "dh_m"): (
        "Grid spacing of this stage. Unset: the PREVIOUS stage's value on the conventional "
        "path (so return to grid.dh explicitly), grid.dh under frequency selection."),
    ("StageSpec", "dt_s"): (
        "Time step of this stage; inherited like dh_m. A new dt_s without nt keeps the "
        "record length."),
    ("StageSpec", "nt"): "Record length in steps of this stage; inherited like dh_m.",
    ("StageSpec", "bandpass"): (
        "Not inherited: a stage without it is broadband. Filters the pristine obs, so "
        "stages never compose filters."),
    ("StageSpec", "lr_scale"): "This stage's base lr = optimizer.lr x lr_scale.",
    ("StageSpec", "inr_lr_scale"): "Scales reparam.lr; refused without reparam:.",
    ("StageSpec", "optimizer_reset"): (
        "Fresh optimizer state at stage entry (a grid model's is rebuilt anyway when dh_m changes)."),
}


def _anchor(name: str) -> str:
    return name.lower()


def _fmt_type(tp, seen: set) -> str:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is type(None):
        return "null"
    if origin is typing.Annotated:
        return _fmt_type(args[0], seen)
    if origin is typing.Literal:
        return " \\| ".join(f"`{a!r}`" if not isinstance(a, bool) else f"`{str(a).lower()}`"
                            for a in args)
    if origin in (typing.Union, types.UnionType):
        return " \\| ".join(_fmt_type(a, seen) for a in args)
    if origin in (list, tuple, dict):
        inner = ", ".join("..." if a is Ellipsis else _fmt_type(a, seen) for a in args)
        return f"{origin.__name__}[{inner}]"
    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        seen.add(tp)
        return f"[{tp.__name__}](#{_anchor(tp.__name__)})"
    if inspect.isclass(tp) and issubclass(tp, Path):
        return "path"
    if inspect.isclass(tp) and issubclass(tp, enum.Enum):
        return " \\| ".join(f"`{m.value!r}`" for m in tp)
    return getattr(tp, "__name__", str(tp)).replace("|", "\\|")


def _fmt_default(field) -> str:
    if field.is_required():
        return "**required**"
    if field.default_factory is not None:
        val = field.default_factory()
    else:
        val = field.default
    if isinstance(val, BaseModel):
        return "`{}` (all defaults)"
    if isinstance(val, Path):
        val = str(val)
    if isinstance(val, bool):
        return f"`{str(val).lower()}`"
    if val is None:
        return "`null`"
    return f"`{val!r}`".replace("|", "\\|")


def _collect(roots) -> list[type]:
    order: list[type] = []
    todo = list(roots)
    while todo:
        cls = todo.pop(0)
        if cls in order:
            continue
        order.append(cls)
        seen: set = set()
        for f in cls.model_fields.values():
            _fmt_type(f.annotation, seen)
        todo.extend(sorted(seen - set(order), key=lambda c: c.__name__))
    return order


def _discriminator(cls) -> str | None:
    for name in ("kind", "task_type"):
        f = cls.model_fields.get(name)
        if (f is not None and typing.get_origin(f.annotation) is typing.Literal
                and len(typing.get_args(f.annotation)) == 1):
            return f"{name}: {typing.get_args(f.annotation)[0]}"
    return None


def render() -> str:
    roots = list(TASK_TYPES.values())
    models = _collect(roots)
    out: list[str] = []
    w = out.append
    w("# Task spec schema (generated)")
    w("")
    w(f"Generated by `scripts/gen_spec_reference.py` from sweepx {sweepx.__version__}. "
      "Do not edit by hand; re-run the script after upgrading.")
    w("")
    w("Every model forbids unknown keys (`extra=\"forbid\"`): a misspelt key is a "
      "validation error. Cross-field rules (which keys a path refuses, one-of "
      "sources, ...) are not in these tables; they are enforced by validators and "
      "summarised in `workflows.md`.")
    w("")
    w("Path resolution: string values expand `$VAR` / `${VAR}` (an unset variable "
      "is an error). Keys named "
      + ", ".join(f"`{k}`" for k in sorted(yaml_io.RESOLVED_PATH_KEYS))
      + " (any depth), and "
      + ", ".join(f"`{k}`" for k in sorted(yaml_io.RESOLVED_WATER_PATH_KEYS))
      + " inside `water:`, are resolved relative to the YAML file. Every other "
      "path (`plan_path`, `index_path`, `data_mask_path`, `topography`, "
      "`init_from`, ...) is relative to the working directory of the process.")
    w("")
    w("## Contents")
    w("")
    w("Task types: " + ", ".join(f"[`{t}`](#{_anchor(c.__name__)})"
                                 for t, c in TASK_TYPES.items()))
    w("")
    w("Sub-models: " + ", ".join(f"[{c.__name__}](#{_anchor(c.__name__)})"
                                 for c in models if c not in roots))
    w("")
    for cls in models:
        w(f"## {cls.__name__}")
        w("")
        disc = _discriminator(cls)
        summary = _doc_summary(cls)
        if disc:
            w(f"Selected by `{disc}`.")
        if summary:
            w(summary)
        if disc or summary:
            w("")
        notes = _field_notes(cls)
        for (cname, fname), note in ERRATA.items():
            if cname == cls.__name__:
                notes[fname] = note.replace("|", "\\|")
        w("| field | type | default | note |")
        w("|---|---|---|---|")
        for name, f in cls.model_fields.items():
            seen: set = set()
            w(f"| `{name}` | {_fmt_type(f.annotation, seen)} | {_fmt_default(f)} "
              f"| {notes.get(name, '')} |")
        w("")
    return "\n".join(out).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / "references" / "spec-schema.md")
    args = ap.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render())
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
