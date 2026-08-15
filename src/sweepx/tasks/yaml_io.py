"""YAML <-> TaskSpec round-trip and `new_template` factory.

Relative paths inside the YAML are resolved against the YAML file's parent
directory at load time. ``$VAR`` / ``${VAR}`` inside path-like values is
expanded against the current process environment first (so a portable
template can be authored with ``path: ${DATA_ROOT}/vp_true.npy``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from sweep_tasks.schemas import TaskSpec

_task_adapter: TypeAdapter[TaskSpec] = TypeAdapter(TaskSpec)


def _resolve_paths(value: Any, base: Path) -> Any:
    """Recursively resolve any string that looks like a path key to absolute.

    We can't know which strings are paths without inspecting the schema, so
    instead we walk the raw dict before validation and treat any key named
    `path` or `npy_path` whose value is a non-absolute string as relative to
    `base`. ``$VAR`` / ``${VAR}`` is expanded against the environment
    before relative-to-absolute resolution.
    """

    if isinstance(value, dict):
        return {
            k: (
                _resolve_one(v, base)
                if k in {"path", "npy_path", "disk_dir", "output_dir",
                         "sources_file", "receivers_file"}
                else _resolve_paths(v, base)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_resolve_paths(v, base) for v in value]
    return value


def _resolve_one(v: Any, base: Path) -> Any:
    if not isinstance(v, str):
        return v
    # Expand $VAR / ${VAR} against the current environment first so templates
    # like ``${DATA_ROOT}/vp_true.npy`` work without pre-`envsubst`.
    # Unset variables are left as-is (matches expandvars semantics) — they
    # then fall through to the relative-path branch, which gives a clear
    # FileNotFoundError downstream rather than silently dropping the prefix.
    v = os.path.expandvars(v)
    p = Path(v)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())


def load_task(path: str | Path) -> TaskSpec:
    """Load and validate a task spec from a YAML file.

    Raises pydantic.ValidationError on schema errors.
    """

    path = Path(path).resolve()
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Task YAML must be a mapping at the top level, got {type(raw).__name__}.")
    return load_task_from_dict(raw, base_dir=path.parent)


def load_task_from_dict(raw: dict, *, base_dir: Path | str | None = None) -> TaskSpec:
    """Validate a task spec from an already-parsed dict.

    Used by the CLI to support ``--override key=value`` flags: the raw
    YAML is loaded into a dict, dotted-key overrides are applied via
    :func:`sweep_tasks.runtime.config.apply_overrides`, then this function does
    path resolution + pydantic validation against the same base_dir the
    YAML lived in.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"Task spec must be a mapping at the top level, got "
            f"{type(raw).__name__}."
        )
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    resolved = _resolve_paths(raw, base)
    return _task_adapter.validate_python(resolved)


def dump_task(spec: TaskSpec, path: str | Path) -> Path:
    """Serialise a task spec back to YAML."""

    path = Path(path)
    data = _task_adapter.dump_python(spec, mode="python")
    # paths become Path objects after dump_python(mode="python"); convert to str so yaml is plain
    data = _stringify_paths(data)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return path


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(v) for v in value]
    if isinstance(value, tuple):
        return [_stringify_paths(v) for v in value]
    return value


_TEMPLATES: dict[str, dict[str, Any]] = {
    "introspect": {
        "task_type": "introspect",
        "action": "list_equations",
    },
    "wavefield": {
        "task_type": "wavefield",
        "output_dir": "./sweep_runs",
        "device": "auto",
        "grid": {"dh": 5.0},
        "time": {"dt": 0.001, "nt": 500},
        "wavelet": {"kind": "ricker", "fm": 18.0, "delay": 0.12, "scale": 1.0e6},
        "geometry": {
            "kind": "line",
            "sources": {"step": 1, "depth": 4, "start": 90, "stop": 91},
            "receivers": {"step": 1, "depth": 8, "start": 8, "stop": None},
        },
        "physics": {
            "equation": "Acoustic",
            "spatial_order": 8,
            "abcn": 20,
            "free_surface": True,
            "pml_type": "cpmlr",
            "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "models": [
            {"name": "vp", "constant": 2200.0, "shape": [120, 180]},
        ],
        "snapshot_times": [150, 250, 380],
        "plot": True,
    },
}


# Task types whose ``new_template`` / ``sweep-tasks new`` output is sourced from
# the bundled annotated ``templates/<task_type>.yaml`` reference — the SAME file
# `sweep-tasks init` ships, so the two commands never drift. This now covers
# every task type in the registry, so ``sweep-tasks init <any task_type>``
# always yields an annotated, runnable starting point.
_FILE_TEMPLATES: tuple[str, ...] = (
    "introspect", "forward", "wavefield", "fwi", "rtm", "lsrtm",
)


def _read_bundled_template(task_type: str) -> str:
    """Raw text of the bundled ``templates/<task_type>.yaml`` annotated reference."""
    import importlib.resources as ir

    return ir.files("sweep_tasks").joinpath(f"templates/{task_type}.yaml").read_text()


def new_template(
    task_type: str,
    equation: str | None = None,
    *,
    backend: str = "eager",
    memory: str = "full",
    storage: str = "gpu",
    compile: bool = False,
) -> dict[str, Any]:
    """Return a YAML-ready dict template for the given task type.

    forward / fwi / rtm / lsrtm are parsed from the bundled annotated
    ``templates/<task_type>.yaml`` reference (the same file ``sweep-tasks init``
    ships), so this dict never drifts from it. Comments are dropped by the dict
    path — ``sweep-tasks new`` serves the raw annotated text instead. introspect
    and wavefield have no bundled file and come from the built-in ``_TEMPLATES``.

    Optional kwargs scaffold the backend block:
      backend: 'eager' | 'c'                    (default: 'eager')
      memory:  'full' | 'boundary' | 'ckpt'     (only valid when backend='c'; default: 'full')
      storage: 'gpu' | 'cpu' | 'disk'           (relevant when memory in {boundary, ckpt}; default: 'gpu')
      compile: bool                             (only valid when backend='eager'; default: False)

    The introspect template has no backend block, so backend kwargs are ignored there.
    """

    known = set(_TEMPLATES) | set(_FILE_TEMPLATES)
    if task_type not in known:
        raise KeyError(
            f"Unknown task_type '{task_type}'. Known: {sorted(known)}"
        )
    _validate_backend_kwargs(backend, memory, storage, compile)

    if task_type in _FILE_TEMPLATES:
        # Single source of truth: parse the bundled annotated reference (the
        # same file `sweep-tasks init` ships). Comments are dropped by the dict
        # path; `sweep-tasks new` serves the raw text so they survive there.
        import yaml as _yaml

        template = _yaml.safe_load(_read_bundled_template(task_type))
    else:
        template = _deep_copy_dict(_TEMPLATES[task_type])
    if equation is not None and "physics" in template:
        template["physics"]["equation"] = equation
    if "backend" in template:
        template["backend"] = _build_backend_block(backend, memory, storage, compile)
    return template


def _validate_backend_kwargs(backend: str, memory: str, storage: str, compile: bool) -> None:
    if backend not in {"eager", "c"}:
        raise ValueError(f"backend must be 'eager' or 'c', got '{backend}'.")
    if memory not in {"full", "boundary", "ckpt"}:
        raise ValueError(f"memory must be 'full' | 'boundary' | 'ckpt', got '{memory}'.")
    if storage not in {"gpu", "cpu", "disk"}:
        raise ValueError(f"storage must be 'gpu' | 'cpu' | 'disk', got '{storage}'.")
    if backend == "eager":
        if memory != "full":
            raise ValueError(
                f"--memory '{memory}' is only valid with --backend c."
            )
    if backend == "c" and compile:
        raise ValueError("--compile is only valid with --backend eager.")
    if memory == "full" and storage != "gpu":
        raise ValueError(
            f"--storage '{storage}' has no effect when --memory full."
        )
    if memory == "ckpt" and storage == "disk":
        raise ValueError("CkptOptions.storage must be 'gpu' or 'cpu' (disk is not supported).")


def _build_backend_block(backend: str, memory: str, storage: str, compile: bool) -> dict[str, Any]:
    if backend == "eager":
        block: dict[str, Any] = {"impl": "eager", "use_ckpt": False}
        if compile:
            block["eager_options"] = {
                "use_compile": True,
                "compile_mode": "default",
                "compile_dynamic": False,
                "compile_fullgraph": False,
            }
        return block

    # backend == "c"
    block = {"impl": "c", "cuda_options": {"memory": None}}
    if memory == "full":
        return block
    if memory == "boundary":
        boundary_block: dict[str, Any] = {"storage": storage}
        if storage == "cpu":
            boundary_block["transfer_interval"] = 10
            boundary_block["pinned_memory"] = True
        elif storage == "disk":
            boundary_block["disk_dir"] = "/tmp/bs"
            boundary_block["transfer_interval"] = 8
            boundary_block["ring_buffers"] = 3
            boundary_block["disk_async_read"] = True
        block["cuda_options"]["memory"] = {
            "strategy": "boundary",
            "boundary": boundary_block,
        }
    elif memory == "ckpt":
        ckpt_block: dict[str, Any] = {"mode": "chunk", "chunks": 64, "storage": storage}
        if storage == "cpu":
            ckpt_block["pinned_memory"] = True
        block["cuda_options"]["memory"] = {
            "strategy": "ckpt",
            "ckpt": ckpt_block,
        }
    return block


def _deep_copy_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_dict(v) for v in value]
    return value
