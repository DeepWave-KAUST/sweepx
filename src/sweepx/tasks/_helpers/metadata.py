"""Run-metadata dump: resolved config + runtime json. Verbatim from runner.py."""
from pathlib import Path


def _capture_launch_env() -> dict:
    """Best-effort snapshot of the distributed / DD / launcher environment.

    ``config_resolved.yaml`` records *what* the pipeline computed; this
    records *how the run was launched* — the piece a reader needs to
    reproduce a multi-GPU run elsewhere (the domain-decomposition mesh, the
    torch.distributed world, and the SLURM allocation). None of it lives in
    the pydantic spec because it is set at launch time via environment
    variables (``SWEEP_DD_*`` / torchrun / SLURM), so without this it never
    reaches the run dir.

    Everything is optional: a plain single-process run yields
    ``{"dd": {"enabled": False, ...}, "torch_distributed": {"initialized": False}}``
    with no ``torchrun_env`` / ``slurm`` keys.
    """
    import os

    info: dict = {}

    # --- DD mesh: sweep's domain-decomposition env. Single source of truth
    #     is dd._dd_config — the exact reader the runner uses at launch, so
    #     this stays correct if that env contract ever changes. ---
    try:
        from sweep_tasks._helpers.dd import _dd_config
        dd_on, py, px, render_chunk = _dd_config()
    except Exception:  # noqa: BLE001 -- fall back to raw env
        dd_on = os.environ.get("SWEEP_DD_ENABLE") == "1"
        py = int(os.environ.get("SWEEP_DD_PY", "1") or 1)
        px = int(os.environ.get("SWEEP_DD_PX", "1") or 1)
        render_chunk = int(os.environ.get("SWEEP_DD_RENDER_CHUNK", "8") or 8)
    info["dd"] = {
        "enabled": bool(dd_on), "py": int(py), "px": int(px),
        "mesh": int(py) * int(px), "render_chunk": int(render_chunk),
    }

    # --- torch.distributed world (authoritative rank / world_size). ---
    td: dict = {"initialized": False}
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            td = {
                "initialized": True, "backend": str(_dist.get_backend()),
                "world_size": int(_dist.get_world_size()),
                "rank": int(_dist.get_rank()),
            }
    except Exception:  # noqa: BLE001
        pass
    info["torch_distributed"] = td

    # --- torchrun / torchelastic env: how the world was spawned. ---
    tr = {k: os.environ[k] for k in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
    ) if os.environ.get(k) is not None}
    if tr:
        info["torchrun_env"] = tr

    # --- SLURM allocation: the batch resources (nodes / gpus / mem). ---
    sl = {k: os.environ[k] for k in (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME", "SLURM_NNODES", "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE", "SLURM_GRES", "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
    ) if os.environ.get(k) is not None}
    if sl:
        info["slurm"] = sl

    return info


def _launch_env_header(info: dict) -> str:
    """Render ``_capture_launch_env`` output as a YAML comment block for the
    top of ``config_resolved.yaml``.

    Comments are ignored on reload, so the file stays a valid, runnable spec
    (the schema is ``extra='forbid'`` — a real ``distributed:`` key would make
    ``sweep-tasks run config_resolved.yaml`` fail to load). The machine-readable
    copy lives in ``run_meta.json['distributed']``.
    """
    dd = info.get("dd", {})
    td = info.get("torch_distributed", {})
    sl = info.get("slurm", {})
    L = ["# " + "=" * 68,
         "# Run environment — CAPTURED at runtime; NOT part of the spec.",
         "# (YAML comment, ignored on reload; machine-readable copy in",
         "#  run_meta.json['distributed'].) Needed to reproduce a multi-GPU run.",
         "# " + "-" * 68]
    if dd.get("enabled"):
        world = td.get("world_size") or dd.get("mesh")
        L.append(f"# Domain decomposition (DD): PY={dd['py']} x PX={dd['px']}"
                 f" = {dd['mesh']} tiles   render_chunk={dd['render_chunk']}")
        L.append(f"# torch.distributed: world_size={world}"
                 f" backend={td.get('backend')}")
        L += ["# Reproduce the launch:",
              f"#   SWEEP_DD_ENABLE=1 SWEEP_DD_PY={dd['py']} SWEEP_DD_PX={dd['px']} \\",
              "#   torchrun --nnodes=<N> --nproc-per-node=<gpus-per-node> \\",
              "#            -m sweep_tasks run config_resolved.yaml",
              f"#   (single-node shortcut: sweep-tasks run --dd-py {dd['py']}"
              f" --dd-px {dd['px']} config_resolved.yaml)"]
    elif td.get("initialized") and int(td.get("world_size") or 1) > 1:
        w = td.get("world_size")
        L += [f"# Shot-parallel torchrun: world_size={w}"
              f" backend={td.get('backend')} (no domain decomposition)",
              f"#   torchrun --nproc-per-node={w} -m sweep_tasks run config_resolved.yaml"]
    else:
        L.append("# Single-process run (no torch.distributed / DD).")
    if sl:
        L.append("# SLURM: " + " ".join(
            f"{k.replace('SLURM_', '').lower()}={sl[k]}" for k in (
                "SLURM_JOB_ID", "SLURM_NNODES", "SLURM_NTASKS",
                "SLURM_GPUS_ON_NODE", "SLURM_JOB_PARTITION") if k in sl))
    L.append("# " + "=" * 68)
    return "\n".join(L) + "\n"


def _dump_run_metadata(
    spec,
    task_dir,
    *,
    extras: dict | None = None,
) -> None:
    """Write the resolved YAML + a runtime metadata json to ``task_dir``.

    Captures everything a reader of the run dir 6 months from now needs
    to reproduce / interpret the result:

    * ``config_resolved.yaml`` — the post-pydantic-validation spec with
      ALL fields explicit (defaults expanded). Diffing this against any
      hand-written YAML shows exactly what fields the pipeline saw.
    * ``run_meta.json`` — host, time, CUDA device, conda env path, key
      package versions (sweep-tasks, sweep-nn, sweep-io, sweep,
      sweep-loss, sweep-preproc, torch, numpy), git SHA / dirty-flag
      for each editable package (best-effort), plus any caller-supplied
      ``extras`` (typically the runner's derived setup quantities like
      ``shape``, ``origin``, ``n_groups``, ``net_params``, ...).
    """
    import json
    import os
    import socket
    import subprocess
    import sys
    from datetime import datetime, timezone

    import yaml

    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0) Launch environment (DD mesh / torch.distributed world / SLURM).
    # Captured once, reused for the config header AND run_meta.json below.
    launch = _capture_launch_env()

    # ---- 1) Resolved config (full pydantic dump, JSON-mode for paths
    # and enums → str). ``mode='json'`` makes the dict yaml.safe_dump-able.
    # Prepend the launch env as a YAML comment header so this single file
    # documents the FULL run config (spec + how it was launched) — comments
    # are ignored on reload, so the file stays a runnable ``extra='forbid'`` spec.
    try:
        cfg = spec.model_dump(mode="json")
    except Exception:  # pydantic v1 fallback
        cfg = spec.dict()
    try:
        (task_dir / "config_resolved.yaml").write_text(
            _launch_env_header(launch)
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )
    except Exception as err:  # noqa: BLE001
        # Yaml dump can hit unrepresentable objects; fall back to json.
        (task_dir / "config_resolved.json").write_text(
            json.dumps(cfg, default=str, indent=2)
        )
        print(f"[run-meta] yaml dump failed ({err}); wrote config_resolved.json")

    # ---- 2) Package versions + git SHAs (best-effort).
    def _pkg_version(name: str) -> str | None:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            return None

    def _git_meta(pkg_name: str) -> dict | None:
        try:
            mod = __import__(pkg_name)
        except Exception:  # noqa: BLE001
            return None
        path = Path(getattr(mod, "__file__", "") or "").resolve().parent
        for _ in range(6):  # walk up looking for .git
            if (path / ".git").exists():
                break
            if path.parent == path:
                return None
            path = path.parent
        else:
            return None
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            dirty = bool(subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip())
            return {"path": str(path), "sha": sha, "dirty": dirty}
        except Exception:  # noqa: BLE001
            return None

    pkgs = [
        "sweep_tasks", "sweep_nn", "sweep_io", "sweep",
        "sweep_loss",
        "torch", "numpy",
    ]
    versions = {n: _pkg_version(n) for n in pkgs}
    git = {n: _git_meta(n) for n in pkgs if _git_meta(n) is not None}

    # ---- 3) Host + CUDA snapshot.
    cuda_info: dict = {"available": False}
    try:
        import torch
        cuda_info["available"] = bool(torch.cuda.is_available())
        if cuda_info["available"]:
            cuda_info["device_count"] = int(torch.cuda.device_count())
            cuda_info["devices"] = [
                {
                    "index": i, "name": torch.cuda.get_device_name(i),
                    "total_mem_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 2,
                    ),
                }
                for i in range(int(torch.cuda.device_count()))
            ]
            cuda_info["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "task_id": getattr(spec, "task_id", None),
        "task_type": getattr(spec, "task_type", None),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "argv": list(sys.argv),
        "cuda": cuda_info,
        "distributed": launch,
        "package_versions": versions,
        "git_repos": git,
        "env_vars_of_interest": {
            k: os.environ.get(k) for k in (
                "FWI_SEGY_ROOT", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )
        },
    }
    if extras:
        meta["runtime"] = extras
    (task_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
