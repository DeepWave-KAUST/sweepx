"""Run-metadata dump: resolved config + runtime json. Verbatim from runner.py."""
from pathlib import Path

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

    # ---- 1) Resolved config (full pydantic dump, JSON-mode for paths
    # and enums → str). ``mode='json'`` makes the dict yaml.safe_dump-able.
    try:
        cfg = spec.model_dump(mode="json")
    except Exception:  # pydantic v1 fallback
        cfg = spec.dict()
    try:
        (task_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
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
        "sweep_loss", "sweep_viz",
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
