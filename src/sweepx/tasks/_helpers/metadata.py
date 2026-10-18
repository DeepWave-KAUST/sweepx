"""Run-metadata dump: resolved config + runtime json. Verbatim from runner.py."""
import re
from pathlib import Path

# The packages run_meta.json records: import name -> distribution name.
_RECORDED_PACKAGES = (
    ("sweepx.tasks", "sweep-tasks"),
    ("sweepx.nn", "sweep-nn"),
    ("sweepx.io", "sweep-io"),
    ("sweep", "sweep-solver"),
    ("sweepx.loss", "sweep-loss"),
    ("torch", "torch"),
    ("numpy", "numpy"),
)


def _toml_load(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _project_version(here: Path, dist: str) -> str | None:
    """``[project] version`` of the source tree a module was imported from: the
    first pyproject.toml at most three levels above ``here`` whose project name
    is ``dist`` (a static version only). Such a tree, on PYTHONPATH, can differ
    from the distribution installed under the same name, whose metadata is what
    the package's ``__version__`` reads."""
    for d in [here, *here.parents][:4]:
        f = d / "pyproject.toml"
        if not f.is_file():
            continue
        try:
            project = _toml_load(f).get("project", {})
        except Exception:  # noqa: BLE001
            return None
        if str(project.get("name", "")).replace("_", "-").lower() != dist:
            return None
        v = project.get("version")
        return str(v) if v else None
    return None


def _package_version(mod_name: str, dist: str) -> tuple[str | None, str | None]:
    """(the version of the code imported as ``mod_name``, the version of the
    installed distribution ``dist`` when that is a different one, else None)."""
    import importlib
    import importlib.metadata as md

    try:
        mod = importlib.import_module(mod_name)
    except Exception:  # noqa: BLE001
        return None, None
    try:
        installed = md.version(dist)
    except Exception:  # noqa: BLE001
        installed = None
    here = Path(getattr(mod, "__file__", "") or "").resolve().parent
    version = _project_version(here, dist) or getattr(mod, "__version__", None) or installed
    return version, (installed if installed is not None and installed != version else None)


def _git_meta(mod_name: str) -> dict | None:
    """Path, HEAD SHA, ``git describe --tags --always`` and dirty flag of the
    git checkout a module was imported from (best-effort; None otherwise)."""
    import importlib
    import subprocess

    try:
        mod = importlib.import_module(mod_name)
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

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.DEVNULL,
        ).decode().strip()

    try:
        sha = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except Exception:  # noqa: BLE001
        return None
    try:
        describe = git("describe", "--tags", "--always")
    except Exception:  # noqa: BLE001
        describe = None
    return {"path": str(path), "sha": sha, "describe": describe, "dirty": dirty}


# What a run dir needs to stay readable once it is archived (moved away from
# the folder it was launched from): the YAML as written, the labels it was run
# under, the inputs it read (where they really were, and a fingerprint to find
# them again), the batch script and logs of its job, the uncommitted code
# changes it ran with, and a manifest of what it wrote. All best-effort.

LABEL_PREFIX = "SWEEP_LABEL_"
QUICK_HASH_BYTES = 1 << 20      # head and tail read for an input's fingerprint
MAX_INPUTS = 200
MAX_DIFF_BYTES = 5 << 20
MAX_MANIFEST_FILES = 20_000
# config keys that name where a run writes, not what it reads
_OUTPUT_KEYS = {"output_dir", "disk_dir"}
# a variable named like a credential (SWEEP_AGENT_LLM_API_KEY, ...) is recorded
# as set, never with its value -- in run_meta.json and in the copied batch script
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE", re.I)
_SECRET_ASSIGN = re.compile(
    r"^(\s*(?:export\s+)?\w*(?:KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE)\w*=).+$", re.I | re.M)
REDACTED = "<redacted>"


def _labels_from_env(environ=None) -> dict:
    """``SWEEP_LABEL_<KEY>=value`` -> ``{"<key>": value}``: free labels a
    launcher sets for the run (``SWEEP_LABEL_LINE`` = the experiment line,
    ``SWEEP_LABEL_PROJECT``, ``SWEEP_LABEL_CAMPAIGN``, ...)."""
    import os

    env = os.environ if environ is None else environ
    return {k[len(LABEL_PREFIX):].lower(): v for k, v in sorted(env.items())
            if k.startswith(LABEL_PREFIX) and len(k) > len(LABEL_PREFIX) and v}


def _env_of_interest(environ=None) -> dict:
    import os

    env = os.environ if environ is None else environ
    keys = ["FWI_SEGY_ROOT", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH", "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV"]
    keys += sorted(k for k in env if k.startswith("SWEEP_") and not k.startswith(LABEL_PREFIX))
    return {k: (None if env.get(k) is None else REDACTED if _SECRET_NAME.search(k)
                else str(env.get(k))[:4096]) for k in keys}


def _quick_hash(path: Path) -> str | None:
    """sha256 of the size and the first / last MiB: cheap to take on a
    multi-GB file, enough to tell copies of an input apart."""
    import hashlib

    try:
        size = path.stat().st_size
        h = hashlib.sha256(str(size).encode())
        with open(path, "rb") as fh:
            h.update(fh.read(QUICK_HASH_BYTES))
            if size > QUICK_HASH_BYTES:
                fh.seek(max(size - QUICK_HASH_BYTES, QUICK_HASH_BYTES))
                h.update(fh.read(QUICK_HASH_BYTES))
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _input_files(cfg, task_dir=None) -> list[dict]:
    """The absolute paths in the resolved config that exist: where they
    really are (links resolved), and for files their size, mtime and quick
    hash. Where the run writes (``output_dir``, ``disk_dir``, anything in its
    own run dir) is not an input; another run's output it starts from is."""
    import os

    own = os.path.realpath(task_dir) if task_dir is not None else None
    found: list[dict] = []
    seen: set[str] = set()

    def walk(v, key: str) -> None:
        if len(found) >= MAX_INPUTS:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                if str(k) not in _OUTPUT_KEYS:
                    walk(x, f"{key}.{k}" if key else str(k))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(x, f"{key}[{i}]")
        elif isinstance(v, str) and v.startswith("/") and "\n" not in v and len(v) < 4096:
            if v in seen:
                return
            p = Path(v)
            try:
                if not p.exists():
                    return
                st = p.stat()
                real = os.path.realpath(v)
            except OSError:
                return
            if own and (real == own or real.startswith(own + os.sep)):
                return
            seen.add(v)
            entry = {"key": key, "path": v, "realpath": real}
            if p.is_dir():
                entry["is_dir"] = True
            else:
                entry.update(size=st.st_size, mtime=st.st_mtime, quick_sha256=_quick_hash(p))
            found.append(entry)

    walk(cfg, "")
    return found


def _copy_task_source(source: dict | None, task_dir: Path) -> dict | None:
    """``task_source.yaml``: the task file as written (comments kept), and
    what the run applied on top of it (``--override``)."""
    import hashlib

    if not source or not source.get("path"):
        return None
    src = Path(source["path"])
    info = {"path": str(src), "overrides": list(source.get("overrides") or []),
            "copied_to": None, "sha256": None}
    try:
        data = src.read_bytes()
        (task_dir / "task_source.yaml").write_bytes(data)
        info.update(copied_to="task_source.yaml", sha256=hashlib.sha256(data).hexdigest())
    except Exception as err:  # noqa: BLE001
        info["error"] = repr(err)
    return info


def _slurm_job(task_dir: Path, environ=None) -> dict | None:
    """The Slurm job this run is in: where it was submitted from, its batch
    script (copied to ``launch/slurm_<job id>.sh``, a resumed run keeps every
    job's) and its log paths, from ``scontrol`` (best-effort, a few seconds at
    most). Values assigned to credential-like names in the script are redacted."""
    import os
    import re
    import subprocess

    env = os.environ if environ is None else environ
    job = env.get("SLURM_JOB_ID")
    if not job:
        return None
    info: dict = {k.replace("SLURM_", "").lower(): env[k] for k in (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME",
        "SLURM_SUBMIT_DIR", "SLURM_SUBMIT_HOST", "SLURM_CLUSTER_NAME", "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_QOS", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST") if env.get(k)}
    name = f"slurm_{re.sub(r'[^0-9A-Za-z_.-]', '_', job)}.sh"
    try:
        r = subprocess.run(["scontrol", "write", "batch_script", job, "-"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            text, n = _SECRET_ASSIGN.subn(rf"\g<1>{REDACTED}", r.stdout)
            (task_dir / "launch").mkdir(exist_ok=True)
            (task_dir / "launch" / name).write_text(text)
            info["batch_script"] = f"launch/{name}"
            if n:
                info["batch_script_redacted"] = n
    except Exception:  # noqa: BLE001
        pass
    try:
        r = subprocess.run(["scontrol", "-o", "show", "job", job], capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0:
            for key in ("StdOut", "StdErr", "WorkDir", "Command"):
                m = re.search(rf"(?:^|\s){key}=(\S+)", r.stdout)
                if m:
                    info[key.lower()] = m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return info


# What a launch takes once: a later dump in the same process (FWI re-dumps with
# runtime extras) reuses it; a resumed run is a new process and takes its own,
# and lists the launches before it.
_ONCE: dict[str, dict] = {}


def _previous_launches(prev: dict) -> list[dict]:
    """The launches that wrote this run dir before (a resumed or re-run
    task_id), oldest first, from the run_meta.json the last one left."""
    if not prev:
        return []
    earlier = [x for x in (prev.get("previous_launches") or []) if isinstance(x, dict)]
    return earlier + [{k: prev.get(k) for k in ("started_at", "host", "argv", "source", "scheduler")}]


def _write_dirty_diffs(git: dict, task_dir: Path) -> None:
    """``provenance/<pkg>.diff`` for every checkout with uncommitted changes:
    ``git diff HEAD`` plus the untracked files' names (capped), so the code
    that ran can be rebuilt from its commit."""
    import subprocess

    for pkg, g in git.items():
        if not g or not g.get("dirty"):
            continue
        try:
            def run(*args: str) -> bytes:
                return subprocess.run(["git", "-C", g["path"], *args], capture_output=True,
                                      timeout=30).stdout
            diff = run("diff", "HEAD", "--binary")
            untracked = run("ls-files", "--others", "--exclude-standard").decode(errors="replace")
            text = diff[:MAX_DIFF_BYTES]
            if len(diff) > MAX_DIFF_BYTES:
                text += f"\n# ... diff truncated at {MAX_DIFF_BYTES} of {len(diff)} bytes\n".encode()
            if untracked.strip():
                text += ("\n# untracked files:\n" + "".join(f"#   {u}\n" for u in untracked.splitlines())).encode()
            out = task_dir / "provenance" / f"{pkg}.diff"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(text)
            g["diff_file"] = f"provenance/{pkg}.diff"
        except Exception:  # noqa: BLE001
            pass


def _write_manifest(task_dir: Path) -> None:
    """``manifest.json``: every file the run dir holds at the end (relative
    path, size), to check a copy of it is whole."""
    import json
    import os
    from datetime import datetime, timezone

    files, total, truncated = [], 0, False
    for dirpath, dirnames, filenames in os.walk(task_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, task_dir)
            if rel == "manifest.json":
                continue
            try:
                size = os.lstat(full).st_size
            except OSError:
                continue
            total += size
            if len(files) < MAX_MANIFEST_FILES:
                files.append({"path": rel, "size": size})
            else:
                truncated = True
    doc = {"written_at": datetime.now(timezone.utc).isoformat(), "n_files": len(files),
           "total_bytes": total, "truncated": truncated, "files": files}
    (task_dir / "manifest.json").write_text(json.dumps(doc, indent=1))


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
        from sweepx.tasks._helpers.dd import _dd_config
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
              "#            -m sweepx.tasks run config_resolved.yaml",
              f"#   (single-node shortcut: sweep-tasks run --dd-py {dd['py']}"
              f" --dd-px {dd['px']} config_resolved.yaml)"]
    elif td.get("initialized") and int(td.get("world_size") or 1) > 1:
        w = td.get("world_size")
        L += [f"# Shot-parallel torchrun: world_size={w}"
              f" backend={td.get('backend')} (no domain decomposition)",
              f"#   torchrun --nproc-per-node={w} -m sweepx.tasks run config_resolved.yaml"]
    else:
        L.append("# Single-process run (no torch.distributed / DD).")
    if sl:
        L.append("# SLURM: " + " ".join(
            f"{k.replace('SLURM_', '').lower()}={sl[k]}" for k in (
                "SLURM_JOB_ID", "SLURM_NNODES", "SLURM_NTASKS",
                "SLURM_GPUS_ON_NODE", "SLURM_JOB_PARTITION") if k in sl))
    L.append("# " + "=" * 68)
    return "\n".join(L) + "\n"


# Blocks whose merge reads ``model_fields_set``: writing them out with every
# field explicit turns "inherit this" into "override with the schema default",
# so the resolved config must record them EXACTLY as the author wrote them.
# (attribute on a stage, the merge that depends on it)
_SETWISE_STAGE_OVERRIDES = (
    ("boundary", "tasks.fwi_freqsel.backend_with_stage_boundary copies only "
                 "the fields the stage named"),
)


def _prune_setwise_overrides(spec, cfg):
    """Rewrite ``model_fields_set``-sensitive override blocks as written.

    ``config_resolved.yaml`` is advertised as re-runnable (the header prints
    the exact ``sweep-tasks run config_resolved.yaml`` command), so a full
    dump has to round-trip to the SAME run. For a partial-override block it
    does not: a stage that named ``storage`` alone comes back naming all eight
    boundary fields, and the seven it never wrote now override the global
    block with their schema defaults. On a production band cascade that
    silently turned an inherited ``storage_dtype: int8`` into ``fp32`` -- a 4x
    bigger boundary buffer in a config nobody had edited.

    The values dropped here are not lost: they are the global block's, which
    is dumped in full a few lines above. Only the *override* shrinks, back to
    what it actually overrides.
    """
    stages = getattr(spec, "stages", None) or ()
    cfg_stages = cfg.get("stages") if isinstance(cfg, dict) else None
    if not stages or not isinstance(cfg_stages, list):
        return cfg
    if len(cfg_stages) != len(stages):
        return cfg  # shape we do not recognise: leave the dump alone
    for stage, dumped in zip(stages, cfg_stages):
        if not isinstance(dumped, dict):
            continue
        for name, _why in _SETWISE_STAGE_OVERRIDES:
            block = dumped.get(name)
            override = getattr(stage, name, None)
            if override is None or not isinstance(block, dict):
                continue
            named = set(getattr(override, "model_fields_set", ()) or ())
            dumped[name] = {k: v for k, v in block.items() if k in named}
    return cfg


def _effective_loss_kind(spec) -> str | None:
    """The misfit the task evaluates, or ``None`` for a task without one.

    RTM's ``imaging.loss_kind`` overrides ``loss.kind`` (tasks/rtm.py). Every
    other task runs the validated ``loss.kind``, which the schema resolves to
    ``steady_gcn`` under ``source_encoding.mode='frequency_selection'``.
    """
    imaging_kind = getattr(getattr(spec, "imaging", None), "loss_kind", None)
    if imaging_kind is not None:
        return imaging_kind
    return getattr(getattr(spec, "loss", None), "kind", None)


def _dump_run_metadata(
    spec,
    task_dir,
    *,
    extras: dict | None = None,
    source: dict | None = None,
) -> None:
    """Write the resolved YAML + a runtime metadata json to ``task_dir``.

    Captures everything a reader of the run dir 6 months from now needs
    to reproduce / interpret the result:

    * ``config_resolved.yaml`` — the post-pydantic-validation spec with
      ALL fields explicit (defaults expanded), EXCEPT the partial-override
      blocks listed in ``_SETWISE_STAGE_OVERRIDES``, which stay exactly as
      written so the file re-runs as itself. Diffing this against any
      hand-written YAML shows what fields the pipeline saw; for an override
      block, read it together with the global block it overrides.
    * ``run_meta.json`` — the misfit the task evaluates (``loss_kind``),
      host, time, CUDA device, conda env path, key
      package versions (sweep-tasks, sweep-nn, sweep-io, sweep (the
      solver, sweep-solver), sweep-loss, torch, numpy) of the code that
      ran (a source tree's pyproject version, else ``__version__``),
      ``installed_versions_differ`` for a distribution installed in the
      environment that is not that code (a stale install shadowed by a
      tree on PYTHONPATH), git SHA / describe / dirty-flag for each
      package imported from a checkout (best-effort; a dirty one's changes in
      ``provenance/<pkg>.diff``), ``labels`` from ``SWEEP_LABEL_*``, the
      ``inputs`` the config names (real path, size, quick hash), the
      ``scheduler`` job (Slurm: submit dir, log paths, the batch script in
      ``launch/``), the ``previous_launches`` of a resumed run, ``source``
      (the task file, copied verbatim to ``task_source.yaml``, and the
      overrides applied), plus any caller-supplied
      ``extras`` (typically the runner's derived setup quantities like
      ``shape``, ``origin``, ``n_groups``, ``net_params``, ...).
    """
    import json
    import os
    import socket
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
    # Partial-override blocks go back to what they actually override, or this
    # file does not re-run as itself. See _prune_setwise_overrides.
    cfg = _prune_setwise_overrides(spec, cfg)
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

    # ---- 2) Package versions + git SHAs (best-effort): the code that ran.
    found = {n: _package_version(n, dist) for n, dist in _RECORDED_PACKAGES}
    versions = {n: v for n, (v, _) in found.items()}
    installed_differ = {n: inst for n, (_, inst) in found.items() if inst is not None}
    git = {}
    for n, _ in _RECORDED_PACKAGES:
        g = _git_meta(n)
        if g is not None:
            git[n] = g
    _write_dirty_diffs(git, task_dir)
    once = _ONCE.get(str(task_dir.resolve()))
    if once is None:
        try:
            prev = json.loads((task_dir / "run_meta.json").read_text())
        except Exception:  # noqa: BLE001
            prev = {}
        try:
            inputs = _input_files(cfg, task_dir)
        except Exception:  # noqa: BLE001
            inputs = []
        once = _ONCE[str(task_dir.resolve())] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source": None, "scheduler": _slurm_job(task_dir), "inputs": inputs,
            "previous_launches": _previous_launches(prev),
        }
    if source:
        once["source"] = _copy_task_source(source, task_dir)

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
        "loss_kind": _effective_loss_kind(spec),
        "started_at": once["started_at"],
        "host": socket.gethostname(),
        "user": os.environ.get("USER"),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python_executable": sys.executable,
        "argv": list(sys.argv),
        "run_dir": str(task_dir.resolve()),
        "labels": _labels_from_env(),
        "source": once["source"],
        "scheduler": once["scheduler"],
        "previous_launches": once["previous_launches"],
        "inputs": once["inputs"],
        "cuda": cuda_info,
        "distributed": launch,
        "package_versions": versions,
        "installed_versions_differ": installed_differ,
        "git_repos": git,
        "env_vars_of_interest": _env_of_interest(),
    }
    if extras:
        meta["runtime"] = extras
    (task_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
