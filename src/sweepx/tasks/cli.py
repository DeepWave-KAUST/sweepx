"""`sweep-tasks` CLI — `run`, `new`, `tasks {list,status,logs}`.

Mirrors what used to live in `sweep.cli` (subcommands `run/new/tasks`)
before the task layer was split into its own package. The engine-level
introspection commands (`sweep list equations`, `sweep show <Eq>`) stay
in `sweep` itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _merge_yaml_into_args(args, parser: argparse.ArgumentParser,
                           section_name: str) -> None:
    """Apply a YAML config file's section onto an argparse Namespace.

    Lets long-argument subcommands accept ``--config FILE`` instead of a
    forest of CLI flags. The YAML is expected to be a mapping with one
    section per subcommand keyed by the subcommand name in snake_case,
    plus an optional ``common:`` section for shared defaults::

        # viking.yaml
        common:
          plan: /path/to/viking_csg_plan.npz
        analyze_wavelet:
          out_dir: /path/to/wavelet/direct_batch
          shot_start: 1
          shot_stop: 1001
          nearest_receivers: 4
          ...
        estimate_wavelet:
          out_dir: /path/to/wavelet/siren_pipeline
          initial: /path/to/wavelet/direct_batch/average/average_direct_wavelet.npz
          tmax_s: 6.0
          ...

    YAML key names are argparse ``dest`` values (so ``--shot-start`` maps to
    ``shot_start``). ``$VAR`` / ``${VAR}`` in string values are expanded
    against the current environment.

    Precedence (lowest → highest): argparse default → YAML ``common:`` →
    YAML ``<section_name>:`` → CLI flag. Unknown YAML keys are silently
    ignored so the same file can carry settings for multiple subcommands.
    """
    cfg_path = getattr(args, "config", None)
    if not cfg_path:
        return

    import yaml as _yaml

    try:
        with open(cfg_path) as fh:
            data = _yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print(f"error: --config file not found: {cfg_path}")
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 — yaml errors / IO errors
        print(f"error: failed to read --config {cfg_path}: {exc}")
        sys.exit(2)
    if not isinstance(data, dict):
        print(f"error: --config {cfg_path} top-level YAML must be a mapping")
        sys.exit(2)
    common = data.get("common") or {}
    section = data.get(section_name) or {}
    if not isinstance(common, dict):
        print(f"error: --config {cfg_path}: 'common' must be a mapping")
        sys.exit(2)
    if not isinstance(section, dict):
        print(f"error: --config {cfg_path}: {section_name!r} must be a mapping")
        sys.exit(2)
    merged = {**common, **section}
    if not merged:
        return  # Nothing to apply.

    # Build a {dest: default} map by reading parser actions. We need this
    # to tell whether a CLI value differs from its default (= user
    # overrode it) or matches the default (= we should accept YAML's
    # value).
    defaults = {}
    for act in parser._actions:
        if act.dest in ("help", "config") or act.dest == argparse.SUPPRESS:
            continue
        defaults[act.dest] = act.default

    for key, raw_val in merged.items():
        if key not in defaults:
            # Silently ignore — the YAML may be shared with other subcommands
            # that recognise this key. We can't bail out on unknown keys.
            continue
        # Only overwrite if the user didn't pass an explicit CLI value.
        current = getattr(args, key, defaults[key])
        if current != defaults[key]:
            continue  # CLI wins.
        # Expand ${ENV} / $ENV in string values for convenience.
        if isinstance(raw_val, str):
            raw_val = os.path.expandvars(raw_val)
        setattr(args, key, raw_val)


def _running_under_torchrun() -> bool:
    """Return True iff the current process was spawned by torchrun / torch elastic.

    torchrun (a.k.a. ``torch.distributed.run`` / ``torchelastic``) sets
    ``LOCAL_RANK`` plus ``TORCHELASTIC_RUN_ID`` in every worker's env.
    Either alone is enough to identify a torchrun child, so we accept
    the union.
    """
    import os

    return bool(os.environ.get("TORCHELASTIC_RUN_ID")
                or os.environ.get("LOCAL_RANK"))


def _maybe_reexec_under_torchrun(args) -> None:
    """If ``--nproc-per-node`` is set and we're not yet under torchrun, re-exec.

    Lets users write ``sweep-tasks run --nproc-per-node 4 foo.yaml`` as
    a shortcut for ``torchrun --standalone --nproc_per_node=4 -m
    sweep_tasks run foo.yaml``. The explicit ``torchrun`` form keeps
    working unchanged: when the current process IS already a torchrun
    worker (detected via ``LOCAL_RANK`` / ``TORCHELASTIC_RUN_ID``), this
    short-circuits and we fall through to the real ``TaskRunner`` call.

    ``--nproc-per-node 1`` is treated identically to "no flag" — the
    runner happily initialises a single-rank process group inside one
    interpreter, so there's no reason to spawn torchrun for one worker.
    """
    import os
    import sys

    # --dd-py/--dd-px shortcut: populate the DD env the runner's _dd_config
    # reads, and auto-derive --nproc-per-node = py*px so the process count
    # always matches the mesh (else the runner raises SWEEP_DD_PY*PX != world).
    py = int(getattr(args, "dd_py", 1) or 1)
    px = int(getattr(args, "dd_px", 1) or 1)
    if py * px > 1:
        os.environ["SWEEP_DD_ENABLE"] = "1"
        os.environ["SWEEP_DD_PY"] = str(py)
        os.environ["SWEEP_DD_PX"] = str(px)
        n_dd = py * px
        n_cli = int(getattr(args, "nproc_per_node", 1) or 1)
        if n_cli <= 1:
            args.nproc_per_node = n_dd
        elif n_cli != n_dd:
            raise SystemExit(
                f"--nproc-per-node {n_cli} conflicts with --dd-py*--dd-px = "
                f"{n_dd}; omit --nproc-per-node or make them match")

    n = int(getattr(args, "nproc_per_node", 1) or 1)
    if n <= 1:
        return
    if _running_under_torchrun():
        return
    # Strip the --nproc-per-node argument from sys.argv before re-exec so
    # the child sees a clean argv and doesn't recurse.
    argv = []
    skip_next = False
    for tok in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--nproc-per-node", "--nproc_per_node"):
            skip_next = True
            continue
        if tok.startswith("--nproc-per-node=") or tok.startswith("--nproc_per_node="):
            continue
        argv.append(tok)
    # argv[0] was the sweep-tasks console script; replace with `-m sweep_tasks`
    # so we don't rely on the entrypoint being on PATH in every torchrun env.
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={n}",
        "-m", "sweep_tasks",
        *argv[1:],
    ]
    print(f"[sweep-tasks] re-exec under torchrun --nproc_per_node={n}: "
          f"{' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)
    raise RuntimeError(f"os.execvp returned for cmd={cmd!r}; "
                       f"torch.distributed.run not importable?")


def _cmd_run(args) -> int:
    import yaml

    from sweep_tasks import TaskRunner, load_task, load_task_from_dict

    # If the user passed --nproc-per-node N (>1) and we're not yet under
    # torchrun, re-exec the same command under `torchrun --standalone
    # --nproc_per_node=N`. Single-rank (--nproc-per-node 1, the default)
    # stays in-process — torchrun has noticeable startup overhead.
    _maybe_reexec_under_torchrun(args)

    overrides = list(getattr(args, "override", None) or [])
    # ``--resume`` / ``--no-resume`` are syntactic sugar for
    # ``--override resume=true|false``. Only forwarded when the user
    # actually passed the flag (default is ``None``), so the YAML's
    # own ``resume:`` value still wins when neither is set.
    resume_flag = getattr(args, "resume", None)
    if resume_flag is True:
        overrides.append("resume=true")
    elif resume_flag is False:
        overrides.append("resume=false")
    if overrides:
        from sweep_tasks.runtime.config import apply_overrides

        task_path = Path(args.task_file).resolve()
        with task_path.open("r") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            print(f"error: {task_path} top-level YAML must be a mapping.")
            return 2
        try:
            apply_overrides(raw, overrides)
        except ValueError as exc:
            print(f"error: invalid --override: {exc}")
            return 2
        spec = load_task_from_dict(raw, base_dir=task_path.parent)
        print(
            f"[run] applied {len(overrides)} override(s): "
            + ", ".join(overrides)
        )
    else:
        spec = load_task(args.task_file)
    result = TaskRunner().run(spec)
    print(f"\nTask: {result.status.task_id} | state={result.status.state}")
    print(f"Directory: {result.task_dir}")
    if result.status.state == "failed":
        print(f"Error: {result.status.error}")
        return 1
    if result.status.summary:
        print("Summary:", json.dumps(result.status.summary, indent=2, default=str))
    return 0


def _list_templates() -> list[str]:
    """Return the names of bundled --config YAML templates (without .yaml)."""
    try:
        import importlib.resources as ir
        anchor = ir.files("sweep_tasks").joinpath("templates")
        return sorted(
            p.name[:-len(".yaml")]
            for p in anchor.iterdir()
            if p.is_file() and p.name.endswith(".yaml")
        )
    except Exception:
        return []


def _cmd_init(args) -> int:
    """``sweep-tasks init`` — emit a starter --config YAML for a dataset.

    Templates live under ``src/sweep_tasks/templates/`` and ship with the
    package, so they're available after ``pip install sweep-tasks``. Each
    template has one section per sweep-tasks subcommand
    (``build_index:``, ``analyze_wavelet:``, ``estimate_wavelet:``, …)
    plus a ``common:`` block, so the resulting file drives the whole
    pipeline via ``sweep-tasks <cmd> --config <file>``.
    """
    import importlib.resources as ir

    available = _list_templates()
    if getattr(args, "list", False):
        if not available:
            print("(no bundled templates found)")
            return 0
        print("Available templates (use as `sweep-tasks init <name>`):")
        for name in available:
            print(f"  - {name}")
        return 0

    template = args.template or "viking"
    if template not in available:
        if available:
            print(
                f"error: unknown template {template!r}. "
                f"Available: {', '.join(available)}. "
                f"Run `sweep-tasks init --list` to see them."
            )
        else:
            print(
                f"error: no bundled templates found and "
                f"{template!r} is not a stdlib resource."
            )
        return 2

    text = ir.files("sweep_tasks").joinpath(f"templates/{template}.yaml").read_text()

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        if out_path.exists() and not args.force:
            print(
                f"error: {out_path} already exists. "
                f"Pass --force to overwrite, or write to a different path."
            )
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"[init] wrote {template!r} template -> {out_path}")
        # Tailored "next steps" hint per template type.
        if template in ("fwi", "rtm"):
            print(
                f"[init] next: edit paths inside the file, then run:\n"
                f"       sweep-tasks run {out_path}\n"
                f"       # multi-GPU: append --nproc-per-node N"
            )
        else:
            print(
                f"[init] next: edit paths inside (especially $VIKING_HOME / "
                f"$HOME), then run:\n"
                f"       sweep-tasks build-index --config {out_path}\n"
                f"       sweep-tasks build-plan  --config {out_path}\n"
                f"       sweep-tasks analyze-wavelet  --config {out_path}\n"
                f"       sweep-tasks estimate-wavelet --config {out_path}"
            )
    else:
        sys.stdout.write(text)
    return 0


def _cmd_new(args) -> int:
    import yaml

    from sweep_tasks import new_template

    try:
        template = new_template(
            args.task_type,
            equation=args.equation,
            backend=args.backend,
            memory=args.memory,
            storage=args.storage,
            compile=args.compile,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    text = yaml.safe_dump(template, sort_keys=False)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Wrote template to {args.output}")
    else:
        print(text)
    return 0


def _resolve_task_dir(output_dir: str, task_id: str) -> Path:
    candidate = Path(output_dir).expanduser() / task_id
    if not candidate.exists():
        raise SystemExit(f"No task '{task_id}' under {output_dir}")
    return candidate


def _cmd_tasks_list(args) -> int:
    root = Path(args.output_dir).expanduser()
    if not root.exists():
        print(f"No task directory at {root}")
        return 0
    rows = []
    for child in sorted(root.iterdir()):
        status_path = child / "status.json"
        if not status_path.exists():
            continue
        try:
            data = json.loads(status_path.read_text())
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "task_id": data.get("task_id", child.name),
            "task_type": data.get("task_type", "?"),
            "state": data.get("state", "?"),
            "started_at": data.get("started_at", ""),
        })
    if not rows:
        print(f"No tasks recorded in {root}")
        return 0
    width = max(len(r["task_id"]) for r in rows)
    print(f"{'task_id'.ljust(width)}  state     task_type  started_at")
    print(f"{'-' * width}  --------  ---------  -------------------")
    for r in rows:
        print(f"{r['task_id'].ljust(width)}  {r['state']:<8}  {r['task_type']:<9}  {r['started_at']}")
    return 0


def _cmd_tasks_status(args) -> int:
    task_dir = _resolve_task_dir(args.output_dir, args.task_id)
    print((task_dir / "status.json").read_text())
    return 0


def _cmd_tasks_logs(args) -> int:
    task_dir = _resolve_task_dir(args.output_dir, args.task_id)
    # Phase 1: no stdout capture yet; status.json is the script-friendly handle.
    print((task_dir / "status.json").read_text())
    return 0


def _detect_mpi(force_mpi: bool) -> tuple[bool, int, bool]:
    """Returns (is_root, mpi_size, use_mpi). Honors --mpi or auto-detects mpiexec."""
    is_root, mpi_size, use_mpi = True, 1, False
    if force_mpi:
        try:
            from mpi4py import MPI  # type: ignore
            comm = MPI.COMM_WORLD
            is_root = (comm.Get_rank() == 0)
            mpi_size = comm.Get_size()
            use_mpi = True
        except ImportError:
            print("error: --mpi requires mpi4py (pip install mpi4py)")
            raise SystemExit(2)
    else:
        try:
            from mpi4py import MPI  # type: ignore
            comm = MPI.COMM_WORLD
            if comm.Get_size() > 1:
                is_root = (comm.Get_rank() == 0)
                mpi_size = comm.Get_size()
                use_mpi = True
        except ImportError:
            pass
    return is_root, mpi_size, use_mpi


def _resolve_segy_paths(segy_root: Path, glob: str, files_from: str | None,
                        is_root: bool) -> list[Path] | None:
    """Resolve a list of SEG-Y file paths from --segy-root + (--glob | --files-from).

    Returns None on error (and prints an error from rank 0).
    """
    if files_from:
        list_path = Path(files_from).expanduser().resolve()
        if not list_path.is_file():
            if is_root:
                print(f"error: --files-from {list_path} not found")
            return None
        raw_lines = [ln.strip() for ln in list_path.read_text().splitlines()]
        raw_lines = [ln for ln in raw_lines if ln and not ln.startswith("#")]
        paths: list[Path] = []
        for ln in raw_lines:
            p = Path(ln)
            if not p.is_absolute():
                p = segy_root / p
            paths.append(p.resolve())
        paths.sort()
        if not paths:
            if is_root:
                print(f"error: --files-from {list_path} produced no paths")
            return None
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            if is_root:
                print(f"error: --files-from refers to {len(missing)} missing "
                      f"files (sample: {missing[:3]})")
            return None
        return paths
    paths = sorted(segy_root.glob(glob))
    if not paths:
        if is_root:
            print(f"error: no SEG-Y files matched '{glob}' under {segy_root}")
        return None
    return paths


def _parse_byte_map_args(args, is_root: bool) -> dict | None:
    """Build a byte_map dict from --byte-map + --source-z-byte + --receiver-z-byte.

    Returns the dict (possibly empty) or None on parse error.
    """
    byte_map: dict[str, int] = {}
    for kv in (getattr(args, "byte_map", None) or []):
        if "=" not in kv:
            print(f"error: --byte-map expects KEY=BYTE; got {kv!r}")
            return None
        k, v = kv.split("=", 1)
        try:
            byte_map[k.strip()] = int(v.strip())
        except ValueError:
            print(f"error: --byte-map BYTE must be an int; got {v!r}")
            return None
    if getattr(args, "source_z_byte", None) is not None:
        byte_map.setdefault("source_depth", int(args.source_z_byte))
        if getattr(args, "source_depth_m", None) is not None and is_root:
            print(f"WARN: --source-z-byte={args.source_z_byte} is redundant "
                  f"when --source-depth-m={args.source_depth_m} is set (constant "
                  "override wins; per-trace byte read is unused).")
    if getattr(args, "receiver_z_byte", None) is not None:
        byte_map.setdefault("receiver_depth", int(args.receiver_z_byte))
        if getattr(args, "receiver_depth_m", None) is not None and is_root:
            print(f"WARN: --receiver-z-byte={args.receiver_z_byte} is redundant "
                  f"when --receiver-depth-m={args.receiver_depth_m} is set.")
    return byte_map


_MPI_CHILD_ENV_VARS = (
    "OMPI_COMM_WORLD_SIZE",   # OpenMPI
    "OMPI_COMM_WORLD_RANK",
    "PMI_SIZE",               # MPICH / Intel MPI / SLURM PMI
    "PMI_RANK",
    "MPI_LOCALRANKID",        # Intel MPI
)


def _running_under_mpi() -> bool:
    """True when the current process was spawned by mpiexec / srun."""
    import os
    return any(k in os.environ for k in _MPI_CHILD_ENV_VARS)


def _build_mpi_reexec_cmd(n: int, argv: list[str]) -> list[str]:
    """Build the ``mpiexec -n N`` re-exec command for sweep-tasks.

    Strips ``-n`` / ``--mpi-ranks`` (and any ``-n=K`` / ``--mpi-ranks=K``
    equals form) from ``argv`` so child ranks don't recurse, and appends
    ``--mpi`` if missing so the build-index branch takes the MPI path.

    ``argv[0]`` (the sweep-tasks entry script, absolute path when invoked
    via the console-script entry point) is launched directly by mpiexec.
    """
    # Only walk argv[1:] — argv[0] is the program name and will be
    # passed separately to mpiexec; including it twice makes mpiexec
    # spawn `sweep-tasks /path/sweep-tasks build-index ...` and argparse
    # then treats the second path as a positional `command` (invalid).
    cleaned: list[str] = []
    skip_next = False
    for a in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ("-n", "--mpi-ranks"):
            skip_next = True
            continue
        if a.startswith("-n=") or a.startswith("--mpi-ranks="):
            continue
        cleaned.append(a)
    if "--mpi" not in cleaned:
        cleaned.append("--mpi")
    return ["mpiexec", "-n", str(int(n)), argv[0]] + cleaned


def _maybe_reexec_under_mpi(args) -> None:
    """If ``--mpi-ranks N`` is set and we're not yet under MPI, exec mpiexec.

    Lets users write ``sweep-tasks build-index -n 50 ...`` as a shortcut
    for ``mpiexec -n 50 sweep-tasks build-index --mpi ...``. The explicit
    mpiexec form keeps working: when this process IS already a child of
    mpiexec (detected via MPI env vars), the call short-circuits and we
    fall through to do real work.
    """
    import os
    import sys

    n = getattr(args, "mpi_ranks", None)
    if not n or int(n) <= 1:
        return
    if _running_under_mpi():
        return  # we ARE the child; fall through
    cmd = _build_mpi_reexec_cmd(int(n), list(sys.argv))
    print(f"[sweep-tasks] re-exec under mpiexec -n {n}: {' '.join(cmd)}",
          flush=True)
    os.execvp(cmd[0], cmd)
    # os.execvp does not return on success; if we get here, something
    # went wrong (mpiexec missing from PATH, etc.).
    raise RuntimeError(f"os.execvp returned for cmd={cmd!r}; mpiexec may "
                       "not be on PATH. Either install MPI or run the "
                       "single-process path without -n.")


def _cmd_build_index(args) -> int:
    """``sweep-tasks build-index`` — scan raw SEG-Y → SEGYIndex npz."""
    import time

    from sweep_io.segy_index import SEGYIndex, build_segy_index

    # When invoked with -n N at the shell prompt (not already under MPI),
    # re-exec self via `mpiexec -n N sweep-tasks build-index --mpi ...`
    # so the user doesn't have to type the mpiexec wrapper themselves.
    _maybe_reexec_under_mpi(args)

    missing = [n for n in ("segy_root", "out") if not getattr(args, n, None)]
    if missing:
        labels = {"segy_root": "--segy-root", "out": "-o/--out"}
        print(f"error: build-index requires {missing} via CLI flag "
              f"({labels[missing[0]]}) or --config YAML key.")
        return 2

    is_root, mpi_size, use_mpi = _detect_mpi(args.mpi)
    segy_root = Path(args.segy_root).expanduser().resolve()
    if not segy_root.is_dir():
        if is_root:
            print(f"error: --segy-root {segy_root} is not a directory")
        return 2
    paths = _resolve_segy_paths(segy_root, args.glob, args.files_from, is_root)
    if paths is None:
        return 2
    byte_map = _parse_byte_map_args(args, is_root)
    if byte_map is None:
        return 2

    if is_root:
        mode = f"mpi ranks={mpi_size}" if use_mpi else f"threads={args.num_workers}"
        src = (f"--files-from={args.files_from}" if args.files_from
               else f"glob={args.glob!r}")
        print(f"[build-index] scanning {len(paths)} SEG-Y files under "
              f"{segy_root} ({src}; {mode})")

    out = Path(args.out).expanduser().resolve()
    t0 = time.perf_counter()
    if use_mpi:
        # Reuse the file-staged MPI scanner from crg_build — it returns
        # a SEGYIndex on rank 0 and None on non-root ranks.
        from mpi4py import MPI  # type: ignore
        from sweep_io.crg_build import _scan_segy_mpi
        index = _scan_segy_mpi(
            paths,
            mpi_comm=MPI.COMM_WORLD,
            byte_map=byte_map or None,
            source_depth_m_override=args.source_depth_m,
            receiver_depth_m_override=args.receiver_depth_m,
            coord_scalar_override=args.coord_scalar,
            show_progress=bool(args.progress) and is_root,
            stage_dir=out.parent,
        )
    else:
        index = build_segy_index(
            paths,
            byte_map=byte_map or None,
            source_depth_m_override=args.source_depth_m,
            receiver_depth_m_override=args.receiver_depth_m,
            coord_scalar_override=args.coord_scalar,
            num_workers=int(args.num_workers),
            show_progress=bool(args.progress),
        )
    if index is None:  # non-root MPI rank
        return 0
    elapsed = time.perf_counter() - t0
    print(f"[build-index] scan complete: {index.n_traces:,} traces across "
          f"{len(index.file_paths)} files, dt={index.dt_s:.4g}s "
          f"nt={index.n_samples}, wall={elapsed:.1f}s")
    out.parent.mkdir(parents=True, exist_ok=True)
    index.save(out)
    print(f"[build-index] wrote {out} ({out.stat().st_size / (1 << 20):.1f} MB)")

    if not bool(getattr(args, "no_qc_png", False)):
        try:
            from sweep_tasks.qc_plots import plot_index_qc
            qc_path = out.with_name(out.stem + "_qc.png")
            plot_index_qc(index, qc_path)
            print(f"[build-index] QC PNG -> {qc_path}")
        except Exception as exc:  # noqa: BLE001 — best-effort plotting
            print(f"[build-index] warning: could not render QC PNG: {exc}")
    return 0


def _cmd_filter_image(args) -> int:
    """``sweep-tasks filter-image`` — depth-tapered z-low-cut on a 2-D image npy.

    Standalone wrapper around
    :func:`sweep_tasks.postproc.filter_image.filter_image_file`. Use this to
    iterate on filter parameters without re-running the underlying RTM /
    FWI-gradient computation. Same algorithm is also auto-invoked by the
    runner when ``RTMImagingSpec.post_filter`` is configured in the YAML.
    """
    from sweep_tasks.postproc.filter_image import filter_image_file

    if not getattr(args, "input", None):
        print("error: filter-image requires a positional input npy path "
              "(or YAML key `input`).")
        return 2
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input {input_path} not found")
        return 2

    dz_m = args.dz_m
    dx_m = args.dx_m if args.dx_m is not None else args.dz_m
    if dz_m is None:
        # Try to fall back to a sibling rtm_result.npz that stashes ``dh``.
        sibling = input_path.parent / "rtm_result.npz"
        if sibling.is_file():
            try:
                import numpy as _np
                with _np.load(sibling) as data:
                    if "dh" in data.files:
                        dz_m = float(data["dh"])
                        if dx_m is None:
                            dx_m = dz_m
                        print(f"[filter-image] inferred dz/dx={dz_m} m from "
                              f"{sibling.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[filter-image] could not read dh from {sibling}: {exc}")
        if dz_m is None:
            print("error: --dz-m is required (no rtm_result.npz next to "
                  f"{input_path.name} to infer from)")
            return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None else input_path.parent
    )
    meta = filter_image_file(
        input_path,
        output_dir=output_dir,
        output_name=args.output_name,
        dz_m=float(dz_m),
        dx_m=(None if dx_m is None else float(dx_m)),
        wavelength_m=float(args.wavelength_m),
        depth_m=float(args.depth_m),
        taper_m=float(args.taper_m),
        clip_percentile=float(args.clip_percentile),
        display_scale=float(args.display_scale),
        x_origin_m=float(args.x_origin_m),
        z_origin_m=float(args.z_origin_m),
        x_max_m=(None if args.x_max_m is None else float(args.x_max_m)),
        cmap=str(args.cmap),
        save_png=not bool(args.no_png),
    )
    print(f"[filter-image] wrote {meta['output']}")
    print(f"[filter-image] removed -> {meta['removed']}")
    if not bool(args.no_png):
        stem = Path(meta["output"]).with_suffix("")
        print(f"[filter-image] PNG     -> {stem}.png")
        print(f"[filter-image] compare -> {stem}_comparison.png")
    return 0


def _cmd_analyze_wavelet(args) -> int:
    """``sweep-tasks analyze-wavelet`` — rank-1 robust direct-wave average.

    Wraps :func:`sweep_tasks.wavelet.analyze.analyze_direct_wavelet_batch`
    with a SeismicPlan path + per-shot selection / quality-filter flags.
    """
    import shlex as _shlex
    from sweep_tasks.wavelet import (
        AnalyzeWaveletConfig,
        analyze_direct_wavelet_batch,
    )

    missing = [n for n in ("plan", "out_dir") if not getattr(args, n, None)]
    if missing:
        print(f"error: analyze-wavelet requires {missing} via CLI flag "
              f"(--{missing[0].replace('_', '-')}) or --config YAML key.")
        return 2

    cfg = AnalyzeWaveletConfig(
        plan_path=Path(args.plan).expanduser().resolve(),
        output_dir=Path(args.out_dir).expanduser().resolve(),
        shot_start=int(args.shot_start),
        shot_stop=int(args.shot_stop),
        shot_stride=int(args.shot_stride),
        nearest_receivers=int(args.nearest_receivers),
        max_abs_offset_m=args.max_abs_offset_m,
        water_velocity_m_s=float(args.water_velocity_m_s),
        rel_t_min_s=float(args.rel_t_min_s),
        rel_t_max_s=float(args.rel_t_max_s),
        iterations=int(args.iterations),
        filter_lowcut_hz=args.filter_lowcut_hz,
        filter_highcut_hz=args.filter_highcut_hz,
        filter_order=int(args.filter_order),
        prepad_s=float(args.prepad_s),
        min_residual_ratio=float(args.min_residual_ratio),
        min_shape_corr_to_average=float(args.min_shape_corr_to_average),
        no_plot=bool(args.no_plot),
        save_per_shot_npz=bool(args.save_per_shot_npz),
        command=" ".join(_shlex.quote(a) for a in [sys.executable, *sys.argv]),
    )
    analyze_direct_wavelet_batch(cfg)
    return 0


def _cmd_estimate_wavelet(args) -> int:
    """``sweep-tasks estimate-wavelet`` — SIREN + sweep wave-equation refinement."""
    import shlex as _shlex
    from sweep_tasks.wavelet import (
        WaveletInversionConfig,
        run_wavelet_inversion,
    )

    missing = [n for n in ("plan", "out_dir") if not getattr(args, n, None)]
    if missing:
        print(f"error: estimate-wavelet requires {missing} via CLI flag "
              f"(--{missing[0].replace('_', '-')}) or --config YAML key.")
        return 2

    cfg = WaveletInversionConfig(
        plan_path=Path(args.plan).expanduser().resolve(),
        output_dir=Path(args.out_dir).expanduser().resolve(),
        shot_start=int(args.shot_start),
        shot_stop=int(args.shot_stop),
        tmax_s=float(args.tmax_s),
        velocity_m_s=float(args.velocity_m_s),
        dx_m=float(args.dx_m),
        dz_m=float(args.dz_m),
        dy_m=(float(args.dy_m) if args.dy_m is not None else None),
        x_padding_m=float(args.x_padding_m),
        z_padding_m=float(args.z_padding_m),
        y_padding_m=(float(args.y_padding_m) if args.y_padding_m is not None else None),
        model_depth_m=float(args.model_depth_m),
        equation=str(args.equation),
        simulation_dt_s=(float(args.simulation_dt_s) if args.simulation_dt_s is not None else None),
        observed_delay_s=(float(args.observed_delay_s) if args.observed_delay_s is not None else None),
        direct_rel_t_min_s=(float(args.direct_rel_t_min_s) if args.direct_rel_t_min_s is not None else None),
        direct_rel_t_max_s=(float(args.direct_rel_t_max_s) if args.direct_rel_t_max_s is not None else None),
        direct_water_velocity_m_s=(float(args.direct_water_velocity_m_s) if args.direct_water_velocity_m_s is not None else None),
        backend=str(args.backend),
        mode=str(args.mode),
        initial_frequency_hz=float(args.initial_frequency_hz),
        epochs=int(args.epochs),
        lr=float(args.lr),
        loss_type=str(args.loss_type),
        filter_lowcut_hz=args.filter_lowcut_hz,
        filter_highcut_hz=args.filter_highcut_hz,
        filter_order=int(args.filter_order),
        max_abs_offset_m=args.max_abs_offset_m,
        nearest_receivers=args.nearest_receivers,
        spatial_order=int(args.spatial_order),
        abcn=int(args.abcn),
        free_surface=bool(args.free_surface),
        inr_hidden_features=int(args.inr_hidden_features),
        inr_hidden_layers=int(args.inr_hidden_layers),
        inr_first_omega0=float(args.inr_first_omega0),
        inr_hidden_omega0=float(args.inr_hidden_omega0),
        wavelet_taper_start_s=args.wavelet_taper_start_s,
        wavelet_taper_end_s=args.wavelet_taper_end_s,
        wavelet_zero_initial_samples=int(args.wavelet_zero_initial_samples),
        seed=int(args.seed),
        save_every=int(args.save_every),
        initial_wavelet_path=(
            Path(args.initial_wavelet).expanduser().resolve()
            if args.initial_wavelet is not None else None
        ),
        prefit_steps=int(args.prefit_steps),
        prefit_lr=float(args.prefit_lr),
        initial_wavelet_prior_weight=float(args.initial_wavelet_prior_weight),
        command=" ".join(_shlex.quote(a) for a in [sys.executable, *sys.argv]),
    )
    out_path = run_wavelet_inversion(cfg)
    print(f"[estimate-wavelet] estimated_wavelet: {out_path}")
    return 0


def _cmd_convert_farfield(args) -> int:
    """``sweep-tasks convert-farfield`` — ASCII farfield → npz."""
    from sweep_tasks.wavelet import convert_farfield

    missing = [n for n in ("input", "output") if not getattr(args, n, None)]
    if missing:
        labels = {"input": "--input", "output": "--out"}
        print(f"error: convert-farfield requires {missing} via CLI flag "
              f"({labels[missing[0]]}) or --config YAML key "
              f"({missing[0]}).")
        return 2

    meta = convert_farfield(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        dt_s=float(args.dt_s),
        zero_initial_samples=int(args.zero_initial_samples),
        taper_start_s=args.taper_start_s,
        taper_end_s=args.taper_end_s,
        save_plot=not bool(args.no_plot),
    )
    print(f"[convert-farfield] wavelet_npz : {meta['output_npz']}")
    print(f"[convert-farfield] n_samples   : {meta['n_samples']} "
          f"(duration {meta['duration_s']:.3f} s @ dt={meta['dt_s']:.6f} s)")
    return 0


def _cmd_plot_wavelet_steps(args) -> int:
    """``sweep-tasks plot-wavelet-steps`` — 4-panel pipeline QC plot."""
    from sweep_tasks.wavelet import plot_wavelet_pipeline_steps

    if not getattr(args, "pipeline_dir", None):
        print("error: plot-wavelet-steps requires --pipeline-dir "
              "(or YAML key `pipeline_dir`).")
        return 2

    stack, overlay = plot_wavelet_pipeline_steps(
        pipeline_dir=Path(args.pipeline_dir).expanduser().resolve(),
        output_path=(
            Path(args.output).expanduser().resolve()
            if args.output is not None else None
        ),
        max_freq_hz=float(args.max_freq_hz),
        dt_s=args.dt_s,
    )
    print(f"[plot-wavelet-steps] stack   -> {stack}")
    print(f"[plot-wavelet-steps] overlay -> {overlay}")
    return 0


def _cmd_build_plan(args) -> int:
    """``sweep-tasks build-plan`` — SEGYIndex npz + filters → SeismicPlan npz."""
    import time

    from sweep_io.segy_index import SEGYIndex
    from sweep_io.seismic_plan import build_seismic_plan

    missing = [n for n in ("index", "out") if not getattr(args, n, None)]
    if missing:
        labels = {"index": "--index", "out": "-o/--out"}
        print(f"error: build-plan requires {missing} via CLI flag "
              f"({labels[missing[0]]}) or --config YAML key.")
        return 2

    index_path = Path(args.index).expanduser().resolve()
    if not index_path.is_file():
        print(f"error: --index {index_path} not found")
        return 2

    print(f"[build-plan] loading index from {index_path}")
    t0 = time.perf_counter()
    index = SEGYIndex.load(index_path)
    print(f"[build-plan] index: {index.n_traces:,} traces, "
          f"{len(index.file_paths)} files, dt={index.dt_s:.4g}s, "
          f"loaded in {time.perf_counter()-t0:.2f}s")

    shot_ids = None
    if args.shot_ids:
        try:
            shot_ids = [int(s) for s in args.shot_ids.split(",") if s.strip()]
        except ValueError as exc:
            print(f"error: --shot-ids must be comma-separated ints: {exc}")
            return 2

    t0 = time.perf_counter()
    plan = build_seismic_plan(
        index,
        grouping=args.grouping,
        shot_ids=shot_ids,
        receiver_quantize_m=args.receiver_quantize_m,
        offset_min_m=args.offset_min_m,
        offset_max_m=args.offset_max_m,
        max_traces_per_group=args.max_traces_per_group,
        seed=int(args.seed),
        build_label=args.label,
    )
    print(f"[build-plan] {args.grouping}: {plan.n_groups} groups, "
          f"{plan.n_rows:,} plan rows, wall={time.perf_counter()-t0:.2f}s")
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plan.save(out)
    print(f"[build-plan] wrote {out} ({out.stat().st_size / (1 << 20):.1f} MB)")

    if not bool(getattr(args, "no_qc_png", False)):
        try:
            from sweep_tasks.qc_plots import plot_plan_qc
            qc_path = out.with_name(out.stem + "_qc.png")
            plot_plan_qc(plan, qc_path, parent_index=index)
            print(f"[build-plan]  QC PNG -> {qc_path}")
        except Exception as exc:  # noqa: BLE001 — best-effort plotting
            print(f"[build-plan]  warning: could not render QC PNG: {exc}")
    return 0




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweep-tasks",
        description="YAML / Pydantic task layer for sweep — run FWI / LSRTM / forward / wavefield jobs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # `sweep-tasks run <task.yaml> [--override key=value ...]`
    run_parser = subparsers.add_parser("run", help="Run a task YAML through TaskRunner")
    run_parser.add_argument("task_file", help="Path to a task YAML spec")
    run_parser.add_argument(
        "--nproc-per-node", "--nproc_per_node", type=int, default=1, metavar="N",
        dest="nproc_per_node",
        help=(
            "Number of local-node ranks. Default 1 (in-process). For N>1, "
            "sweep-tasks re-execs itself under `torchrun --standalone "
            "--nproc_per_node=N` automatically — no need to type `torchrun` "
            "yourself. The explicit form (`torchrun --nproc_per_node=N -m "
            "sweep_tasks run …`) keeps working unchanged."
        ),
    )
    run_parser.add_argument(
        "--dd-py", "--dd_py", type=int, default=1, metavar="PY", dest="dd_py",
        help=(
            "Domain-decomposition tiles along inline/y (model parallel). "
            "With PY*PX>1 this sets SWEEP_DD_ENABLE=1 / SWEEP_DD_PY / "
            "SWEEP_DD_PX (what the runner's _dd_config reads) AND auto-derives "
            "--nproc-per-node = PY*PX — so `sweep-tasks run --dd-py 2 --dd-px 2 "
            "foo.yaml` is all you need (no manual env export, no torchrun, "
            "process count always matches the mesh). Default 1 (no DD)."
        ),
    )
    run_parser.add_argument(
        "--dd-px", "--dd_px", type=int, default=1, metavar="PX", dest="dd_px",
        help="Domain-decomposition tiles along crossline/x. See --dd-py.",
    )
    run_parser.add_argument(
        "--override", action="append", metavar="KEY=VALUE", default=None,
        help=(
            "Dotted-key override applied to the parsed YAML before validation. "
            "Repeatable. Example: --override epochs=20 "
            "--override illumination_precondition.enabled=true. "
            "Values are parsed as YAML so booleans / numbers / lists work."
        ),
    )
    # ``--resume`` / ``--no-resume`` shortcut for the FWI/LSRTM ``resume:``
    # spec field. With ``--resume``, if a checkpoint.pt is present in the
    # same task_dir (deterministic when ``task_id:`` is fixed in the
    # YAML), the runner picks up where it left off — pairs with ctrl-c
    # mid-training, which now writes a clean checkpoint at the next iter
    # boundary. ``--no-resume`` forces a fresh start even if the YAML
    # has ``resume: true``. When neither is passed, the YAML's value
    # wins (default ``false``).
    run_parser.add_argument(
        "--resume", dest="resume", action="store_true", default=None,
        help=(
            "Auto-resume from <task_dir>/checkpoint.pt if it exists. "
            "Equivalent to --override resume=true. Pairs with ctrl-c "
            "during training (which saves a clean checkpoint at the "
            "next iter boundary)."
        ),
    )
    run_parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force a fresh start (overrides resume:true in the YAML).",
    )

    # `sweep-tasks init [TEMPLATE] [-o FILE]` — emit a starter --config YAML.
    init_parser = subparsers.add_parser(
        "init",
        help="Emit a starter --config YAML for a dataset workflow "
             "(viking, …). Edit the file then drive every step with "
             "`sweep-tasks <cmd> --config <file>`.",
    )
    init_parser.add_argument(
        "template", nargs="?", default=None,
        help="Template name (default: viking). Use --list to see all.",
    )
    init_parser.add_argument(
        "-o", "--output", default=None,
        help="Write to this file instead of stdout. Refuses to overwrite "
             "an existing file unless --force is also passed.",
    )
    init_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite -o/--output target if it already exists.",
    )
    init_parser.add_argument(
        "--list", action="store_true",
        help="List bundled templates and exit.",
    )

    # `sweep-tasks new <task_type> [...]`
    new_parser = subparsers.add_parser("new", help="Emit a YAML template for a task type")
    new_parser.add_argument(
        "task_type",
        choices=["introspect", "forward", "wavefield", "fwi", "lsrtm", "rtm"],
        help="Task type to scaffold",
    )
    new_parser.add_argument("--equation", default=None, help="Equation name override (e.g. Acoustic)")
    new_parser.add_argument("--backend", choices=["eager", "c"], default="eager",
                            help="Propagator backend (default: eager)")
    new_parser.add_argument("--memory", choices=["full", "boundary", "ckpt"], default="full",
                            help="CUDA memory strategy (only valid with --backend c; default: full)")
    new_parser.add_argument("--storage", choices=["gpu", "cpu", "disk"], default="gpu",
                            help="Memory storage location for boundary / ckpt (default: gpu)")
    new_parser.add_argument("--compile", action="store_true",
                            help="Enable torch.compile (only valid with --backend eager)")
    new_parser.add_argument("-o", "--output", default=None, help="Write to this file instead of stdout")

    # `sweep-tasks build-index` — scan raw SEG-Y → SEGYIndex npz (no grouping).
    bi = subparsers.add_parser(
        "build-index",
        help="Scan raw SEG-Y files and save a SEGYIndex (per-trace catalog) npz. "
             "The index is the canonical input for `build-plan`.",
    )
    bi.add_argument("--segy-root", default=None,
                    help="Directory containing SEG-Y files.")
    bi.add_argument("--glob", default="*.sgy",
                    help="Glob pattern under --segy-root (default: *.sgy). "
                         "Ignored when --files-from is supplied.")
    bi.add_argument("--files-from", default=None,
                    help="Text file with one SEG-Y path per line (absolute or "
                         "relative to --segy-root). Overrides --glob.")
    bi.add_argument("-o", "--out", default=None,
                    help="Output SEGYIndex npz path.")
    bi.add_argument("--num-workers", type=int, default=16,
                    help="Parallel header-scan workers (single-process mode).")
    bi.add_argument("--mpi", action="store_true",
                    help="Run under MPI: rank 0 partitions files, each rank "
                         "scans its slice, rank 0 stages + merges. Launch via "
                         "`mpiexec -n N sweep-tasks build-index --mpi ...`. "
                         "Auto-detected when launched under mpiexec.")
    bi.add_argument("-n", "--mpi-ranks", type=int, default=None,
                    metavar="N",
                    help="Spawn N MPI ranks via `mpiexec -n N` by "
                         "re-execing this command. Convenience shortcut for "
                         "`mpiexec -n N sweep-tasks build-index --mpi ...`. "
                         "Ignored when already running as an MPI child (so "
                         "the explicit mpiexec form keeps working).")
    bi.add_argument("--progress", action="store_true",
                    help="Print one line per scanned file (single-process mode).")
    bi.add_argument("--source-z-byte", type=int, default=None,
                    help="0-based byte offset for source-z. Alias for "
                         "--byte-map source_depth=BYTE.")
    bi.add_argument("--receiver-z-byte", type=int, default=None,
                    help="0-based byte offset for receiver-z (byte 41-44 on some OBN datasets). "
                         "Default reads SEG-Y rev1 byte 53-56.")
    bi.add_argument("--source-depth-m", type=float, default=None,
                    help="Constant source-depth override (m).")
    bi.add_argument("--receiver-depth-m", type=float, default=None,
                    help="Constant receiver-depth override (m).")
    bi.add_argument("--coord-scalar", type=float, default=None,
                    help="Override SEG-Y coord scalar.")
    bi.add_argument("--byte-map", action="append", default=None, metavar="KEY=BYTE",
                    help="Override standard SEG-Y trace-header bytes. Repeatable.")
    bi.add_argument("--no-qc-png", action="store_true",
                    help="Skip the acquisition-geometry QC PNG (default: write "
                         "<out>_qc.png next to the npz).")
    bi.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with a `build_index:` section to source "
                         "default values from. CLI flags override YAML.")

    # `sweep-tasks build-plan` — SEGYIndex npz + filters → SeismicPlan npz.
    bp = subparsers.add_parser(
        "build-plan",
        help="Derive a SeismicPlan (seismic_plan_v1, CSG or CRG grouping) from "
             "a SEGYIndex npz. Plans are cheap to rebuild — the SEG-Y header "
             "scan only happens once via `build-index`.",
    )
    bp.add_argument("--index", default=None,
                    help="Input SEGYIndex npz (from `sweep-tasks build-index`).")
    bp.add_argument("-o", "--out", default=None,
                    help="Output SeismicPlan npz path.")
    bp.add_argument("--grouping", default="csg", choices=["csg", "crg"],
                    help="Group rows by shot (CSG, default — typical 2-D / 3-D "
                         "streamer) or by quantized receiver cell (CRG — OBN). "
                         "CRG requires --receiver-quantize-m.")
    bp.add_argument("--receiver-quantize-m", type=float, default=None,
                    help="Receiver-cell quantization tolerance (m). REQUIRED "
                         "when --grouping=crg, e.g. 0.5.")
    bp.add_argument("--shot-ids", type=str, default=None,
                    help="Comma-separated whitelist of shot IDs (FFIDs) to keep.")
    bp.add_argument("--offset-min-m", type=float, default=None,
                    help="Per-trace minimum source-receiver horizontal offset (m).")
    bp.add_argument("--offset-max-m", type=float, default=None,
                    help="Per-trace maximum source-receiver horizontal offset (m). "
                         "Use to drop far-offset noise (typical OBN-3D max-offset budget).")
    bp.add_argument("--max-traces-per-group", type=int, default=None,
                    help="Hard cap on rows per group after grouping; excess "
                         "rows randomly sub-sampled.")
    bp.add_argument("--seed", type=int, default=0,
                    help="RNG seed for --max-traces-per-group sub-sampling.")
    bp.add_argument("--label", type=str, default=None,
                    help="Optional human-readable tag stored in plan.build_meta.")
    bp.add_argument("--no-qc-png", action="store_true",
                    help="Skip the plan QC PNG (default: write "
                         "<out>_qc.png next to the plan npz; if --index is "
                         "supplied the dropped groups are underlaid in gray).")
    bp.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with a `build_plan:` section to source "
                         "default values from. CLI flags override YAML.")

    # `sweep-tasks filter-image <input.npy> ...` — depth-tapered z-low-cut
    # on an RTM / gradient image. Standalone wrapper around the same algorithm
    # the runner uses when ``RTMImagingSpec.post_filter`` is configured.
    fi = subparsers.add_parser(
        "filter-image",
        help="Apply depth-tapered z-axis low-cut filter to a 2-D image npy. "
             "Useful for iterating on filter params without re-running RTM.",
        description=(
            "Depth-tapered z-axis low-cut filter (port of legacy "
            "07_filter_imaging.py). Removes the slowly-varying-with-depth "
            "drift that contaminates shallow stacked RTM / gradient images, "
            "leaving the deep section untouched via a cosine taper. The same "
            "algorithm is auto-applied by `sweep-tasks run` on every RTM "
            "imaging product when the YAML sets `imaging.post_filter.enabled "
            "= true`."
        ),
    )
    fi.add_argument("input", nargs="?", default=None,
                    help="Path to a 2-D float npy (e.g. rtm_image_per_shot_"
                         "normalised.npy). Output lands next to it as "
                         "<stem>_shallow_zlowcut.npy + matching PNG. "
                         "Can also be supplied via --config (key: input).")
    fi.add_argument("--dz-m", type=float, default=None,
                    help="Vertical grid spacing in metres. Required unless a "
                         "sibling rtm_result.npz next to the input provides "
                         "`dh` (the RTM runner writes this automatically).")
    fi.add_argument("--dx-m", type=float, default=None,
                    help="Horizontal grid spacing in metres (defaults to "
                         "--dz-m; only used for PNG axes).")
    fi.add_argument("--wavelength-m", type=float, default=300.0,
                    help="z-direction wavelength threshold for the Gaussian "
                         "low-pass that defines 'slow drift'. Features with "
                         "wavelength > this are subtracted. Default: 300 m "
                         "(matches legacy Viking recipe).")
    fi.add_argument("--depth-m", type=float, default=600.0,
                    help="Filter is full-strength in 0..depth_m. Default: "
                         "600 m.")
    fi.add_argument("--taper-m", type=float, default=400.0,
                    help="Cosine ramp from full-strength at depth_m to zero "
                         "at depth_m + taper_m. Default: 400 m.")
    fi.add_argument("--clip-percentile", type=float, default=1.0,
                    help="Signed percentile for PNG colour-bar clipping "
                         "(default 1.0 -> vmin/vmax at the 1st/99th "
                         "percentile of the data).")
    fi.add_argument("--display-scale", type=float, default=1.0,
                    help="Scale factor applied to vmin/vmax half-width "
                         "(default 1.0). <1 narrows the colour range, >1 "
                         "widens it.")
    fi.add_argument("--cmap", default="sweep_image",
                    help="matplotlib colormap for PNG output (default: "
                         "sweep_image, the bundled diverging LUT from "
                         "sweep_viz tuned for percentile-clipped RTM / "
                         "kernel images; pass 'seismic' or 'gray' for "
                         "legacy display).")
    fi.add_argument("--output-dir", default=None,
                    help="Where to write outputs (default: same directory as "
                         "the input npy).")
    fi.add_argument("--output-name", default=None,
                    help="Override the default <input_stem>_shallow_zlowcut "
                         "filename stem (omit the suffix).")
    fi.add_argument("--x-origin-m", type=float, default=0.0,
                    help="x-axis origin for PNG extent (m). Default 0.")
    fi.add_argument("--z-origin-m", type=float, default=0.0,
                    help="z-axis origin for PNG extent (m). Default 0.")
    fi.add_argument("--x-max-m", type=float, default=None,
                    help="If set, crop the PNG x-axis to [x_origin_m, x_max_m].")
    fi.add_argument("--no-png", action="store_true",
                    help="Skip PNG generation (npy + metadata only).")
    fi.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with a `filter_image:` section to source "
                         "default values from. CLI flags override YAML.")

    # `sweep-tasks analyze-wavelet` — rank-1 robust direct-wave average.
    aw = subparsers.add_parser(
        "analyze-wavelet",
        help="Estimate one rank-1 direct-wave wavelet per group from a CSG or "
             "CRG plan, polarity-align + shape-corr-mask, write the robust "
             "average npz.",
    )
    aw.add_argument("--plan", default=None,
                    help="SeismicPlan npz (grouping='csg' or 'crg').")
    aw.add_argument("--out-dir", default=None,
                    help="Output directory. Will contain average/, analysis/, summary.json.")
    aw.add_argument("--shot-start", type=int, default=1,
                    help="1-based first shot ordinal (default: 1).")
    aw.add_argument("--shot-stop", type=int, default=200,
                    help="1-based inclusive last shot ordinal (default: 200).")
    aw.add_argument("--shot-stride", type=int, default=1)
    aw.add_argument("--nearest-receivers", type=int, default=10,
                    help="Number of near-offset traces per shot (default: 10; Viking canonical: 4).")
    aw.add_argument("--max-abs-offset-m", type=float, default=None,
                    help="Optional max |offset| in meters before nearest selection.")
    aw.add_argument("--water-velocity-m-s", type=float, default=1500.0,
                    help="Direct-wave water velocity in m/s (default: 1500).")
    aw.add_argument("--rel-t-min-s", type=float, default=-0.05,
                    help="Wavelet relative-time start in seconds (default: -0.05).")
    aw.add_argument("--rel-t-max-s", type=float, default=0.40,
                    help="Wavelet relative-time end in seconds (default: 0.40).")
    aw.add_argument("--iterations", type=int, default=20,
                    help="Rank-1 alternation iterations per shot (default: 20).")
    aw.add_argument("--filter-lowcut-hz", type=float, default=None,
                    help="Optional Butterworth low cut.")
    aw.add_argument("--filter-highcut-hz", type=float, default=None,
                    help="Optional Butterworth high cut.")
    aw.add_argument("--filter-order", type=int, default=4)
    aw.add_argument("--prepad-s", type=float, default=0.0,
                    help="Causal-wavelet front zero-pad in seconds.")
    aw.add_argument("--min-residual-ratio", type=float, default=0.6,
                    help="Drop shots whose direct-window residual/observed RMS ratio exceeds this.")
    aw.add_argument("--min-shape-corr-to-average", type=float, default=0.9,
                    help="Shape-correlation threshold for inclusion in the robust average.")
    aw.add_argument("--no-plot", action="store_true",
                    help="Skip the analysis PNGs (overlay / spectrum / residual).")
    aw.add_argument("--save-per-shot-npz", action="store_true",
                    help="Also save one direct_wavelet.npz per shot.")
    aw.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with an `analyze_wavelet:` section to "
                         "source default values from (and optional `common:` "
                         "fallback). CLI flags override YAML.")

    # `sweep-tasks estimate-wavelet` — SIREN LBFGS prefit + wave-equation refinement.
    ew = subparsers.add_parser(
        "estimate-wavelet",
        help="Refine a source wavelet by inverting a small batch of shots through "
             "the sweep acoustic propagator (SIREN or discrete parameterisation).",
    )
    ew.add_argument("--plan", default=None,
                    help="SeismicPlan npz (grouping='csg' or 'crg').")
    ew.add_argument("--out-dir", default=None,
                    help="Output dir for estimated_wavelet.npz, prefit_siren_wavelet.npz, etc.")
    ew.add_argument("--initial", default=None, dest="initial_wavelet",
                    help="Optional initial wavelet npz (e.g. analyze-wavelet output) used for SIREN prefit / prior.")
    ew.add_argument("--shot-start", type=int, default=1)
    ew.add_argument("--shot-stop", type=int, default=20,
                    help="Number of shots used by the diagnostic inversion batch (default: 20).")
    ew.add_argument("--nearest-receivers", type=int, default=None,
                    help="Sub-select N nearest-offset traces per shot (default: keep all).")
    ew.add_argument("--max-abs-offset-m", type=float, default=None)
    ew.add_argument("--tmax-s", type=float, default=1.0,
                    help="Trace time truncation in seconds (default: 1.0).")
    ew.add_argument("--velocity-m-s", type=float, default=1500.0,
                    help="Background velocity for the 1-D forward grid (default: 1500).")
    ew.add_argument("--dx-m", type=float, default=12.5)
    ew.add_argument("--dz-m", type=float, default=12.5)
    ew.add_argument("--dy-m", type=float, default=None,
                    help="3D only: y-direction grid spacing (defaults to --dx-m).")
    ew.add_argument("--x-padding-m", type=float, default=500.0)
    ew.add_argument("--z-padding-m", type=float, default=100.0)
    ew.add_argument("--y-padding-m", type=float, default=None,
                    help="3D only: y-direction padding (defaults to --x-padding-m).")
    ew.add_argument("--model-depth-m", type=float, default=1200.0)
    ew.add_argument("--equation", choices=["Acoustic", "Acoustic3D"], default="Acoustic",
                    help="2D Acoustic (default) or 3D Acoustic3D propagator.")
    ew.add_argument("--simulation-dt-s", type=float, default=None,
                    help="Run the propagator at this dt (must divide plan.dt_s into an "
                         "integer stride). Lets you satisfy CFL on a finer grid without "
                         "rebuilding the plan; syn is stride-sampled back to plan.dt_s "
                         "before loss. Default: plan.dt_s.")
    ew.add_argument("--observed-delay-s", type=float, default=None,
                    help="Right-shift obs by this many seconds before training (legacy "
                         "fwi_workflow-dev delay_traces alignment). Default: derive from "
                         "the initial wavelet's analyze metadata (max(0, -rel_t.min()) "
                         "+ prepad_s); 0.0 disables the shift entirely.")
    ew.add_argument("--direct-rel-t-min-s", type=float, default=None,
                    help="Direct-window mask: lower bound (seconds, relative to predicted "
                         "direct arrival). When both --direct-rel-t-min/max-s are set, the "
                         "loss only counts samples in [arrival+min, arrival+max] — matches "
                         "legacy fwi_workflow-dev masked_cosine_loss. Default: full-trace loss.")
    ew.add_argument("--direct-rel-t-max-s", type=float, default=None,
                    help="Direct-window mask: upper bound (seconds, relative to predicted "
                         "direct arrival). See --direct-rel-t-min-s.")
    ew.add_argument("--direct-water-velocity-m-s", type=float, default=None,
                    help="Velocity used to predict direct arrivals for the mask. Default: "
                         "fall back to --velocity-m-s.")
    ew.add_argument("--backend", choices=["cuda", "eager"], default="cuda")
    ew.add_argument("--mode", choices=["discrete", "siren"], default="siren")
    ew.add_argument("--initial-frequency-hz", type=float, default=8.0,
                    help="Ricker peak frequency for discrete-mode initial wavelet.")
    ew.add_argument("--epochs", type=int, default=100)
    ew.add_argument("--lr", type=float, default=1.0e-3)
    ew.add_argument("--loss-type", choices=["mse", "trace_cosine"], default="trace_cosine")
    ew.add_argument("--filter-lowcut-hz", type=float, default=None)
    ew.add_argument("--filter-highcut-hz", type=float, default=None)
    ew.add_argument("--filter-order", type=int, default=4)
    ew.add_argument("--spatial-order", type=int, default=4)
    ew.add_argument("--abcn", type=int, default=40)
    ew.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
    ew.add_argument("--inr-hidden-features", type=int, default=64)
    ew.add_argument("--inr-hidden-layers", type=int, default=3)
    ew.add_argument("--inr-first-omega0", type=float, default=30.0)
    ew.add_argument("--inr-hidden-omega0", type=float, default=30.0)
    ew.add_argument("--wavelet-taper-start-s", type=float, default=None)
    ew.add_argument("--wavelet-taper-end-s", type=float, default=None)
    ew.add_argument("--wavelet-zero-initial-samples", type=int, default=0)
    ew.add_argument("--seed", type=int, default=0)
    ew.add_argument("--save-every", type=int, default=10)
    ew.add_argument("--prefit-steps", type=int, default=10,
                    help="LBFGS prefit iterations against the initial wavelet (SIREN mode only).")
    ew.add_argument("--prefit-lr", type=float, default=1.0e-3)
    ew.add_argument("--initial-wavelet-prior-weight", type=float, default=0.0,
                    help="Prior MSE weight anchoring SIREN to the initial wavelet during main loop.")
    ew.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with an `estimate_wavelet:` section to "
                         "source default values from (and optional `common:` "
                         "fallback). CLI flags override YAML.")

    # `sweep-tasks convert-farfield` — ASCII FarField → npz adapter.
    cf = subparsers.add_parser(
        "convert-farfield",
        help="Convert a one-column ASCII FarField wavelet (e.g. Viking Farfield.dat) "
             "into a sweep-compatible npz with optional taper / zero-padding.",
    )
    cf.add_argument("--input", default=None,
                    help="ASCII file, one sample per line.")
    cf.add_argument("--out", default=None, dest="output",
                    help="Output wavelet npz path.")
    cf.add_argument("--dt-s", type=float, default=0.004,
                    help="Sample interval in seconds (default: 0.004 = Viking).")
    cf.add_argument("--zero-initial-samples", type=int, default=0)
    cf.add_argument("--taper-start-s", type=float, default=None)
    cf.add_argument("--taper-end-s", type=float, default=None)
    cf.add_argument("--no-plot", action="store_true")
    cf.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with a `convert_farfield:` section to "
                         "source default values from. CLI flags override YAML.")

    # `sweep-tasks plot-wavelet-steps` — 4-step pipeline QC plot.
    pw = subparsers.add_parser(
        "plot-wavelet-steps",
        help="Render a 4-panel comparison of the wavelet-estimation pipeline outputs "
             "(initial_wavelet_target / prefit_siren_wavelet / estimated_wavelet).",
    )
    pw.add_argument("--pipeline-dir", default=None,
                    help="Directory holding the three pipeline npz files.")
    pw.add_argument("--out", default=None, dest="output",
                    help="Output PNG path. Defaults to <pipeline-dir>/wavelet_pipeline_steps.png.")
    pw.add_argument("--max-freq-hz", type=float, default=80.0)
    pw.add_argument("--dt-s", type=float, default=None,
                    help="Override sample interval (only needed if time_s is missing from the npz).")
    pw.add_argument("--config", default=None, metavar="FILE",
                    help="YAML file with a `plot_wavelet_steps:` section to "
                         "source default values from. CLI flags override YAML.")

    # `sweep-tasks tasks {list,status,logs}`
    tasks_parser = subparsers.add_parser("tasks", help="Inspect previously run tasks")
    tasks_sub = tasks_parser.add_subparsers(dest="tasks_command")

    tasks_list_parser = tasks_sub.add_parser("list", help="List tasks in a tasks directory")
    tasks_list_parser.add_argument("--output-dir", default="./sweep_runs",
                                   help="Tasks root (default: ./sweep_runs)")

    tasks_status_parser = tasks_sub.add_parser("status", help="Print status.json for a task")
    tasks_status_parser.add_argument("task_id")
    tasks_status_parser.add_argument("--output-dir", default="./sweep_runs")

    tasks_logs_parser = tasks_sub.add_parser("logs", help="Show captured logs (Phase 1: status.json only)")
    tasks_logs_parser.add_argument("task_id")
    tasks_logs_parser.add_argument("--output-dir", default="./sweep_runs")

    args = parser.parse_args(argv)

    # Subcommands that accept --config FILE: apply the YAML section to args
    # *before* dispatching to the cmd function. The cmd functions read
    # args.* as normal, so they don't need to know whether the value came
    # from the CLI or YAML.
    _config_subparsers = {
        "build-index":        (bi, "build_index"),
        "build-plan":         (bp, "build_plan"),
        "filter-image":       (fi, "filter_image"),
        "analyze-wavelet":    (aw, "analyze_wavelet"),
        "estimate-wavelet":   (ew, "estimate_wavelet"),
        "convert-farfield":   (cf, "convert_farfield"),
        "plot-wavelet-steps": (pw, "plot_wavelet_steps"),
    }
    if args.command in _config_subparsers:
        sub_parser, section_name = _config_subparsers[args.command]
        _merge_yaml_into_args(args, sub_parser, section_name)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "new":
        return _cmd_new(args)
    if args.command == "build-index":
        return _cmd_build_index(args)
    if args.command == "build-plan":
        return _cmd_build_plan(args)
    if args.command == "filter-image":
        return _cmd_filter_image(args)
    if args.command == "analyze-wavelet":
        return _cmd_analyze_wavelet(args)
    if args.command == "estimate-wavelet":
        return _cmd_estimate_wavelet(args)
    if args.command == "convert-farfield":
        return _cmd_convert_farfield(args)
    if args.command == "plot-wavelet-steps":
        return _cmd_plot_wavelet_steps(args)
    if args.command == "tasks":
        if args.tasks_command == "list":
            return _cmd_tasks_list(args)
        if args.tasks_command == "status":
            return _cmd_tasks_status(args)
        if args.tasks_command == "logs":
            return _cmd_tasks_logs(args)
        tasks_parser.print_help()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main() or 0)
