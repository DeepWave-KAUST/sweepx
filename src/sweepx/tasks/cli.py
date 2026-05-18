"""`sweep-tasks` CLI — `run`, `new`, `tasks {list,status,logs}`.

Mirrors what used to live in `sweep.cli` (subcommands `run/new/tasks`)
before the task layer was split into its own package. The engine-level
introspection commands (`sweep list equations`, `sweep show <Eq>`) stay
in `sweep` itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_run(args) -> int:
    import yaml

    from sweep_tasks import TaskRunner, load_task, load_task_from_dict

    overrides = list(getattr(args, "override", None) or [])
    if overrides:
        from sweep_runner.config import apply_overrides

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


def _cmd_build_plan(args) -> int:
    """``sweep-tasks build-plan`` — SEGYIndex npz + filters → SeismicPlan npz."""
    import time

    from sweep_io.segy_index import SEGYIndex
    from sweep_io.seismic_plan import build_seismic_plan

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
        "--override", action="append", metavar="KEY=VALUE", default=None,
        help=(
            "Dotted-key override applied to the parsed YAML before validation. "
            "Repeatable. Example: --override epochs=20 "
            "--override illumination_precondition.enabled=true. "
            "Values are parsed as YAML so booleans / numbers / lists work."
        ),
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
    bi.add_argument("--segy-root", required=True,
                    help="Directory containing SEG-Y files.")
    bi.add_argument("--glob", default="*.sgy",
                    help="Glob pattern under --segy-root (default: *.sgy). "
                         "Ignored when --files-from is supplied.")
    bi.add_argument("--files-from", default=None,
                    help="Text file with one SEG-Y path per line (absolute or "
                         "relative to --segy-root). Overrides --glob.")
    bi.add_argument("-o", "--out", required=True,
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

    # `sweep-tasks build-plan` — SEGYIndex npz + filters → SeismicPlan npz.
    bp = subparsers.add_parser(
        "build-plan",
        help="Derive a SeismicPlan (seismic_plan_v1, CSG or CRG grouping) from "
             "a SEGYIndex npz. Plans are cheap to rebuild — the SEG-Y header "
             "scan only happens once via `build-index`.",
    )
    bp.add_argument("--index", required=True,
                    help="Input SEGYIndex npz (from `sweep-tasks build-index`).")
    bp.add_argument("-o", "--out", required=True,
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
    fi.add_argument("input",
                    help="Path to a 2-D float npy (e.g. rtm_image_per_shot_"
                         "normalised.npy). Output lands next to it as "
                         "<stem>_shallow_zlowcut.npy + matching PNG.")
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

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "new":
        return _cmd_new(args)
    if args.command == "build-index":
        return _cmd_build_index(args)
    if args.command == "build-plan":
        return _cmd_build_plan(args)
    if args.command == "filter-image":
        return _cmd_filter_image(args)
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
