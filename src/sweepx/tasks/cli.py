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
    from sweep_tasks import TaskRunner, load_task

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sweep-tasks",
        description="YAML / Pydantic task layer for sweep — run FWI / LSRTM / forward / wavefield jobs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # `sweep-tasks run <task.yaml>`
    run_parser = subparsers.add_parser("run", help="Run a task YAML through TaskRunner")
    run_parser.add_argument("task_file", help="Path to a task YAML spec")

    # `sweep-tasks new <task_type> [...]`
    new_parser = subparsers.add_parser("new", help="Emit a YAML template for a task type")
    new_parser.add_argument(
        "task_type",
        choices=["introspect", "forward", "wavefield", "fwi", "lsrtm"],
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

    # `sweep-tasks tasks {list,status,logs}`
    tasks_parser = subparsers.add_parser("tasks", help="Inspect previously run tasks")
    tasks_sub = tasks_parser.add_subparsers(dest="tasks_command")

    tasks_list_parser = tasks_sub.add_parser("list", help="List tasks in a tasks directory")
    tasks_list_parser.add_argument("--output-dir", default="./sweep_tasks",
                                   help="Tasks root (default: ./sweep_tasks)")

    tasks_status_parser = tasks_sub.add_parser("status", help="Print status.json for a task")
    tasks_status_parser.add_argument("task_id")
    tasks_status_parser.add_argument("--output-dir", default="./sweep_tasks")

    tasks_logs_parser = tasks_sub.add_parser("logs", help="Show captured logs (Phase 1: status.json only)")
    tasks_logs_parser.add_argument("task_id")
    tasks_logs_parser.add_argument("--output-dir", default="./sweep_tasks")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "new":
        return _cmd_new(args)
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
