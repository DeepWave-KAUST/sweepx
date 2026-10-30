"""``sweepx`` command: one entry point for the toolkit's command lines.

    sweepx tasks run examples/tasks/synthetic/09_forward_constant_box.yaml
    sweepx agent chat
    sweepx skill install
    sweepx --version

``sweep-tasks`` and ``sweep-agent`` stay installed as the same commands.
"""

import argparse
import importlib
import sys

# subcommand -> module whose main(argv) takes over the rest of the line
COMMANDS = {
    "tasks": ("sweepx.tasks.cli", "YAML task runner (same as `sweep-tasks`)"),
    "agent": ("sweepx.agent.cli", "natural-language agent (same as `sweep-agent`)"),
    "skill": ("sweepx.skill", "install the Agent Skills for Claude Code / Codex"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        # Load only the chosen command: the agent and the runner import torch.
        return importlib.import_module(COMMANDS[argv[0]][0]).main(argv[1:])

    from sweepx import __version__

    ap = argparse.ArgumentParser(
        prog="sweepx",
        description="The sweep toolkit. Run `sweepx <command> --help` for a command's options.",
    )
    ap.add_argument("--version", action="version", version=f"sweepx {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    for name, (_, help_) in COMMANDS.items():
        sub.add_parser(name, help=help_, add_help=False)
    ap.parse_args(argv)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
