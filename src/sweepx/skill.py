"""``sweepx skill``: install the Agent Skills for an AI coding assistant.

Two skills: ``sweepx`` (``skills/sweepx/`` in this repository, carried in
the wheel under ``sweepx/_skill/``) and the solver's ``sweep-solver``,
which comes from the installed ``sweep-solver`` (0.3.7 and later carry it).

    sweepx skill install                 # the assistants found on this machine
    sweepx skill install --for codex     # one of them, found or not
    sweepx skill install --project .     # into a project instead of $HOME
    sweepx skill zip -o DIR              # one zip per skill, for Claude.ai
    sweepx skill path                    # where the bundled skills are

An assistant reads skills from a folder, nothing has to be registered:
Claude Code from ``~/.claude/skills/<name>/`` (or a project's
``.claude/skills/``), Codex from ``~/.agents/skills/<name>/`` (or a
project's ``.agents/skills/``). A skill copied there before the assistant
is installed is picked up when it is. Nothing is installed when no
assistant is found and none is named: the command lists the choices.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

SKILL_NAME = "sweepx"

#: assistant -> skills folder, relative to $HOME or to a project
TARGETS = {
    "claude": Path(".claude") / "skills",
    "codex": Path(".agents") / "skills",
}

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
_NAME_LINE = re.compile(r"^name:\s*(\S+)\s*$", re.MULTILINE)


class SkillError(RuntimeError):
    """An install that would do the wrong thing; the message says why."""


def skill_dir() -> Path:
    """The sweepx skill folder: the wheel's copy, else the source checkout's."""
    pkg = Path(__file__).resolve().parent
    for cand in (pkg / "_skill" / SKILL_NAME,              # wheel
                 pkg.parents[1] / "skills" / SKILL_NAME):  # src/sweepx -> repo
        if (cand / "SKILL.md").is_file():
            return cand
    raise SkillError(
        f"the {SKILL_NAME} skill is not in this installation ({pkg}); "
        "take skills/sweepx from https://github.com/DeepWave-KAUST/sweepx")


def solver_skill_dir() -> Optional[Path]:
    """The installed sweep-solver's skill folder, or None if it has none."""
    try:
        from sweep.skill import skill_dir as solver_dir
        return solver_dir()
    except Exception:  # sweep-solver <= 0.3.6, or a build without the skill
        return None


def skill_dirs() -> List[Path]:
    solver = solver_skill_dir()
    return [skill_dir()] + ([solver] if solver is not None else [])


def skill_name_of(folder: Path) -> Optional[str]:
    """The ``name:`` in a folder's SKILL.md front matter, or None."""
    try:
        text = (Path(folder) / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    m = _NAME_LINE.search(text.split("---", 2)[1])
    return m.group(1) if m else None


def detect(home: Optional[Path] = None) -> List[str]:
    """Assistants that look installed for this user: their config folder
    exists in ``home`` or their command is on PATH."""
    home = Path.home() if home is None else Path(home)
    found = []
    if (home / ".claude").is_dir() or shutil.which("claude"):
        found.append("claude")
    if ((home / ".agents").is_dir() or (home / ".codex").is_dir()
            or shutil.which("codex")):
        found.append("codex")
    return found


def copy_skill(src: Path, dest: Path) -> str:
    """Copy ``src`` to ``dest``; return ``"installed"`` or ``"updated"``.

    An existing ``dest`` is replaced only when it is this same skill (its
    SKILL.md has the same ``name:``); anything else there, or a symlink
    (a checkout someone linked on purpose), is left alone with an error.
    The new copy is written next to ``dest`` and renamed into place, so a
    failed copy never leaves a half-written skill.
    """
    src, dest = Path(src), Path(dest)
    name = skill_name_of(src)
    if dest.is_symlink():
        raise SkillError(f"{dest} is a symlink to {os.readlink(dest)}; "
                         "remove it to install a copy")
    if dest.exists() and skill_name_of(dest) != name:
        raise SkillError(f"{dest} exists and is not the {name} skill; "
                         "move it away to install here")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        shutil.copytree(src, tmp / "skill", ignore=_IGNORE)
        old = None
        if dest.exists():
            old = tmp / "old"
            dest.rename(old)
        try:
            (tmp / "skill").rename(dest)
        except OSError:
            if old is not None:
                old.rename(dest)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "updated" if old is not None else "installed"


def install(targets: Iterable[str], *, project: Optional[Path] = None,
            home: Optional[Path] = None,
            sources: Optional[Iterable[Path]] = None) -> List[tuple]:
    """Install every skill of ``sources`` (default :func:`skill_dirs`) for
    ``targets``, into ``project`` when given, else ``home``.

    Returns ``[(target, path, "installed"|"updated")]``.
    """
    base = (Path.home() if home is None else Path(home)) if project is None else Path(project)
    sources = skill_dirs() if sources is None else [Path(s) for s in sources]
    done = []
    for t in dict.fromkeys(targets):
        if t not in TARGETS:
            raise SkillError(f"unknown assistant {t!r}; one of {sorted(TARGETS)}")
        for src in sources:
            dest = base / TARGETS[t] / (skill_name_of(src) or src.name)
            done.append((t, dest, copy_skill(src, dest)))
    return done


def make_zip(src: Path, out: Path) -> Path:
    """Zip one skill with its folder at the top, as Claude.ai expects."""
    src, out = Path(src), Path(out)
    name = skill_name_of(src) or src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            rel = f.relative_to(src)
            if f.is_file() and "__pycache__" not in rel.parts and f.suffix != ".pyc":
                zf.write(f, Path(name) / rel)
    return out


NO_ASSISTANT = (
    "no Claude Code or Codex found for this user (no ~/.claude, ~/.agents "
    "or ~/.codex, no `claude` or `codex` on PATH), so nothing was installed.\n"
    "  sweepx skill install --for claude   # Claude Code: ~/.claude/skills/\n"
    "  sweepx skill install --for codex    # Codex: ~/.agents/skills/\n"
    "  sweepx skill install --for all      # both\n"
    "  sweepx skill zip                    # Claude.ai: upload the zips in settings\n"
    "Run it on the machine where the assistant runs; add --project DIR to "
    "install into a project (what a cloud session sees).")

NO_SOLVER_SKILL = (
    "note: the installed sweep-solver carries no skill (0.3.7 and later do); "
    "for the solver's skill upgrade it, or take skills/sweep-solver from "
    "https://github.com/DeepWave-KAUST/sweep")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sweepx skill", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", metavar="<install|zip|path>")
    pi = sub.add_parser("install", help="copy the skills to where an assistant reads them")
    pi.add_argument("--for", dest="targets", choices=[*TARGETS, "all"], action="append",
                    help="assistant to install for (repeatable); default: the ones "
                         "found on this machine")
    pi.add_argument("--project", type=Path, metavar="DIR", nargs="?", const=Path("."),
                    help="install into DIR/.claude/skills or DIR/.agents/skills "
                         "(default DIR: the current directory) instead of $HOME")
    pz = sub.add_parser("zip", help="zip each skill for upload to Claude.ai")
    pz.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    sub.add_parser("path", help="print where the bundled skills are")
    args = ap.parse_args(argv)

    try:
        if args.command == "install":
            targets = args.targets or detect()
            if not targets:
                print(NO_ASSISTANT, file=sys.stderr)
                return 2
            if "all" in targets:
                targets = list(TARGETS)
            for t, path, how in install(targets, project=args.project):
                print(f"{how} for {t}: {path}")
            if solver_skill_dir() is None:
                print(NO_SOLVER_SKILL)
            print("A running Claude Code or Codex session may need a restart "
                  "to see a skill folder it did not have before.")
            return 0
        if args.command == "zip":
            for src in skill_dirs():
                out = make_zip(src, args.output_dir / f"{skill_name_of(src) or src.name}.zip")
                print(f"wrote {out}")
            print("Upload them in the skills section of Claude.ai's settings.")
            if solver_skill_dir() is None:
                print(NO_SOLVER_SKILL)
            return 0
        if args.command == "path":
            for src in skill_dirs():
                print(src)
            return 0
    except SkillError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
