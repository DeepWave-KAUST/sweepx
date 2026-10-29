"""`sweepx skill install` puts the skills where Claude Code / Codex read them.

Pins: each skill lands at ``<base>/.claude/skills/<name>`` and
``<base>/.agents/skills/<name>`` with the same files as the source; the
solver's skill comes along when the installed sweep-solver carries one;
with no assistant found and none named nothing is written; a re-install
replaces our own copy but never a foreign folder or a symlink.
"""
import zipfile
from pathlib import Path

import pytest

from sweepx import skill
from sweepx.cli import main


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A user with no assistant: empty HOME, nothing on PATH."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
    return h


@pytest.fixture
def solver_skill(tmp_path, monkeypatch):
    """A stand-in for the skill a newer sweep-solver carries."""
    d = tmp_path / "solver" / "sweep-solver"
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: sweep-solver\ndescription: x\n---\n# s\n")
    (d / "references" / "a.md").write_text("a\n")
    monkeypatch.setattr(skill, "solver_skill_dir", lambda: d)
    return d


def test_the_bundled_skill_is_found_and_named_like_its_folder():
    src = skill.skill_dir()
    assert (src / "SKILL.md").is_file()
    assert skill.skill_name_of(src) == src.name == "sweepx"


def test_nothing_found_nothing_named_installs_nothing(home, capsys):
    assert skill.detect() == []
    assert main(["skill", "install"]) == 2
    assert "--for claude" in capsys.readouterr().err
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("marker, target", [(".claude", "claude"),
                                            (".agents", "codex"),
                                            (".codex", "codex")])
def test_an_assistant_folder_is_detected(home, marker, target):
    (home / marker).mkdir()
    assert skill.detect() == [target]


def test_install_for_all_copies_both_skills(home, solver_skill):
    assert main(["skill", "install", "--for", "all"]) == 0
    for base in (".claude/skills", ".agents/skills"):
        assert _files(home / base / "sweepx") == _files(skill.skill_dir())
        assert _files(home / base / "sweep-solver") == _files(solver_skill)
        assert sorted(p.name for p in (home / base).iterdir()) == ["sweep-solver", "sweepx"]


def test_an_old_solver_without_a_skill_still_installs_sweepx(home, monkeypatch, capsys):
    monkeypatch.setattr(skill, "solver_skill_dir", lambda: None)
    assert main(["skill", "install", "--for", "claude"]) == 0
    assert sorted(p.name for p in (home / ".claude/skills").iterdir()) == ["sweepx"]
    assert "carries no skill" in capsys.readouterr().out


def test_default_installs_only_for_the_detected_assistant(home, solver_skill):
    (home / ".codex").mkdir()
    assert main(["skill", "install"]) == 0
    assert (home / ".agents/skills/sweepx/SKILL.md").is_file()
    assert not (home / ".claude").exists()


def test_project_install(home, tmp_path, solver_skill):
    proj = tmp_path / "proj"
    proj.mkdir()
    assert main(["skill", "install", "--for", "claude", "--project", str(proj)]) == 0
    assert (proj / ".claude/skills/sweepx/SKILL.md").is_file()
    assert (proj / ".claude/skills/sweep-solver/SKILL.md").is_file()
    assert list(home.iterdir()) == []


def test_reinstall_replaces_our_own_copy(home, solver_skill):
    skill.install(["codex"])
    dest = home / ".agents/skills/sweepx"
    (dest / "stale.md").write_text("from an older release")
    done = skill.install(["codex"])
    assert {how for _, _, how in done} == {"updated"}
    assert _files(dest) == _files(skill.skill_dir())


def test_a_foreign_folder_is_left_alone(home, solver_skill):
    dest = home / ".claude/skills/sweepx"
    dest.mkdir(parents=True)
    (dest / "notes.txt").write_text("mine")
    assert main(["skill", "install", "--for", "claude"]) == 1
    assert _files(dest) == ["notes.txt"]


def test_a_symlink_is_left_alone(home, tmp_path, solver_skill):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    dest = home / ".agents/skills/sweepx"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(checkout)
    with pytest.raises(skill.SkillError, match="symlink"):
        skill.install(["codex"])
    assert dest.is_symlink() and list(checkout.iterdir()) == []


def test_zip_writes_one_zip_per_skill(home, tmp_path, solver_skill):
    out = tmp_path / "zips"
    assert main(["skill", "zip", "-o", str(out)]) == 0
    for name, src in (("sweepx", skill.skill_dir()), ("sweep-solver", solver_skill)):
        names = zipfile.ZipFile(out / f"{name}.zip").namelist()
        assert sorted(n[len(name) + 1:] for n in names) == _files(src)
