"""run_meta.json records the version of the code that ran.

A package imported from a source tree on PYTHONPATH reads its ``__version__``
from the metadata of whatever distribution is installed under its name; a stale
install (sweep-solver 0.1.0 while the tree is 0.3.5) made run_meta.json name the
wrong solver version.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sweepx.tasks._helpers import metadata as M


def _tree(root: Path, mod: str, *, name: str, version: str, dunder: str) -> Path:
    """A source tree <root>/src/<mod> with a pyproject.toml; returns <root>/src."""
    (root / "src" / mod).mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n')
    (root / "src" / mod / "__init__.py").write_text(f'__version__ = "{dunder}"\n')
    return root / "src"


def _installed(root: Path, name: str, version: str) -> Path:
    """A bare installed distribution (dist-info only) of ``name`` under ``root``."""
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return root


def test_source_tree_version_wins_over_a_stale_dunder(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_tree(tmp_path / "t", "fakesolver_a", name="fakesolver-a",
                                          version="9.9.9", dunder="0.1.0")))
    assert M._package_version("fakesolver_a", "fakesolver-a") == ("9.9.9", None)


def test_a_shadowed_installed_distribution_is_reported(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_installed(tmp_path / "site", "fakesolver-b", "0.1.0")))
    monkeypatch.syspath_prepend(str(_tree(tmp_path / "t", "fakesolver_b", name="fakesolver-b",
                                          version="0.3.5", dunder="0.1.0")))
    assert M._package_version("fakesolver_b", "fakesolver-b") == ("0.3.5", "0.1.0")


def test_another_projects_pyproject_is_not_taken(tmp_path, monkeypatch):
    # the tree's pyproject names another project: the package's own __version__ stays
    monkeypatch.syspath_prepend(str(_tree(tmp_path / "t", "fakesolver_c", name="something-else",
                                          version="7.0.0", dunder="1.2.3")))
    assert M._package_version("fakesolver_c", "fakesolver-c") == ("1.2.3", None)


def test_git_meta_has_describe(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    root = tmp_path / "repo"
    src = _tree(root, "fakesolver_d", name="fakesolver-d", version="9.9.9", dunder="9.9.9")

    def git(*args):
        subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org",
                        *args], check=True, capture_output=True)

    git("init", "-q")
    # As in any real checkout: _git_meta imports the module, which writes
    # __pycache__/ into the tree unless PYTHONDONTWRITEBYTECODE is set.
    (root / ".gitignore").write_text("__pycache__/\n")
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    git("tag", "v9.9.9")
    (root / "notes.txt").write_text("x\n")
    git("add", "-A")
    git("commit", "-q", "-m", "two")
    monkeypatch.syspath_prepend(str(src))
    meta = M._git_meta("fakesolver_d")
    assert meta is not None and meta["path"] == str(root.resolve())
    assert meta["describe"].startswith("v9.9.9-1-g") and meta["dirty"] is False


def test_run_meta_has_the_new_fields(tmp_path):
    import numpy as np
    import yaml

    from sweepx.tasks import load_task
    from sweepx.tasks.runner import _dump_run_metadata

    shape = (48, 48)
    np.save(tmp_path / "true.npy", np.full(shape, 2600.0, dtype=np.float32))
    np.save(tmp_path / "init.npy", np.full(shape, 2500.0, dtype=np.float32))
    spec = {
        "task_type": "fwi", "output_dir": str(tmp_path / "tasks"), "task_id": "meta_versions",
        "grid": {"dh": 10.0}, "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {"kind": "line", "sources": {"step": 6, "depth": 2, "start": 6, "stop": 42},
                     "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44}},
        "physics": {"equation": "Acoustic", "spatial_order": 4, "abcn": 8, "free_surface": False,
                    "pml_type": "cpmlr", "source_type": ["h1"], "receiver_type": ["h1"]},
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(tmp_path / "init.npy")},
        "obs": {"synthetic_from": {"name": "vp", "path": str(tmp_path / "true.npy")}},
        "model_bounds": {"vp": {"min": 1500.0, "max": 4500.0}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 1, "show_every": 1,
    }
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    _dump_run_metadata(load_task(tmp_path / "t.yaml"), tmp_path / "dumped")
    meta = json.loads((tmp_path / "dumped" / "run_meta.json").read_text())
    assert set(meta["package_versions"]) == {n for n, _ in M._RECORDED_PACKAGES}
    assert isinstance(meta["installed_versions_differ"], dict)
    for entry in meta["git_repos"].values():
        assert {"path", "sha", "describe", "dirty"} <= set(entry)
