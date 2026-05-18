"""Tests for ``sweep-tasks run --override key=value``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sweep_tasks.cli import main


def _make_2d_fwi_yaml(tmp_path: Path) -> Path:
    shape = (32, 32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_vp = (2200.0 + 600.0 * np.linspace(0, 1, shape[0])[:, None]
               * np.ones(shape)).astype(np.float32)
    init_path = tmp_path / "init.npy"
    true_path = tmp_path / "true.npy"
    np.save(init_path, init_vp)
    np.save(true_path, true_vp)
    spec = {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 6, "depth": 2, "start": 6, "stop": 28},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 30},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "optimizer": {"kind": "adam", "lr": 5.0, "eps": 1.0e-22},
        "epochs": 1, "batchsize": 2, "show_every": 1,
    }
    yaml_path = tmp_path / "fwi.yaml"
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return yaml_path


def test_cli_run_with_override_changes_epochs(tmp_path, capsys):
    yaml_path = _make_2d_fwi_yaml(tmp_path)
    rc = main(["run", str(yaml_path), "--override", "epochs=2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "applied 1 override(s)" in captured.out
    # The runner's per-epoch print confirms two iterations ran.
    assert "epoch 0001" in captured.out, captured.out


def test_cli_run_with_multiple_overrides(tmp_path, capsys):
    yaml_path = _make_2d_fwi_yaml(tmp_path)
    rc = main([
        "run", str(yaml_path),
        "--override", "epochs=1",
        "--override", "batchsize=1",
        "--override", "optimizer.lr=2.5",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "applied 3 override(s)" in captured.out


def test_cli_build_plan_crg_grouping_end_to_end(tmp_path, capsys):
    """``sweep-tasks build-index`` then ``build-plan --grouping crg`` on a
    tiny SEG-Y fixture → unified ``seismic_plan_v1`` npz.

    Replaces the deprecated ``build-crg-plan`` CLI (removed in TASK 019);
    CRG plans now go through the same two-step pipeline as CSG, just with
    ``--grouping crg --receiver-quantize-m ...``.
    """
    import struct

    from sweep_io.segy import (
        FORMAT_IEEE_FLOAT32,
        SEGY_BIN_HEADER_SIZE,
        SEGY_TEXT_HEADER_SIZE,
        SEGY_TRACE_HEADER_SIZE,
    )

    segy_dir = tmp_path / "raw"
    segy_dir.mkdir()
    # Two SEG-Y files, 2 traces each → 2 unique receiver cells (OBN nodes)
    # seeing 1 shot from each source line.
    for fi, line_sx in enumerate(([100.0, 200.0], [300.0, 400.0])):
        path = segy_dir / f"sourceLine_{fi:03d}.sgy"
        with path.open("wb") as f:
            f.write(b"\x00" * SEGY_TEXT_HEADER_SIZE)
            bh = bytearray(SEGY_BIN_HEADER_SIZE)
            struct.pack_into(">H", bh, 16, 2000)   # dt = 2 ms
            struct.pack_into(">H", bh, 20, 16)     # n_samples
            struct.pack_into(">H", bh, 24, FORMAT_IEEE_FLOAT32)
            f.write(bytes(bh))
            for shot_idx, sxi in enumerate(line_sx):
                for ri, gxi in enumerate([500.0, 700.0]):
                    th = bytearray(SEGY_TRACE_HEADER_SIZE)
                    struct.pack_into(">i", th, 8, shot_idx + 1)
                    struct.pack_into(">i", th, 12, ri)
                    struct.pack_into(">h", th, 70, 1)
                    struct.pack_into(">i", th, 72, int(sxi))
                    struct.pack_into(">i", th, 76, 0)
                    struct.pack_into(">i", th, 80, int(gxi))
                    struct.pack_into(">i", th, 84, 1000)
                    struct.pack_into(">i", th, 52, 50)  # rz
                    f.write(bytes(th))
                    f.write(b"\x00" * 16 * 4)

    idx = tmp_path / "idx.npz"
    rc = main([
        "build-index",
        "--segy-root", str(segy_dir),
        "--glob", "sourceLine_*.sgy",
        "--out", str(idx),
        "--num-workers", "1",
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert idx.exists()

    out = tmp_path / "plan.npz"
    rc = main([
        "build-plan",
        "--index", str(idx),
        "--grouping", "crg",
        "--receiver-quantize-m", "1.0",
        "--out", str(out),
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert out.exists()

    from sweep_io.seismic_plan import SeismicPlan
    plan = SeismicPlan.load(out)
    assert plan.grouping == "crg"
    # 2 files × 2 shots × 2 receivers = 8 traces; receiver-quantize=1 m
    # leaves 2 unique receiver cells -> 2 groups, 4 rows each.
    assert plan.n_groups == 2
    assert plan.n_rows == 8
    import numpy as np
    np.testing.assert_array_equal(plan.per_group_row_counts(), [4, 4])


# ============================================================================
# -n / --mpi-ranks: build-index re-exec under mpiexec
# ============================================================================


def test_build_mpi_reexec_cmd_strips_short_form_and_adds_mpi():
    from sweep_tasks.cli import _build_mpi_reexec_cmd
    argv = ["/path/to/sweep-tasks", "build-index",
            "--segy-root", "/a", "-o", "/b", "-n", "50"]
    cmd = _build_mpi_reexec_cmd(50, argv)
    assert cmd[:3] == ["mpiexec", "-n", "50"]
    assert cmd[3] == "/path/to/sweep-tasks"
    # Original -n / 50 stripped from the spawned argv so children don't
    # recurse forever.
    tail = cmd[4:]
    assert "-n" not in tail
    assert "50" not in tail
    assert "--mpi" in tail
    # Other args preserved.
    for needle in ("build-index", "--segy-root", "/a", "-o", "/b"):
        assert needle in tail, f"missing {needle} in {tail}"


def test_build_mpi_reexec_cmd_strips_long_form():
    from sweep_tasks.cli import _build_mpi_reexec_cmd
    argv = ["/p/sweep-tasks", "build-index", "--mpi-ranks", "8",
            "--segy-root", "/a", "--out", "/b"]
    cmd = _build_mpi_reexec_cmd(8, argv)
    tail = cmd[4:]
    assert "--mpi-ranks" not in tail
    # The "8" before mpiexec ("-n 8") should be the only one.
    assert cmd.count("8") == 1


def test_build_mpi_reexec_cmd_strips_equals_form():
    from sweep_tasks.cli import _build_mpi_reexec_cmd
    argv = ["/p/sweep-tasks", "build-index", "-n=8", "--mpi-ranks=8",
            "--segy-root", "/a"]
    cmd = _build_mpi_reexec_cmd(8, argv)
    tail = cmd[4:]
    assert "-n=8" not in tail
    assert "--mpi-ranks=8" not in tail
    # And the long --mpi-ranks gets stripped too.
    assert "--mpi-ranks" not in tail


def test_build_mpi_reexec_cmd_does_not_duplicate_mpi_flag():
    from sweep_tasks.cli import _build_mpi_reexec_cmd
    argv = ["/p/sweep-tasks", "build-index", "--mpi", "-n", "4",
            "--segy-root", "/a"]
    cmd = _build_mpi_reexec_cmd(4, argv)
    # --mpi should appear exactly once (we don't double-add it).
    assert cmd[4:].count("--mpi") == 1


def test_build_mpi_reexec_cmd_does_not_duplicate_program_path():
    """Regression: an earlier bug walked all of argv (incl. argv[0]) when
    building cleaned, so the spawned cmd was
    `mpiexec -n N /path/sweep-tasks /path/sweep-tasks build-index ...`
    and argparse rejected the second program path as an invalid command.
    """
    from sweep_tasks.cli import _build_mpi_reexec_cmd
    argv = ["/full/path/sweep-tasks", "build-index",
            "--segy-root", "/a", "-o", "/b", "-n", "16"]
    cmd = _build_mpi_reexec_cmd(16, argv)
    # The program path should appear exactly once (at cmd[3], between
    # mpiexec args and the rest of argv).
    assert cmd.count("/full/path/sweep-tasks") == 1
    assert cmd[3] == "/full/path/sweep-tasks"
    # First positional after the program must be the subcommand name.
    assert cmd[4] == "build-index"


def test_maybe_reexec_under_mpi_noop_when_n_unset(monkeypatch):
    """No --mpi-ranks → no re-exec, no exception."""
    from sweep_tasks.cli import _maybe_reexec_under_mpi

    class _Args:
        mpi_ranks = None

    called = {}
    monkeypatch.setattr("os.execvp", lambda *a, **kw: called.setdefault("c", 1))
    _maybe_reexec_under_mpi(_Args())
    assert "c" not in called  # execvp not invoked


def test_maybe_reexec_under_mpi_noop_when_already_under_mpi(monkeypatch):
    """--mpi-ranks=N + MPI env var set → no re-exec (we ARE the child)."""
    from sweep_tasks.cli import _maybe_reexec_under_mpi

    class _Args:
        mpi_ranks = 50

    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "50")
    called = {}
    monkeypatch.setattr("os.execvp", lambda *a, **kw: called.setdefault("c", 1))
    _maybe_reexec_under_mpi(_Args())
    assert "c" not in called


def test_maybe_reexec_under_mpi_does_exec_when_eligible(monkeypatch):
    """--mpi-ranks=N at the shell prompt → triggers os.execvp."""
    import sys
    from sweep_tasks.cli import _maybe_reexec_under_mpi

    class _Args:
        mpi_ranks = 4

    # Ensure no MPI env vars.
    for k in ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK", "PMI_SIZE",
              "PMI_RANK", "MPI_LOCALRANKID"):
        monkeypatch.delenv(k, raising=False)
    # Synthesize a controlled argv.
    monkeypatch.setattr(
        sys, "argv",
        ["/p/sweep-tasks", "build-index", "--segy-root", "/a", "-n", "4"],
    )
    captured = {}

    def _fake_execvp(file, args):
        captured["file"] = file
        captured["args"] = args
        # Don't actually re-exec; let the test continue. Real os.execvp
        # never returns; if it did, _maybe_reexec_under_mpi would raise
        # RuntimeError on the next line — that's covered by the file/args
        # invariants we check below.
        raise SystemExit(0)

    monkeypatch.setattr("os.execvp", _fake_execvp)
    import pytest
    with pytest.raises(SystemExit):
        _maybe_reexec_under_mpi(_Args())
    assert captured["file"] == "mpiexec"
    assert captured["args"][:3] == ["mpiexec", "-n", "4"]
    assert "--mpi" in captured["args"]
    # The triggering -n / 4 stripped from the spawned argv.
    assert "-n" not in captured["args"][4:]


def test_cli_run_with_invalid_override_returns_2(tmp_path, capsys):
    yaml_path = _make_2d_fwi_yaml(tmp_path)
    rc = main(["run", str(yaml_path), "--override", "no_equals_sign"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "invalid --override" in captured.out
