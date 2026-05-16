"""End-to-end test of the new sweep task-layer CLI on the Marmousi model.

Exercises every subcommand introduced in `[TASK 002] LLM-ready task layer`:

  sweep list equations           sweep show Acoustic
  sweep new <task_type>          sweep run <task.yaml>
  sweep tasks list               sweep tasks status <id>

The forward and wavefield tasks run on the full 141x681 Marmousi grid; the
FWI and LSRTM tasks are deliberately truncated to 3 epochs so the whole
script finishes in minutes on a single GPU.

Usage
-----
    cd <repo_root>
    python examples/tasks/test_cli_marmousi.py [--device cuda|cpu|auto]

Pass `--keep-out` to keep the generated `examples/tasks/cli_demo_out/`
folder for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR / "cli_demo_out"
TASKS_DIR = DEMO_DIR / "tasks"


def _env() -> dict[str, str]:
    """Inherit the current process env; assume `sweep` and `sweep-tasks`
    are already importable (installed via pip / `install_ecosystem.sh`)."""
    return os.environ.copy()


def _step(idx: object, label: str) -> None:
    print(f"\n========== step {idx}: {label} ==========", flush=True)


def _run_cli(*args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Dispatch to `sweep` (engine introspection) or `sweep-tasks` (task layer).

    After the task layer was split into the `sweep-tasks` companion package,
    `list` / `show` live in `sweep.cli` while `run` / `new` / `tasks` live in
    `sweep_tasks.cli`. The first arg picks the module.
    """
    task_layer_cmds = {"run", "new", "tasks"}
    module = "sweep_tasks.cli" if (args and args[0] in task_layer_cmds) else "sweep.cli"
    cmd = [sys.executable, "-m", module, *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    if capture:
        return subprocess.run(cmd, env=_env(), check=check, capture_output=True, text=True)
    return subprocess.run(cmd, env=_env(), check=check)


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _marmousi_paths(marmousi_dir: Path) -> dict[str, Path]:
    paths = {
        "true": marmousi_dir / "true.npy",
        "smooth": marmousi_dir / "smooth.npy",
    }
    for label, p in paths.items():
        if not p.exists():
            raise SystemExit(
                f"Marmousi {label} model not found at {p}. "
                f"Place true.npy / smooth.npy under {marmousi_dir} "
                "(e.g. via the sweep main repo's "
                "`examples/models/marmousi/download_marmousi.py` script), "
                "or override the location with --marmousi-dir."
            )
    return paths


def _forward_yaml(device: str, models: dict[str, Path]) -> dict:
    return {
        "task_type": "forward",
        "output_dir": str(TASKS_DIR),
        "device": device,
        "seed": 0,
        "grid": {"dh": 25.0},
        "time": {"dt": 0.002, "nt": 2500},
        "wavelet": {"kind": "ricker", "fm": 5.0, "delay": 0.256, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 80, "depth": 1},
            "receivers": {"step": 1, "depth": 18},
        },
        "physics": {
            "equation": "Acoustic",
            "spatial_order": 8,
            "abcn": 20,
            "free_surface": False,
            "pml_type": "cpmlr",
            "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "models": [{"name": "vp", "path": str(models["true"])}],
    }


def _wavefield_yaml(device: str, models: dict[str, Path]) -> dict:
    spec = _forward_yaml(device, models)
    spec["task_type"] = "wavefield"
    # tighten to a single shot for cleaner snapshots
    spec["geometry"]["sources"] = {"step": 1, "depth": 1, "start": 340, "stop": 341}
    spec["geometry"]["receivers"] = {"step": 4, "depth": 18}
    spec["snapshot_times"] = [400, 900, 1600]
    spec["plot"] = True
    return spec


def _fwi_short_yaml(device: str, models: dict[str, Path]) -> dict:
    return {
        "task_type": "fwi",
        "output_dir": str(TASKS_DIR),
        "device": device,
        "seed": 0,
        "grid": {"dh": 25.0},
        "time": {"dt": 0.002, "nt": 2500},
        "wavelet": {"kind": "ricker", "fm": 5.0, "delay": 0.256, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 80, "depth": 1},
            "receivers": {"step": 1, "depth": 18},
        },
        "physics": {
            "equation": "Acoustic",
            "spatial_order": 8,
            "abcn": 20,
            "free_surface": False,
            "pml_type": "cpmlr",
            "source_type": ["h1"],
            "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(models["smooth"])},
        "obs": {"synthetic_from": {"name": "vp", "path": str(models["true"])}},
        "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},
        "epochs": 3,
        "batchsize": 4,
        "show_every": 1,
    }


def _lsrtm_short_yaml(device: str, models: dict[str, Path]) -> dict:
    spec = _fwi_short_yaml(device, models)
    spec["task_type"] = "lsrtm"
    spec["physics"]["equation"] = "AcousticLSRTM"
    spec["physics"]["receiver_type"] = ["sh1"]
    spec["wavelet"]["fm"] = 10.0
    spec.pop("init_model")
    spec.pop("obs")
    spec.pop("modeling_override", None)
    spec["background_model"] = {"name": "vp", "path": str(models["smooth"])}
    spec["true_model"] = {"name": "vp", "path": str(models["true"])}
    spec["optimizer"] = {"kind": "adam", "lr": 0.01, "eps": 1.0e-22}
    spec["epochs"] = 3
    return spec


def _explicit_forward_yaml(device: str, models: dict[str, Path]) -> dict:
    """Marmousi forward with literal hand-picked shot/receiver positions."""

    spec = _forward_yaml(device, models)
    spec["geometry"] = {
        "kind": "explicit",
        # 4 shots irregularly spaced across the 681-wide grid
        "sources": [[80, 1], [200, 1], [400, 1], [600, 1]],
        # 12 evenly-spaced receivers shared across all shots
        "receivers": [[x, 18] for x in range(20, 661, 60)],
    }
    return spec


def _from_npy_wavelet_forward_yaml(device: str, models: dict[str, Path],
                                   wavelet_path: Path) -> dict:
    """Marmousi forward that reads its wavelet samples from a .npy file."""

    spec = _forward_yaml(device, models)
    spec["wavelet"] = {"kind": "from_npy", "path": str(wavelet_path), "scale": 1.0}
    return spec


def _fwi_with_override_yaml(device: str, models: dict[str, Path]) -> dict:
    """Marmousi FWI where obs is synthesised with a different (lower-freq) wavelet."""

    spec = _fwi_short_yaml(device, models)
    spec["modeling_override"] = {
        "wavelet": {"kind": "ricker", "fm": 3.0, "delay": 0.5, "scale": 1.0},
    }
    return spec


def _verify_state(yaml_path: Path, expected_state: str = "success") -> dict:
    """Walk TASKS_DIR for the freshest task directory matching the YAML's task_type and check status."""

    task_type = yaml.safe_load(yaml_path.read_text())["task_type"]
    candidates = sorted(
        (p for p in TASKS_DIR.iterdir() if p.is_dir() and p.name.startswith(f"{task_type}-")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"No task directory found for task_type='{task_type}' under {TASKS_DIR}")
    status_path = candidates[-1] / "status.json"
    status = json.loads(status_path.read_text())
    actual = status.get("state")
    print(f"  -> task_id={status.get('task_id')} state={actual}")
    if actual != expected_state:
        print(f"  !! status.json contents:\n{status_path.read_text()}")
        raise SystemExit(f"Expected state '{expected_state}', got '{actual}' for {yaml_path}")
    return status


def main() -> int:
    default_marmousi = os.environ.get(
        "SWEEP_MARMOUSI_DIR",
        # Fallback: the canonical location in the sweep main repo.
        "${SWEEP_USER_ROOT}/geophyai/examples/models/marmousi",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--keep-out", action="store_true", help="Keep cli_demo_out/ after running")
    parser.add_argument("--skip-fwi", action="store_true", help="Skip the 3-epoch FWI step")
    parser.add_argument("--skip-lsrtm", action="store_true", help="Skip the 3-epoch LSRTM step")
    parser.add_argument(
        "--marmousi-dir",
        type=Path,
        default=Path(default_marmousi),
        help=(
            "Directory containing true.npy and smooth.npy. "
            "Default: $SWEEP_MARMOUSI_DIR or the sweep main repo's examples/models/marmousi."
        ),
    )
    args = parser.parse_args()

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)

    models = _marmousi_paths(args.marmousi_dir)
    timings: dict[str, float] = {}

    _step(1, "sweep list equations")
    _run_cli("list", "equations")

    _step(2, "sweep show Acoustic")
    _run_cli("show", "Acoustic")

    _step(3, "sweep new forward -> file")
    template_path = DEMO_DIR / "forward_template.yaml"
    _run_cli("new", "forward", "--equation", "Acoustic", "--output", str(template_path))
    print("(first 12 lines of template:)")
    print("\n".join(template_path.read_text().splitlines()[:12]))

    _step(4, "sweep run marmousi forward (line geometry, nt=2500)")
    fwd_yaml = _write_yaml(DEMO_DIR / "forward_marmousi.yaml",
                           _forward_yaml(args.device, models))
    t0 = time.perf_counter()
    _run_cli("run", str(fwd_yaml))
    timings["forward"] = time.perf_counter() - t0
    fwd_status = _verify_state(fwd_yaml)

    _step("4b", "sweep run marmousi forward (EXPLICIT geometry, 4 shots)")
    expl_yaml = _write_yaml(DEMO_DIR / "forward_marmousi_explicit.yaml",
                            _explicit_forward_yaml(args.device, models))
    t0 = time.perf_counter()
    _run_cli("run", str(expl_yaml))
    timings["forward_explicit"] = time.perf_counter() - t0
    expl_status = _verify_state(expl_yaml)

    _step("4c", "sweep run marmousi forward (FROM_NPY wavelet)")
    # save a chirp-like wavelet of length nt=2500 to a .npy file
    nt, dt = 2500, 0.002
    t = np.arange(nt, dtype=np.float32) * dt
    chirp = (np.sin(2 * np.pi * 8.0 * (t - 0.3)) * np.exp(-((t - 0.3) / 0.18) ** 2)).astype(np.float32)
    wavelet_path = DEMO_DIR / "chirp_wavelet.npy"
    np.save(wavelet_path, chirp)
    npy_yaml = _write_yaml(DEMO_DIR / "forward_marmousi_npy_wavelet.yaml",
                           _from_npy_wavelet_forward_yaml(args.device, models, wavelet_path))
    t0 = time.perf_counter()
    _run_cli("run", str(npy_yaml))
    timings["forward_npy_wavelet"] = time.perf_counter() - t0
    npy_status = _verify_state(npy_yaml)

    _step(5, "sweep run marmousi wavefield + snapshots")
    wf_yaml = _write_yaml(DEMO_DIR / "wavefield_marmousi.yaml",
                          _wavefield_yaml(args.device, models))
    t0 = time.perf_counter()
    _run_cli("run", str(wf_yaml))
    timings["wavefield"] = time.perf_counter() - t0
    wf_status = _verify_state(wf_yaml)

    fwi_status = None
    fwi_override_status = None
    if not args.skip_fwi:
        _step(6, "sweep run marmousi FWI (3 epochs, no override)")
        fwi_yaml = _write_yaml(DEMO_DIR / "fwi_marmousi_short.yaml",
                               _fwi_short_yaml(args.device, models))
        t0 = time.perf_counter()
        _run_cli("run", str(fwi_yaml))
        timings["fwi"] = time.perf_counter() - t0
        fwi_status = _verify_state(fwi_yaml)

        _step("6b", "sweep run marmousi FWI (modeling_override: obs at 3 Hz, inversion at 5 Hz)")
        fwi_ov_yaml = _write_yaml(DEMO_DIR / "fwi_marmousi_short_override.yaml",
                                  _fwi_with_override_yaml(args.device, models))
        t0 = time.perf_counter()
        _run_cli("run", str(fwi_ov_yaml))
        timings["fwi_with_override"] = time.perf_counter() - t0
        fwi_override_status = _verify_state(fwi_ov_yaml)

    lsrtm_status = None
    if not args.skip_lsrtm:
        _step(7, "sweep run marmousi LSRTM (3 epochs)")
        lsrtm_yaml = _write_yaml(DEMO_DIR / "lsrtm_marmousi_short.yaml",
                                 _lsrtm_short_yaml(args.device, models))
        t0 = time.perf_counter()
        _run_cli("run", str(lsrtm_yaml))
        timings["lsrtm"] = time.perf_counter() - t0
        lsrtm_status = _verify_state(lsrtm_yaml)

    _step(8, "sweep tasks list")
    _run_cli("tasks", "list", "--output-dir", str(TASKS_DIR))

    _step(9, "sweep tasks status <forward task>")
    _run_cli("tasks", "status", fwd_status["task_id"], "--output-dir", str(TASKS_DIR))

    _step(10, "summary")
    print(f"Demo artifacts under: {DEMO_DIR}")
    print("Per-task timings:")
    for k, v in timings.items():
        print(f"  {k:>22s}: {v:7.1f} s")
    print("Final task states:")
    for label, status in [
        ("forward", fwd_status),
        ("forward_explicit", expl_status),
        ("forward_npy_wavelet", npy_status),
        ("wavefield", wf_status),
        ("fwi", fwi_status),
        ("fwi_with_override", fwi_override_status),
        ("lsrtm", lsrtm_status),
    ]:
        if status is None:
            print(f"  {label:>22s}: SKIPPED")
        else:
            print(f"  {label:>22s}: {status['state']}  ({status['task_id']})")
            if status.get("summary"):
                print(f"             summary: {json.dumps(status['summary'], default=str)}")

    if not args.keep_out:
        shutil.rmtree(DEMO_DIR)
        print(f"\nCleaned {DEMO_DIR}.  Pass --keep-out to retain it.")
    else:
        print(f"\nKept {DEMO_DIR} for inspection.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
