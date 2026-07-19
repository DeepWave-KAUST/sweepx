"""P15: FWI parameter coverage — optimizers, loss functionals, vp bounds, and a
multi-parameter (elastic) inversion spec; plus a tiny adam/mse run."""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def _init_true(tmp_path, shape=(48, 60)):
    true = np.full(shape, 1500.0, dtype=np.float32)
    true[shape[0] // 2:, :] = 2200.0
    init = np.full(shape, 1500.0, dtype=np.float32)
    tp, ip = tmp_path / "true.npy", tmp_path / "init.npy"
    np.save(tp, true)
    np.save(ip, init)
    return str(ip), str(tp)


def _build_fwi(tmp_path, **over):
    ip, tp = _init_true(tmp_path)
    args = {
        "init_model_path": ip, "synthetic_true_vp_path": tp,
        "dh": 10.0, "dt": 1.5e-3, "nt": 120, "fm": 12.0,
        "optimizer": "adam", "lr": 20.0, "loss": "mse", "epochs": 2,
        "source_step": 40, "receiver_step": 2,
        "device": "cpu", "backend_impl": "eager", "use_compile": False,
        "output_dir": str(tmp_path / "runs"), "task_id": "fwi",
    }
    args.update(over)
    return registry.get("build_fwi_spec").invoke(args)


@pytest.mark.parametrize("optimizer,lr", [("adam", 20.0), ("sgd", 5.0), ("lbfgs", 1.0)])
def test_fwi_optimizers(tmp_path, optimizer, lr):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = _build_fwi(tmp_path, optimizer=optimizer, lr=lr, task_id=f"opt_{optimizer}")
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["optimizer"]["kind"] == optimizer


@pytest.mark.parametrize("loss", ["mse", "l1", "huber", "trace_cosine"])
def test_fwi_losses(tmp_path, loss):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = _build_fwi(tmp_path, loss=loss, task_id=f"loss_{loss}")
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["loss"]["kind"] == loss


def test_fwi_vp_bounds(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    r = _build_fwi(tmp_path, vp_min=1450.0, vp_max=2600.0, task_id="bounds")
    assert r.ok and "error" not in r.value, r.value
    # bounds land somewhere in the spec (model_bounds / clamp); just assert the
    # values are present in the serialized spec.
    blob = str(r.value["spec"])
    assert "1450" in blob and "2600" in blob


def test_fwi_elastic_multiparam(tmp_path):
    """Elastic FWI needs vs + rho alongside the init vp."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    shape = (48, 60)
    for n, v in (("vs", 1200.0), ("rho", 2200.0)):
        np.save(tmp_path / f"{n}.npy", np.full(shape, np.float32(v)))
    r = _build_fwi(
        tmp_path, equation="Elastic",
        extra_init_models={"vs": str(tmp_path / "vs.npy"), "rho": str(tmp_path / "rho.npy")},
        task_id="el_fwi",
    )
    assert r.ok and "error" not in r.value, r.value
    assert r.value["spec"]["physics"]["equation"] == "Elastic"
    assert [m["name"] for m in r.value["spec"]["init_models"]] == ["vp", "vs", "rho"]


@pytest.mark.timeout(180)
def test_fwi_run_tiny(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    b = _build_fwi(tmp_path, epochs=3, nt=200, task_id="run")
    assert b.ok and "error" not in b.value, b.value
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 150})
    assert r.value.get("state") == "success", r.value
