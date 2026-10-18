"""Equivalence + integration tests for the three reparam backward modes.

Mirrors ``fwi_workflow-dev``'s ``inr_backward_mode``:
  * ``two_pass_full``  (default) — leaf + second-pass full-graph backward
  * ``two_pass_chunked``         — leaf + row-chunked re-render
  * ``single_step``              — original combined-graph autograd
"""

from pathlib import Path

import numpy as np
import torch
import yaml

from sweep_tasks import load_task, TaskRunner


def _write(spec_dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(spec_dict, sort_keys=False))
    return path


def _tiny_models(tmp_path: Path):
    shape = (32, 48)
    rng = np.random.default_rng(0)
    true_vp = (2200 + 600 * np.linspace(0, 1, shape[0])[:, None]
               * np.ones(shape)).astype(np.float32)
    true_vp += 50.0 * rng.standard_normal(shape).astype(np.float32)
    init_vp = np.full(shape, 2500.0, dtype=np.float32)
    true_path = tmp_path / "true.npy"
    init_path = tmp_path / "init.npy"
    np.save(true_path, true_vp)
    np.save(init_path, init_vp)
    return true_path, init_path


def _spec(tmp_path, init_path, true_path, *, backward_mode, task_id, epochs=1):
    return {
        "task_type": "fwi",
        "output_dir": str(tmp_path / "tasks"),
        "task_id": task_id,
        "grid": {"dh": 10.0},
        "time": {"dt": 0.001, "nt": 200},
        "wavelet": {"kind": "ricker", "fm": 15.0, "delay": 0.08, "scale": 1.0},
        "geometry": {
            "kind": "line",
            "sources": {"step": 8, "depth": 2, "start": 8, "stop": 40},
            "receivers": {"step": 2, "depth": 4, "start": 4, "stop": 44},
        },
        "physics": {
            "equation": "Acoustic", "spatial_order": 8, "abcn": 12,
            "free_surface": False, "pml_type": "cpmlr",
            "source_type": ["h1"], "receiver_type": ["h1"],
        },
        "backend": {"impl": "eager", "use_ckpt": False},
        "init_model": {"name": "vp", "path": str(init_path)},
        "obs": {"synthetic_from": {"name": "vp", "path": str(true_path)}},
        "model_bounds": {"vp": {"min": 1500.0, "max": 4500.0}},
        "optimizer": {"kind": "adam", "lr": 25.0, "eps": 1.0e-22},
        "epochs": epochs, "batchsize": 3, "show_every": 1,
        "reparam": {
            "kind": "velocity_inr",
            "hidden_features": 16, "hidden_layers": 1,
            "hash": {"enabled": True, "levels": 3, "log2_size": 8,
                     "base_resolution": 2, "finest_resolution": 16},
            "lr": 1.0e-3,
            "backward_mode": backward_mode,
        },
    }


def test_two_pass_full_runs(tmp_path):
    true_path, init_path = _tiny_models(tmp_path)
    spec = _spec(tmp_path, init_path, true_path, backward_mode="two_pass_full",
                 task_id="reparam_two_pass_full")
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "full.yaml")))
    assert result.status.state == "success", result.status.error


def test_two_pass_chunked_runs(tmp_path):
    true_path, init_path = _tiny_models(tmp_path)
    spec = _spec(tmp_path, init_path, true_path, backward_mode="two_pass_chunked",
                 task_id="reparam_two_pass_chunked")
    spec["reparam"]["backward_chunk_rows"] = 4
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "chunked.yaml")))
    assert result.status.state == "success", result.status.error


def test_single_step_runs(tmp_path):
    true_path, init_path = _tiny_models(tmp_path)
    spec = _spec(tmp_path, init_path, true_path, backward_mode="single_step",
                 task_id="reparam_single_step")
    result = TaskRunner().run(load_task(_write(spec, tmp_path / "single.yaml")))
    assert result.status.state == "success", result.status.error


def test_three_modes_produce_equivalent_param_grads():
    """Functional core test: run a single solver step under each backward mode
    with deterministic init, and verify all three give the same net.parameters().grad.

    Bypasses the YAML/runner pipeline and exercises the math directly to
    keep this fast and tightly scoped."""
    from sweep_nn import VelocityINR
    torch.manual_seed(42)

    # Tiny vp grid (avoids real wave-equation cost).
    nz, nx = 16, 24
    base_vp = torch.full((nz, nx), 2500.0)

    def fresh_net():
        torch.manual_seed(7)
        return VelocityINR(
            base_vp.clone(),
            vp_std=50.0,
            hidden_features=8, hidden_layers=1,
            hash_levels=3, hash_log2_size=8,
            hash_base_resolution=2, hash_finest_resolution=16,
        )

    # Synthetic "FWI gradient" — what the solver would push back on the
    # rendered vp. Pretend a tiny linear misfit to exercise the math.
    target = torch.randn(nz, nx)

    def loss_fn(vp):
        return ((vp - target) ** 2).sum()

    # ---- single_step ----
    net_a = fresh_net()
    rendered_a = net_a()                # autograd graph attached
    loss_fn(rendered_a).backward()
    grad_a = {n: p.grad.detach().clone() for n, p in net_a.named_parameters()
              if p.grad is not None}

    # ---- two_pass_full ----
    net_b = fresh_net()
    with torch.no_grad():
        leaf_b = net_b().detach().clone()
    leaf_b = leaf_b.requires_grad_(True)
    loss_fn(leaf_b).backward()          # populates leaf_b.grad
    rendered_b = net_b()                # fresh autograd render
    rendered_b.backward(leaf_b.grad)    # push through network
    grad_b = {n: p.grad.detach().clone() for n, p in net_b.named_parameters()
              if p.grad is not None}

    # ---- two_pass_chunked ----
    net_c = fresh_net()
    with torch.no_grad():
        leaf_c = net_c().detach().clone()
    leaf_c = leaf_c.requires_grad_(True)
    loss_fn(leaf_c).backward()
    net_c.backward_velocity_gradient(leaf_c.grad, chunk_rows=4)
    grad_c = {n: p.grad.detach().clone() for n, p in net_c.named_parameters()
              if p.grad is not None}

    # All three should agree to ~machine precision on every parameter.
    for name in grad_a:
        assert torch.allclose(grad_a[name], grad_b[name], atol=1e-5, rtol=1e-4), \
            f"two_pass_full diverged on {name}: max diff {(grad_a[name]-grad_b[name]).abs().max()}"
        assert torch.allclose(grad_a[name], grad_c[name], atol=1e-5, rtol=1e-4), \
            f"two_pass_chunked diverged on {name}: max diff {(grad_a[name]-grad_c[name]).abs().max()}"


def test_default_backward_mode_is_two_pass_full(tmp_path):
    """The reparam default must match fwi_workflow-dev (``inr_backward_mode='full'``)."""
    from sweep_tasks import ReparamSpec
    spec = ReparamSpec()
    assert spec.backward_mode == "two_pass_full"
