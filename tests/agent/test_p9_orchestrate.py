"""P9: compare_equation_wavefields — one-call multi-equation wavefront comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sweep_agent.tools import registry


@pytest.mark.timeout(240)
def test_compare_equation_wavefields(tmp_path):
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    sh = (40, 56)
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full(sh, 2500.0, dtype=np.float32))
    eps = tmp_path / "eps.npy"
    np.save(eps, np.full(sh, 0.2, dtype=np.float32))
    dlt = tmp_path / "delta.npy"
    np.save(dlt, np.full(sh, 0.08, dtype=np.float32))
    res = registry.get("compare_equation_wavefields").invoke({
        "vp_path": str(vp), "equations": ["Acoustic", "AcousticVTI"],
        "extra_models": {"epsilon": str(eps), "delta": str(dlt)},
        "out_path": str(tmp_path / "cmp.png"),
        "dh": 12.5, "dt": 0.001, "nt": 50, "fm": 12.0, "abcn": 10, "spatial_order": 4,
        "source_step": 25, "source_depth": 20, "receiver_step": 5, "receiver_depth": 2,
        "device": "cpu", "output_dir": str(tmp_path / "runs"),
    })
    assert res.ok and "error" not in res.value, res.value
    assert Path(res.value["image_path"]).is_file()
    assert len(res.value["task_dirs"]) == 2


def test_compare_equation_missing_model(tmp_path):
    """AcousticVTI needs epsilon/delta — empty pool must error before running."""
    pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((40, 56), 2500.0, dtype=np.float32))
    res = registry.get("compare_equation_wavefields").invoke({
        "vp_path": str(vp), "equations": ["Acoustic", "AcousticVTI"],
        "out_path": str(tmp_path / "cmp.png"), "output_dir": str(tmp_path / "runs"), "nt": 50,
    })
    assert "error" in res.value
    assert "epsilon" in res.value["error"] or "delta" in res.value["error"]
