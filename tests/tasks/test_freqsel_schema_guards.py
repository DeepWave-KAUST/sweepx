"""What a frequency-selection FWI spec accepts, and what it must refuse."""
import numpy as np
import pytest

def _freqsel_fwi_doc(**extra):
    doc = dict(
        task_type="fwi", output_dir="./x", task_id="t",
        grid={"dh": 12.5}, time={"dt": 0.001, "nt": 100},
        physics={"equation": "Acoustic"},
        init_model={"name": "vp", "constant": 1500.0, "shape": [20, 30]},
        source_encoding={
            "enabled": True, "mode": "frequency_selection",
            "frequency": {"coeff_shards": "s.npz", "probe_samples": 100,
                          "k_lo": 4, "k_hi": 8, "steady_samples": 50},
        },
        optimizer={"kind": "adam", "lr": 25.0}, epochs=1,
    )
    doc.update(extra)
    return doc


def test_freqsel_rejects_freeze_top_n_rows():
    """The freqsel loop pins water by value, so a row count would be a no-op."""
    from pydantic import TypeAdapter
    from sweep_tasks.schemas import TaskSpec

    ta = TypeAdapter(TaskSpec)
    ta.validate_python(_freqsel_fwi_doc())                    # baseline parses
    with pytest.raises(Exception, match="freeze_top_n_rows is not supported"):
        ta.validate_python(_freqsel_fwi_doc(freeze_top_n_rows=37))


def test_freqsel_makes_wavelet_geometry_obs_optional():
    """The doc above has none of the three and must still validate."""
    from pydantic import TypeAdapter
    from sweep_tasks.schemas import TaskSpec

    spec = TypeAdapter(TaskSpec).validate_python(_freqsel_fwi_doc())
    assert spec.wavelet is None and spec.geometry is None and spec.obs is None
