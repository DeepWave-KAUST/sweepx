"""`FromPlanGeometry.rotation_metadata` is optional — the code must honour that."""
import json

import numpy as np
import pytest

from sweep_tasks._helpers.plan_apply import resolve_rotation_frame


def test_none_gives_an_identity_frame():
    """The 2-D case: plan coordinates ARE the model frame, so nothing moves."""
    f = resolve_rotation_frame(None)
    xy = np.array([[0.0, 0.0], [3237.0, 0.0], [-125.5, 42.25]])
    assert np.allclose(f.to_model(xy), xy)


def test_a_real_metadata_file_still_loads(tmp_path):
    p = tmp_path / "rot.json"
    p.write_text(json.dumps({"origin_xy": [1000.0, 2000.0],
                             "rotation_matrix": [[0.0, 1.0], [-1.0, 0.0]]}))
    f = resolve_rotation_frame(str(p))
    # origin subtracted then rotated: (1000,2000) -> (0,0)
    assert np.allclose(f.to_model(np.array([[1000.0, 2000.0]])), [[0.0, 0.0]])
    assert not np.allclose(f.to_model(np.array([[1100.0, 2000.0]])), [[100.0, 0.0]])


def test_identity_is_not_silently_applied_to_a_bad_path():
    with pytest.raises((FileNotFoundError, OSError)):
        resolve_rotation_frame("/nonexistent/rot.json")
