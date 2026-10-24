"""Suite-wide test settings.

Inside a Slurm job every ``TaskRunner.run`` asks ``scontrol`` about the job
(two calls, 10 s timeout each) for the run metadata, which made the task tests
several times slower on a cluster than on a laptop or CI runner.
``SLURM_JOB_ID`` is removed for the whole session, before any fixture runs a
task and for every subprocess a test starts; the run-metadata tests set it
themselves.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_slurm_job_metadata():
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("SLURM_JOB_ID", raising=False)
        yield
