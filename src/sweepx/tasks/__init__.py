"""Sweep task layer: serialisable task specs + synchronous TaskRunner.

Phase 1 only ships local synchronous execution; a future RemoteRunner can
subclass TaskRunner without changing the `run(spec) -> TaskResult` contract.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from sweep_tasks.registry import TASK_TYPES
from sweep_tasks.runner import TaskResult, TaskRunner, TaskStatus
from sweep_tasks.schemas import (
    BaseTaskSpec,
    DataPlanSpec,
    ForwardSpec,
    FWISpec,
    IntrospectSpec,
    LocalModelWindowSpec,
    LSRTMSpec,
    ModelPlanSpec,
    PostFilterImageSpec,
    QCSpec,
    ReparamSpec,
    RTMImagingSpec,
    RTMSpec,
    TaskSpec,
    WavefieldSpec,
)
from sweep_tasks.yaml_io import (
    dump_task,
    load_task,
    load_task_from_dict,
    new_template,
)

try:
    __version__ = _pkg_version("sweep-tasks")
except PackageNotFoundError:  # a source tree with nothing installed
    __version__ = "unknown"

__all__ = [
    "BaseTaskSpec",
    "DataPlanSpec",
    "ForwardSpec",
    "FWISpec",
    "IntrospectSpec",
    "LocalModelWindowSpec",
    "LSRTMSpec",
    "ModelPlanSpec",
    "PostFilterImageSpec",
    "QCSpec",
    "ReparamSpec",
    "RTMImagingSpec",
    "RTMSpec",
    "TASK_TYPES",
    "TaskResult",
    "TaskRunner",
    "TaskSpec",
    "TaskStatus",
    "WavefieldSpec",
    "__version__",
    "dump_task",
    "load_task",
    "load_task_from_dict",
    "new_template",
]
