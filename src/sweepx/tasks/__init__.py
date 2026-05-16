"""Sweep task layer: serialisable task specs + synchronous TaskRunner.

Phase 1 only ships local synchronous execution; a future RemoteRunner can
subclass TaskRunner without changing the `run(spec) -> TaskResult` contract.
"""

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
    QCSpec,
    ReparamSpec,
    TaskSpec,
    WavefieldSpec,
)
from sweep_tasks.yaml_io import dump_task, load_task, new_template

__all__ = [
    "BaseTaskSpec",
    "DataPlanSpec",
    "ForwardSpec",
    "FWISpec",
    "IntrospectSpec",
    "LocalModelWindowSpec",
    "LSRTMSpec",
    "ModelPlanSpec",
    "QCSpec",
    "ReparamSpec",
    "TASK_TYPES",
    "TaskResult",
    "TaskRunner",
    "TaskSpec",
    "TaskStatus",
    "WavefieldSpec",
    "dump_task",
    "load_task",
    "new_template",
]
