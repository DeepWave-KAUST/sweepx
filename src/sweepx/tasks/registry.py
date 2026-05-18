"""Single source of truth mapping task_type strings to schema classes."""

from sweep_tasks.schemas import (
    ForwardSpec,
    FWISpec,
    IntrospectSpec,
    LSRTMSpec,
    RTMSpec,
    WavefieldSpec,
)

TASK_TYPES: dict[str, type] = {
    "introspect": IntrospectSpec,
    "forward": ForwardSpec,
    "wavefield": WavefieldSpec,
    "fwi": FWISpec,
    "lsrtm": LSRTMSpec,
    "rtm": RTMSpec,
}
