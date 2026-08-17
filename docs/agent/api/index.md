# API reference

The public surface (`sweep_agent.__all__`): the agent loop, the LLM backend
interface, and the tool registry. Auto-generated from source docstrings.

## Agent

::: sweep_agent.Agent
    options:
      show_root_heading: true

## LLM backend

Subclass this to point the agent at any endpoint.

::: sweep_agent.BaseLLM
    options:
      show_root_heading: true

## Tools

Every registered tool is a `Tool`; call it directly via `.fn(params)` (no LLM):

```python
from sweep_agent.tools.inspect import inspect_file, InspectFileParams
inspect_file.fn(InspectFileParams(path="vp_init.npy"))
```

::: sweep_agent.Tool
    options:
      show_root_heading: true
