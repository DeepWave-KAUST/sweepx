# API reference

The public surface (`sweepx.agent.__all__`): the agent loop, the LLM backend
interface, and the tool registry. Auto-generated from source docstrings.

## Agent

::: sweepx.agent.Agent
    options:
      show_root_heading: true

## LLM backend

Subclass this to point the agent at any endpoint.

::: sweepx.agent.BaseLLM
    options:
      show_root_heading: true

## Tools

Every registered tool is a `Tool`; call it directly via `.fn(params)` (no LLM):

```python
from sweepx.agent.tools.inspect import inspect_file, InspectFileParams
inspect_file.fn(InspectFileParams(path="vp_init.npy"))
```

::: sweepx.agent.Tool
    options:
      show_root_heading: true
