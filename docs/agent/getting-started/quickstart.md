# Quickstart

Start a chat — it auto-detects a running Ollama (`:11434`) or vLLM
(`:8000`/`:8001`), picks a 7B model, and pulls it on first run:

```bash
sweep-agent chat
```

```text
>>> load the Marmousi benchmark model and run a forward — show the shot gather
>>> here is vp_init.npy — run a 2-D acoustic forward and show me the shot gather
>>> make an elastic model (vp/vs/rho) and run an Elastic forward
>>> :reset              # clear conversation history
```

Other entry points:

```bash
sweep-agent ui          # same agent in a browser (needs [ui] + a running LLM) → :7860
sweep-agent tools       # list the ~30 tools — no LLM/GPU needed; --json emits tool specs
```

Every tool is also a plain function (`.fn`, with a pydantic params model) — handy
for scripts and tests, no LLM required:

```python
from sweep_agent.tools.inspect import inspect_file, InspectFileParams
print(inspect_file.fn(InspectFileParams(path="vp_init.npy")))
```
