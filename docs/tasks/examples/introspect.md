# Introspection

```bash
sweep-tasks run examples/synthetic/15_introspect_equations.yaml
```

`task_type: introspect` asks the **installed** solver which equations it
registers and what each one needs, instead of trusting documentation that may
not match your build. No propagator runs, no data is read, no GPU is used. It
writes `output/introspect.json`. Run it first on a new machine.

Each row carries `torch_binding_support` and `torch_binding_available`:
whether a fused CUDA kernel exists for that equation, and whether **this**
build has it compiled. That is how you find out that `backend.impl: c` will
fall over for your equation before a three-hour job finds out for you.

On sweep-solver 0.3.5 it returns **39 equations**. 26 have the fused kernel;
the other 13 — `AcousticVTI` among them, which is why
[16](multiparameter.md) runs on `eager` — are eager-only. Only three take `vp`
alone (`Acoustic`, `Acoustic3D`, `AcousticCurvilinear`):

```json
{"name": "Elastic",       "models": ["vp", "vs", "rho"], ...}
{"name": "AcousticVTI",   "models": ["vp", "epsilon", "delta"], ...}
{"name": "ViscoAcoustic", "models": ["vp", "Q", "omega"], ...}
{"name": "AcousticVRZ",   "models": ["vp", "z"], ...}
{"name": "ElasticTTI",    "models": ["vp0", "vs0", "rho", "epsilon", "delta",
                                     "gamma", "theta", "phi"], ...}
```

## The five actions

| `action` | returns |
|---|---|
| `list_equations` | every equation, with its models and fields — start here |
| `describe_equation` | one equation in full — set `target` |
| `describe_model` | one model of one equation: units, role, defaults — set `target` + `field_or_model` |
| `describe_field` | one wavefield component — same two keys |
| `list_supported_bindings` | which compiled backends this build has |

So to find out what `Elastic` needs before writing a `models:` block:

```bash
sweep-tasks run examples/synthetic/15_introspect_equations.yaml \
    --override action=describe_equation --override target=Elastic
```
