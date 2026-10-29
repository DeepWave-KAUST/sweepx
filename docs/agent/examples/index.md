# Examples

Prompts that work on a plain install (`pip install sweepx` + a local LLM) —
nothing to download:

```text
load the Marmousi benchmark model and run a forward — show the shot gather
here is vp_init.npy — run a 2-D acoustic forward and show me the shot gather
make an elastic model (vp/vs/rho) and run an Elastic forward
list the equations you can model
inspect obs.segy and tell me its geometry      # needs sweepx[segy]
```

The web UI has one-click **demo chips** for similar requests:

```bash
pip install "sweepx[agent-ui]"
sweep-agent ui          # → open :7860
```

The agent can also build and run FWI / LSRTM tasks through `sweepx.tasks` from
plain language (e.g. *"run a 2-D acoustic FWI on Marmousi, 30 epochs, Adam lr 25"*).

`examples/agent/demos/` has a script per workflow (forward + gather, elastic and
anisotropic wavefields, topography, FWI with Adam or L-BFGS, …), each sending one
natural-language request to a running LLM server; `outputs/` holds what they
produce.
