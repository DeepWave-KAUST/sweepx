# Examples

Prompts that work on the **base install** (`pip install sweep-agent` + a local
LLM) — no `sweep-tasks`, nothing to download:

```text
load the Marmousi benchmark model and run a forward — show the shot gather
here is vp_init.npy — run a 2-D acoustic forward and show me the shot gather
make an elastic model (vp/vs/rho) and run an Elastic forward
list the equations you can model
inspect obs.segy and tell me its geometry
```

The same prompts are one-click **demo chips** in the web UI:

```bash
pip install "sweep-agent[ui]"
sweep-agent ui          # → open :7860
```

Once `sweep-tasks` is installed, the agent can also build and run FWI / LSRTM
tasks from plain language (e.g. *"run a 2-D acoustic FWI on Marmousi, 30 epochs,
Adam lr 25"*).
