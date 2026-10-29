# sweepx

The toolkit on top of the [`sweep`](../solver/index.md) engine: one package,
`pip install sweepx`, imported one module at a time as `sweepx.<module>`.

## Workflows

<div class="grid cards" markdown>

-   :material-cog-play-outline: __[Tasks](../tasks/index.md)__ · `sweepx.tasks`

    ---

    The YAML task runner: FWI, LSRTM, RTM, forward and wavefield runs from one
    file, on one GPU or many. Command: `sweep-tasks`.

-   :material-message-processing-outline: __[Agent](../agent/index.md)__ · `sweepx.agent`

    ---

    Plain-language control: a local LLM turns a request into a validated run.
    Command: `sweep-agent`.

</div>

## Libraries

<div class="grid cards" markdown>

-   :material-database-arrow-right-outline: __[IO](../io/index.md)__ · `sweepx.io`

    ---

    SEG-Y to gathers: header index, data plans, prefetched reads, models,
    acquisition geometry.

-   :material-chart-bell-curve: __[Loss](../loss/index.md)__ · `sweepx.loss`

    ---

    Misfit functions for inversion, from L2 to envelope, phase and optimal
    transport.

-   :material-vector-polyline: __[NN](../nn/index.md)__ · `sweepx.nn`

    ---

    Implicit FWI, the model as a network: SIREN, hash encoding, priors.

-   :material-chart-timeline-variant: __[Opt](../opt/index.md)__ · `sweepx.opt`

    ---

    FWI optimizers: L-BFGS, truncated Newton / Gauss-Newton, Adam,
    Shampoo / SOAP, preconditioners.

</div>

To have an AI coding assistant write sweepx tasks with the real schema, install
the [sweepx Agent Skill](agent-skills.md).

## How the modules depend on each other

```
workflows   sweepx.agent  →  sweepx.tasks
libraries   sweepx.io    sweepx.loss    sweepx.nn    sweepx.opt
engine      sweep  (pip name: sweep-solver)
```

A module imports only from the layers below it: the agent drives the task
runner, which builds on the engine and on `sweepx.io`, `sweepx.loss` and
`sweepx.nn`. The libraries import neither each other nor the engine, so
`sweepx.loss`, `sweepx.nn` or `sweepx.opt` drop into any PyTorch inversion
loop. CI checks these rules with import-linter.
