# sweep-tasks

The **typed YAML task layer** for [`sweep`](../solver/) — the front door for
**CLI-** and **LLM-driven** FWI / LSRTM / RTM / forward / wavefield runs.

If you just want a Python loop around the wave solver, use `sweep` directly.
`sweep-tasks` is for when you want to describe a job in **YAML** (or JSON, from
an LLM) and have it executed **reproducibly** — one process or `torchrun`
multi-rank, shot-parallel by default, with `status.json` + `checkpoint.pt` +
figures written per run.

Installed with `pip install sweep-tasks` (also bundled by `pip install sweepx`)
→ `import sweep_tasks`.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __[Getting started](getting-started/installation.md)__

    ---

    Install, then run a Marmousi FWI in two commands — nothing to download.

-   :material-book-open-variant-outline: __[User guide](user-guide/index.md)__

    ---

    Task types, the YAML schema, `ModelRef`, the CLI, and distributed runs.

-   :material-notebook-outline: __[Examples](examples/index.md)__

    ---

    Synthetic (no downloads) and field (SEG-Y) task configs.

-   :material-api: __[API reference](api/index.md)__

    ---

    `TaskRunner`, `load_task`, and the Pydantic task schemas.

</div>
