# AI assistants (Agent Skills)

The repository ships an [Agent Skill](https://agentskills.io) for sweepx in
`skills/sweepx/`. A skill is a folder of Markdown instructions and helper
scripts that an AI coding assistant loads when a task needs it. With it
installed, Claude Code, Claude.ai, Codex and other assistants that read the
Agent Skills format write sweepx task YAMLs and Python with the real schema
instead of guessing it: which keys each task type accepts and what the
validator refuses, multiscale stages, the water layer, frequency selection,
INR / hash reparameterization, the SEG-Y → plan → FWI pipeline, multi-GPU and
resume, the `sweepx.io` / `sweepx.loss` / `sweepx.opt` / `sweepx.nn`
libraries, and the settings that load but silently do the wrong thing.

This is not the [Agent](../agent/index.md) module. `sweep-agent` is a chat
front end that runs a local LLM inside sweepx; the skill is documentation that
an assistant you already use reads while it helps you write code.

The skill is written for the current release (0.4.1, with sweep 0.3.6). It is
not part of the wheel; take it from the repository. For the solver itself
(equations, propagators, boundaries, gradients and memory), the sweep
repository has a companion skill, `skills/sweep-solver`; install both.

## Contents

| Path | What it holds |
|---|---|
| `SKILL.md` | Entry point: the task-file → run-directory workflow, install, which task type does what, a runnable forward example, the outputs of each task type, a list of pitfalls, and pointers to the references |
| `references/spec-schema.md` | Every task type and sub-model: fields, types, defaults, required keys (generated from the schema) |
| `references/workflows.md` | FWI and the Python API, the full run directory, multiscale stages, water, wavefield, RTM / LSRTM, the field-data pipeline, frequency selection, multi-GPU, resume, and what the validator refuses |
| `references/io.md` | SEG-Y index, plans and reader, geometry, data layouts |
| `references/loss.md` | YAML loss kinds and the `sweepx.loss` library |
| `references/opt.md` | Optimizers, step-size units, bounds and preconditioners, in a YAML and in Python |
| `references/nn.md` | `VelocityINR`, hash grids, the `reparam:` block |
| `references/datasets.md` | Benchmark models, embedded demos, the Viking field line |
| `references/agent.md` | `sweep-agent`: chat, UI, LLM backends, tools as Python functions |
| `scripts/validate_spec.py` | Validates a task YAML with sweepx's own loader, without running it; `--preflight` also loads models and geometry and checks CFL, points per wavelength and ignored settings |
| `scripts/gen_spec_reference.py` | Regenerates `references/spec-schema.md` |

The assistant reads `SKILL.md` first and opens a reference file only when the
task needs it.

## Get the folder

Clone only the skill:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/DeepWave-KAUST/sweepx.git sweepx-skill
git -C sweepx-skill sparse-checkout set skills/sweepx
```

The folder is then `sweepx-skill/skills/sweepx`. A full clone of the
repository works as well.

## Install

Copy the folder, keeping its name `sweepx`, to where your assistant looks for
skills.

**Claude Code**: for all your projects, or only for one project:

```bash
mkdir -p ~/.claude/skills && cp -r sweepx-skill/skills/sweepx ~/.claude/skills/
mkdir -p .claude/skills && cp -r sweepx-skill/skills/sweepx .claude/skills/   # run in the project
```

**Codex**: the same, with `~/.agents/skills/` or the project's
`.agents/skills/`.

**Claude.ai**: zip the folder and upload the zip in the skills section of the
settings.

```bash
cd sweepx-skill/skills && zip -r sweepx.zip sweepx
```

**Other assistants**: give them `SKILL.md` and let them open the reference
files it points to.

## Use

The assistant decides from the skill's description when to load it: writing
or fixing a task YAML, running `sweepx tasks` / `sweep-tasks` commands,
setting up FWI, reading SEG-Y with `sweepx.io`, choosing a misfit, optimizer
or reparameterization, reading a run directory, or driving `sweep-agent`. You
can also ask for it by name, for example "use the sweepx skill to write a
multiscale FWI task for Marmousi with a water layer".

Before running a task it wrote, the assistant can check it without running it:

```bash
python ~/.claude/skills/sweepx/scripts/validate_spec.py task.yaml --preflight
```

## Keeping it correct

Every code block in the skill runs in the test suite:
`tests/test_skill_snippets.py` runs each Markdown file's blocks in a fresh
working directory with `skills/run_snippets.py`, so a schema or API change
that breaks an example fails the tests. To check the skill by hand:

```bash
python skills/run_snippets.py skills/sweepx
```

After a change to the task schema, regenerate the schema reference:

```bash
cd skills/sweepx/scripts && python gen_spec_reference.py -o ../references/spec-schema.md
```
