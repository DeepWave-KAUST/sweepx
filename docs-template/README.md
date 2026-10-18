# docs-template — the docs skeleton for a sweepx module

Copy this when a new module joins sweepx (`src/sweepx/<module>/`) so its docs match
the rest of the site. The **style** (theme, CSS, API rendering) is deliberately NOT
here — it lives once in this repo (`../mkdocs.yml`, `../overrides/`,
`../docs/stylesheets/`). A module's docs carry **content only**.

## Add a module to the docs in 4 steps

1. Copy `docs/` from here to `docs/<module>/` in this repo.
2. Replace the placeholders everywhere:
   - `PACKAGE_NAME` → `sweepx` (the pip name; extras as `sweepx[extra]`)
   - `IMPORT_NAME`  → the python import (e.g. `sweepx.tasks`)
   - fill in the prose, code, and examples.
3. Point the API page(s) at the module: `::: sweepx.<module>`.
4. Add the module's nav block under the `sweepx` section of `nav:` in `../mkdocs.yml`,
   every path prefixed with `<module>/` (copy this template's `mkdocs.yml` nav and
   prefix it), and give it a one-line note under a group of `extra.tab_menus.sweepx`
   so it shows in the sweepx tab's dropdown. Add a card for it to
   `../docs/sweepx/index.md`.

The docs build installs this repo (`pip install -e .` in `.github/workflows/docs.yml`),
so mkdocstrings can import the new module with no further setup.

The solver is the one section that still comes from another repository
(`!import https://github.com/DeepWave-KAUST/sweep?...` in `../mkdocs.yml`, mapped
under `extra.section_repos`): its repo carries `docs/` plus a thin nav-only
`mkdocs.yml`, which is what this template's `mkdocs.yml` is.

Do NOT add a theme, plugins, or mkdocstrings options to a nav-only `mkdocs.yml`;
keeping them out is what keeps every section identical.

## Structure (mirrors sweep-solver)

```
docs/
├─ index.md                 # overview + grid cards
├─ getting-started/
│  ├─ installation.md
│  └─ quickstart.md
├─ user-guide/
│  └─ index.md              # add more pages, list them in nav
├─ examples/
│  └─ index.md
└─ api/
   └─ index.md              # ::: your_import_name  (mkdocstrings)
```

## Keeping the landing page current

The umbrella's homepage carries a hand-curated **What's new** strip — six cards
in a 3 × 2 grid, in `docs/index.md`. It is written by hand on purpose: the copy
is a pitch, not a changelog, and it is short enough that generating it from the
package repos would cost more than it saves.

**When a package ships something user-visible**, add a card at the top of the
grid and drop the oldest, so the count stays at six. The badge states where the
feature actually is — readers use it to decide whether `pip install` already
gives it to them:

| badge | meaning |
|---|---|
| `v0.1.0`, `agent 0.0.3` | in a release, available from PyPI |
| `dev` | merged, not released yet — add the `sweep-new__badge--dev` class |
| `docs` | a docs-site change, no package version |

This site builds from each package's development branch, so a feature stays
`dev` until it is released.
