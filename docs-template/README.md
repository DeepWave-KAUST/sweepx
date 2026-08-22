# docs-template — the per-package docs skeleton

Copy this into a new `sweep-*` package so its docs match the ecosystem. The
**style** (theme, CSS, API rendering) is deliberately NOT here — it lives once
in the sweepx umbrella (`../mkdocs.yml`, `../overrides/`, `../docs/stylesheets/`)
and is applied at build time. Packages carry **content only**.

## Add a package to the docs in 5 steps

1. Copy `docs/` and `mkdocs.yml` from here into your package repo.
2. Replace the placeholders everywhere:
   - `PACKAGE_NAME` → the pip name (e.g. `sweep-tasks`)
   - `IMPORT_NAME`  → the python import (e.g. `sweep_tasks`)
   - fill in the prose, code, and examples.
3. Point the API page(s) at your package: `::: your_import_name`.
4. In the **sweepx** repo, add one line to `nav:` in `mkdocs.yml`:
   ```yaml
   - Tasks: '!import https://github.com/DeepWave-KAUST/sweep-tasks?branch=main&docs_dir=docs/*'
   ```
5. Make sure the docs build env can `pip install PACKAGE_NAME` — mkdocstrings
   imports the package to render its API.

That's it. Do NOT add a theme, plugins, or mkdocstrings options to the
package's `mkdocs.yml`; keeping them out is what keeps every package identical.

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
