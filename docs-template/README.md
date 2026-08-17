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
