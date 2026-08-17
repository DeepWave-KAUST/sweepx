# API reference

Auto-generated from source docstrings by mkdocstrings. The rendering options
(signature block, docstring sections, filters) are supplied by the sweepx
umbrella, so this page only needs the `:::` directives.

Point it at your package:

```
::: IMPORT_NAME
    options:
      show_root_heading: true
      members: false
```

::: IMPORT_NAME
    options:
      show_root_heading: true
      members: false

For a class- or module-per-page layout (like sweep-solver's API), add one file
per object under `api/` and list them in `mkdocs.yml`, each with a single
directive, e.g.:

```
# api/core.md
::: IMPORT_NAME.core.SomeClass
```
