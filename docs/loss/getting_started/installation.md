# Installation

`fwiloss` is a pure-Python package with PyTorch and NumPy as its only
required runtime dependencies.

## From source (recommended while we are pre-release)

```bash
git clone <repo-url> fwiloss
cd fwiloss
pip install -e .
```

With test / dev extras:

```bash
pip install -e ".[test]"
# or
pip install -e ".[dev]"
```

## Verifying the install

```bash
pytest -q
```

All tests should pass; if they don't, please file an issue with the
`pytest -q --tb=short` output.

## Optional dependencies

* `scipy` — required only for some reference comparisons in the tests.
* `mkdocs-material` — to render this documentation site locally.
