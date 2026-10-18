# Installation

`sweep-opt` is pure Python; the core only needs `torch` and `numpy`.

## Standard install

```bash
git clone https://github.com/DeepWave-KAUST/sweep-opt.git
cd sweep-opt
pip install -e .
```

## Optional extras

```bash
pip install -e .[test]    # adds pytest + scipy (used by some unit tests)
pip install -e .[docs]    # adds mkdocs + mkdocs-material
pip install -e .[dev]     # docs + test + ruff + black
```

## Verifying the install

```bash
python -c "import sweep_opt; print(sweep_opt.__version__)"
pytest -q
mkdocs build --strict
```
