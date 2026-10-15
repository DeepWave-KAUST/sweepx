# Installation

```bash
pip install sweepx
```

`sweepx.loss` is part of `sweepx`; on its own it needs only PyTorch and NumPy
(SciPy for the graph-space OT loss, which `sweepx` installs).

## From source

```bash
git clone https://github.com/DeepWave-KAUST/sweepx.git
cd sweepx
pip install -e ".[test]"
pytest tests/loss
```

All tests should pass; if they don't, please file an issue with the
`pytest tests/loss --tb=short` output.
