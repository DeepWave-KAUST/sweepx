# Installation

```bash
pip install sweepx
```

`sweepx.opt` is part of `sweepx`; on its own it needs only PyTorch and NumPy.

## From source

```bash
git clone https://github.com/DeepWave-KAUST/sweepx.git
cd sweepx
pip install -e ".[test]"
pytest tests/opt
```

## Verifying the install

```bash
python -c "import sweepx.opt; print(sweepx.opt.__version__)"
```
