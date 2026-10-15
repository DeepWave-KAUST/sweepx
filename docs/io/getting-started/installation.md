# Installation

```bash
pip install sweepx
```

`sweepx.io` is part of `sweepx`; on its own it needs only NumPy and SciPy. The
optional dependencies are imported only by the code that needs them:

| Extra | Adds | For |
|---|---|---|
| `segy` | `segyio` | `read_segy` / `write_segy`, whole-file reads and writes |
| `hdf5` | `h5py` | `.h5` velocity models |

```bash
pip install "sweepx[segy,hdf5]"
```

The byte-offset readers and everything on the index → plan → reader path need none of
them. The parallel CRG plan build (`build_crg_plan_from_segy(..., use_mpi=True)`) also
needs `mpi4py`.

Verify the install:

```python
import sweepx.io
print(sweepx.io.__version__)
```
