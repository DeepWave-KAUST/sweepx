# Installation

```bash
pip install sweep-io
```

That needs only NumPy. It also comes with `pip install sweepx`, through sweep-tasks.

The optional dependencies are imported only by the code that needs them:

| Extra | Adds | For |
|---|---|---|
| `segy` | `segyio` | `read_segy` / `write_segy`, whole-file reads and writes |
| `hdf5` | `h5py` | `.h5` velocity models |
| `torch` | `torch` | `ShotGatherDataset`, `CRGBatchDataset`, `CUDAPrefetcher` |

```bash
pip install "sweep-io[segy,hdf5,torch]"
```

The byte-offset readers and everything on the index → plan → reader path need none of
them. The parallel CRG plan build (`build_crg_plan_from_segy(..., use_mpi=True)`) also
needs `mpi4py`.

Verify the install:

```python
import sweep_io
print(sweep_io.__version__)
```
