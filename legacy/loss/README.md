# fwiloss

A PyTorch library of misfit (loss) functions for **Full Waveform Inversion (FWI)**.

The package collects, from a single API, the loss functions that have been
proposed in the FWI literature - from classical least-squares to optimal-transport,
adaptive matching filters, and instantaneous-phase / envelope variants - and
exposes them as `torch.nn.Module` so they can be dropped into any PyTorch-based
FWI workflow (e.g. the `sweep` propagator).

## Data convention

All losses accept tensors with the shape used throughout the `sweep` ecosystem:

```
(nshots, nt, nreceivers, nchannel)
```

* `nshots`     - number of independent source experiments in the mini-batch.
* `nt`         - number of time samples (axis = -3, the **time axis**).
* `nreceivers` - number of receivers per shot.
* `nchannel`   - number of recorded components (1 for pressure, 2/3 for elastic, ...).

For convenience every loss also accepts a plain `(nt,)` or `(nt, nrec)` tensor
(treated as a single trace / single shot).

## Quick start

```python
import torch
from fwiloss import L2Loss

syn = torch.randn(2, 1024, 64, 1, requires_grad=True)   # (nshots, nt, nrec, nchan)
obs = torch.randn(2, 1024, 64, 1)

loss_fn = L2Loss(reduction="mean")
loss = loss_fn(syn, obs)
loss.backward()
```

## Implemented losses

See [`docs/report.md`](docs/report.md) (also rendered on the mkdocs site,
see below) for the full list of misfit formulas with **DOI-linked
references**. Each loss also has its own page under
[`docs/losses/`](docs/losses/).

## Documentation site

The repository ships with an [MkDocs](https://www.mkdocs.org/) +
[Material](https://squidfunk.github.io/mkdocs-material/) site:

```bash
pip install mkdocs mkdocs-material
mkdocs serve   # http://127.0.0.1:8000
# or
mkdocs build   # outputs to ./site
```

The site is configured in `mkdocs.yml`; pages live under `docs/`.

## Installation

```bash
pip install -e .
# or, with dev/test extras
pip install -e .[test]
```

## Running the tests

```bash
pytest -q
```

## License

MIT
