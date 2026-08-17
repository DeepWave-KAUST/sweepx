# Getting started

One install gets you the whole stack. Pick the entry point that matches how you
want to work — Python, or plain language.

## Install

```bash
pip install sweepx
```

That pulls the engine ([sweep-solver](../solver/)) and the natural-language layer
([sweep-agent](../agent/)). **Import as `sweep`**, not `sweepx` — same pattern as
`scikit-learn` → `sklearn`:

```python
import sweep
print(sweep.__version__)
```

Python 3.10+. The pure-Python torch / JAX backends need nothing else; the
native-CUDA backend (`impl='c'`) is JIT-compiled against your own PyTorch on
first use and needs a CUDA GPU with `nvcc >= 12.4`:

```python
import sweep
print(sweep.is_torch_binding_available())   # torch + CUDA GPU + nvcc present?
sweep.precompile()                          # optional: build it now (~3-5 min, then cached)
```

## Your first forward model

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

dev    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
solver = PropTorch(Acoustic(device=dev), shape=(96, 128), dh=10.0, dt=2e-3, dev=dev)

wavelet   = ricker(np.arange(600) * 2e-3 - 0.12, f=10.0).astype(np.float32)
sources   = np.array([[64, 2]], dtype=np.int64)
receivers = np.array([[[ix, 4] for ix in range(0, 128, 2)]], dtype=np.int64)
vp        = torch.full((96, 128), 2000.0, device=dev)

gather = solver(wavelet, sources, receivers, models=[vp])   # (1, nt, nrec, 1)
```

## Your first gradient

Every propagator call is an autograd node, so inversion is ordinary PyTorch:

```python
vp   = vp.clone().requires_grad_(True)
pred = solver(wavelet, sources, receivers, models=[vp])
loss = 0.5 * (pred - obs).pow(2).sum()
loss.backward()          # vp.grad is a plain torch Tensor — feed any optimizer
```

## …or say it in plain language

[sweep-agent](../agent/) turns a sentence into a validated run, driven by a **local**
LLM (Ollama on a Mac, vLLM on a GPU node):

```bash
sweep-agent chat
>>> load the Marmousi benchmark model and run a forward — show the shot gather
```

## Where to go next

<div class="grid cards" markdown>

-   :material-waves: __[Solver](../solver/)__

    ---

    Equations, propagators, backends, boundary-saving — and 26 runnable
    notebooks under [Examples](../solver/examples/).

-   :material-message-processing-outline: __[Agent](../agent/)__

    ---

    Tools, install tiers, and how to point it at any OpenAI-compatible LLM.

-   :material-api: __[API reference](../solver/api/)__

    ---

    Every equation, propagator, and operator, documented from source.

</div>
