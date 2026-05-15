# Trace-normalised L2 and Global-correlation (NCC) misfits

## Definitions

Let $\hat d(t) = d(t)/\|d\|_2$ denote the per-trace L2-normalised waveform.

**Global-correlation / normalised cross-correlation (NCC):**

$$
\mathcal J_{\mathrm{NCC}}(m) \;=\; \sum_{\mathrm{trace}}\bigl(1 - \langle\hat d_{\mathrm s},\hat d_{\mathrm o}\rangle\bigr).
$$

With `offset_one=False` you instead get $-\sum\langle\hat d_{\mathrm s},
\hat d_{\mathrm o}\rangle$.

**Trace-normalised L2** (Choi & Alkhalifah 2012, eq. 9):

$$
\mathcal J_{\mathrm{nL2}}(m) \;=\; \tfrac{1}{2}\sum_{\mathrm{trace}}\sum_t \bigl(\hat d_{\mathrm s}(t) - \hat d_{\mathrm o}(t)\bigr)^2.
$$

Because $\|\hat d\|_2 = 1$ per trace,

$$
\mathcal J_{\mathrm{nL2}} \;=\; \sum_{\mathrm{trace}}\bigl(1 - \langle\hat d_{\mathrm s},\hat d_{\mathrm o}\rangle\bigr) \;=\; \mathcal J_{\mathrm{NCC}}\,,
$$

so the two functionals are **identical**.  We expose both APIs because the
literature uses both names interchangeably and we want the user to be able
to write whichever feels natural.

## Properties

* **Amplitude invariance** — both losses are invariant to *any positive
  per-trace rescaling* of either input.  Tested explicitly.
* **Cycle-skipping is *not* mitigated** by NCC alone; this misfit is for the
  amplitude problem, not the kinematic one.

## When to use

* Field data where source signature / receiver coupling makes absolute
  amplitudes unreliable.
* As a building block for envelope-corr / instantaneous-phase-corr
  misfits.

## API

```python
from fwiloss import GlobalCorrelationLoss, TraceNormalizedL2Loss
GlobalCorrelationLoss(offset_one=True)(syn, obs)
TraceNormalizedL2Loss()(syn, obs)
```

## Tests

`tests/test_correlation.py` checks:

* invariance under per-trace positive rescaling,
* perfect match → 0,
* perfectly anti-correlated traces → 2 per trace,
* `offset_one=False` returns $-1$ for a perfect match,
* `TraceNormalizedL2Loss == GlobalCorrelationLoss(offset_one=True)` exactly,
* autograd flows.

## References

* Choi, Y. & Alkhalifah, T. (2012). *Application of multi-source waveform
  inversion to marine streamer data using the global correlation norm.*
  **Geophysical Prospecting** 60 (4), 748-758.
  doi:[10.1111/j.1365-2478.2012.01079.x](https://doi.org/10.1111/j.1365-2478.2012.01079.x)
* Routh, P., Krebs, J., Lazaratos, S., et al. (2011). *Encoded simultaneous
  source full-wavefield inversion for spectrally-shaped marine streamer
  data.* SEG Tech. Progr. Expanded Abstracts, pp. 2433-2438.
  doi:[10.1190/1.3627696](https://doi.org/10.1190/1.3627696)
