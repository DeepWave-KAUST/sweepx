# Optimal Transport of the Matching Filter (OTMF)

## Definition

Sun & Alkhalifah (2019) combine adaptive matching-filter and OT
ideas:

1. Compute the Wiener filter $w(\tau)$ that maps $d_{\mathrm s}\ast w$ to
   $d_{\mathrm o}$ (same regularised Wiener filter as in
   `AWILoss`).
2. Preprocess $w$ to a non-negative density $\hat w(\tau)$ with one of
   `positive ∈ {"square", "abs", "linear", "exp"}`.
3. Measure Wasserstein distance between $\hat w$ and the Dirac delta at
   zero lag:

$$
W_2^2(\hat w, \delta_0) = \int \tau^2\,\hat w(\tau)\,\mathrm d\tau,\qquad
W_1(\hat w, \delta_0) = \int |\tau|\,\hat w(\tau)\,\mathrm d\tau.
$$

`order=2` returns the second moment (default); `order=1` returns the
first absolute moment.

## Why it works

For a pure time shift $\Delta t$ between $d_{\mathrm s}$ and
$d_{\mathrm o}$, the Wiener filter is a Dirac at lag $\Delta t$
convolved with a small low-pass kernel.  Its second moment is

$$
W_2^2(\hat w, \delta_0) \approx (\Delta t)^2 + \mathrm{baseline},
$$

so the OTMF misfit is **convex and monotone in $|\Delta t|$**, even
when L2 oscillates due to cycle skipping.  The matching-filter step
also absorbs amplitude / waveform discrepancies — OTMF is
**amplitude-invariant** under positive scaling of either input.

## API

```python
from sweep_loss import OTMFLoss, otmf_loss
OTMFLoss(dt=1e-3, epsilon=1e-4, positive="square", order=2)(syn, obs)
```

## Tests

`tests/test_otmf.py` checks:

* small but non-zero baseline for identical inputs (low-pass-kernel
  width from the Tikhonov stabiliser),
* amplitude invariance under positive rescaling of either input,
* monotone growth with shift,
* **monotone across an L2 cycle-skipping range**,
* `order=2` baseline-subtracted growth matches $(\Delta t)^2$ to ~15%,
* `order=1` monotone growth,
* gradients flow,
* parameter validation.

## References

* Sun, B. & Alkhalifah, T. (2019). *Adaptive traveltime inversion.*
  **Geophysics** 84 (4), U13-U29.
  doi:[10.1190/geo2018-0595.1](https://doi.org/10.1190/geo2018-0595.1)
* Sun, B. & Alkhalifah, T. (2019). *The application of an optimal
  transport to a preconditioned data matching function for robust
  waveform inversion.* **Geophysics** 84 (6), R923-R945.
  doi:[10.1190/geo2018-0413.1](https://doi.org/10.1190/geo2018-0413.1)
* Sun, B. & Alkhalifah, T. (2019). *Stereo optimal transport of the
  matching filter.* SEG Tech. Progr. Expanded Abstracts, pp. 1505-1509.
  doi:[10.1190/segam2019-3199662.1](https://doi.org/10.1190/segam2019-3199662.1)
