# Soft-DTW misfit

## Definition

Soft-DTW (Cuturi & Blondel 2017) is a smoothed version of classical
Dynamic Time Warping that replaces the discrete min over alignment
paths $\mathcal A$ with a soft-min:

$$
\mathrm{sDTW}_\gamma(d_{\mathrm s}, d_{\mathrm o}) \;=\; -\gamma\,\log\sum_{A\in\mathcal A}\exp\!\Bigl(-\tfrac{1}{\gamma}\langle A, \Delta\rangle\Bigr),
$$

with pointwise cost $\Delta_{ij} = (d_{\mathrm s,i} - d_{\mathrm o,j})^2$.
The recursion (Cuturi & Blondel 2017, alg. 1)

$$
R_{i,j} = \Delta_{i-1, j-1} + \mathrm{softmin}_\gamma\bigl(R_{i-1, j},\;R_{i, j-1},\;R_{i-1, j-1}\bigr),
$$

evaluated in $O(n_t^2)$ time and memory, gives a $C^\infty$ misfit whose
gradients flow through PyTorch autograd.

* $\gamma \to 0^+$: classical (non-differentiable) DTW.
* $\gamma \to \infty$: heavily smoothed, approaches a soft average of
  all alignment costs.

## When to use

Use Soft-DTW when the two traces are related by an unknown, possibly
*non-stationary* time warp.  This is the setting where classical DTW
(Hale 2013; Ma & Hale 2013) and its smooth-version variants have been
used for FWI velocity-model building.

## API

```python
from fwiloss import SoftDTWLoss, soft_dtw_loss
SoftDTWLoss(gamma=0.1)(syn, obs)
soft_dtw_loss(syn, obs, gamma=0.1, normalize_by_length=True)
```

## Tests

`tests/test_soft_dtw.py` checks:

* small $\gamma$ ⇒ Soft-DTW $\approx 0$ for identical signals,
* growth with shift,
* Soft-DTW absorbs the warp so it is < L2 for shifted Rickers,
* gradient flow,
* larger $\gamma$ yields smaller per-trace loss (soft-min property),
* parameter validation.

## References

* Cuturi, M. & Blondel, M. (2017). *Soft-DTW: a differentiable loss
  function for time-series.* **ICML** 70, 894-903.
  arXiv:[1703.01541](https://arxiv.org/abs/1703.01541)
* Sava, P. (2014). *3D dynamic time warping for traveltime inversion.*
  SEG Tech. Progr. Expanded Abstracts, pp. 4830-4834.
  doi:[10.1190/segam2014-1452.1](https://doi.org/10.1190/segam2014-1452.1)
* Ma, Y. & Hale, D. (2013). *Wave-equation reflection traveltime
  inversion with dynamic warping and full-waveform inversion.*
  **Geophysics** 78 (6), R223-R233.
  doi:[10.1190/geo2013-0058.1](https://doi.org/10.1190/geo2013-0058.1)
