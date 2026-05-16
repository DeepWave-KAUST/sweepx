# Cross-correlation travel-time misfit

## Definition

For each trace compute the cross-correlation

$$
c(\tau) \;=\; \int d_{\mathrm s}(t)\,d_{\mathrm o}(t+\tau)\,\mathrm dt
$$

and let $\tau^\star$ be the lag that maximises $c$.  The travel-time
misfit is then

$$
\mathcal J_{\mathrm{CCT}}(m) \;=\; \tfrac{1}{2}\sum_{\mathrm{trace}}(\tau^\star)^2.
$$

This is the Luo–Schuster (1991) wave-equation traveltime objective; van
Leeuwen & Mulder (2010) showed it can be written equivalently as a
*weighted* cross-correlation criterion that is fully differentiable, which
is the form we implement.

## Smooth differentiable surrogate

Argmax is not differentiable in PyTorch.  We use the
**correlation-weighted centroid** approximation

$$
\tau^\star \;\approx\; \frac{\sum_\tau \tau\,[c(\tau)]_+^{\,p} w(\tau)}{\sum_\tau [c(\tau)]_+^{\,p} w(\tau)},
$$

where $[\cdot]_+$ clamps the correlation to non-negative values, $p$
(``power``) controls the sharpness ($p\to\infty$ recovers argmax) and
$w(\tau) = \exp(-\tau^2/2\sigma^2)$ (``sigma``) optionally gates the
allowed lag range.  The cross-correlation itself is computed via FFTs in
$O(n_t \log n_t)$.

## When to use

* Severely cycle-skipped data: the kinematic information is in the
  correlation peak position, not in waveform shape.
* Building good starting models for subsequent waveform inversion.

## API

```python
from sweep_loss import CrossCorrelationTraveltimeLoss
loss = CrossCorrelationTraveltimeLoss(
    dt=1e-3, power=4.0, sigma=200.0,
)(syn, obs)
```

## Tests

`tests/test_traveltime.py` checks:

* perfect match → 0,
* recovery of a known integer-sample Ricker shift to within ~0.5 dt,
* monotone growth with shift magnitude,
* finite & non-zero gradients,
* `module == functional` alias,
* parameter validation.

## References

* Luo, Y. & Schuster, G. T. (1991). *Wave-equation travel-time inversion.*
  **Geophysics** 56 (5), 645-653.
  doi:[10.1190/1.1443081](https://doi.org/10.1190/1.1443081)
* Marquering, H., Dahlen, F. A. & Nolet, G. (1999). *Three-dimensional
  sensitivity kernels for finite-frequency traveltimes.*
  **Geophys. J. Int.** 137 (3), 805-815.
  doi:[10.1046/j.1365-246x.1999.00837.x](https://doi.org/10.1046/j.1365-246x.1999.00837.x)
* van Leeuwen, T. & Mulder, W. A. (2010). *A correlation-based misfit
  criterion for wave-equation traveltime tomography.*
  **Geophys. J. Int.** 182 (3), 1383-1394.
  doi:[10.1111/j.1365-246X.2010.04681.x](https://doi.org/10.1111/j.1365-246X.2010.04681.x)
