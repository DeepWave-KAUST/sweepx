# Exponentiated-phase misfit (Yuan et al. 2020)

## Definition

Normalise the analytic signal by its envelope:

$$
\tilde s(t) = \frac{s(t) + i\,\mathcal H[s](t)}{E_s(t)} = e^{i\phi_s(t)},\qquad
E_s(t) = \sqrt{s^2(t)+\mathcal H^2[s](t)}.
$$

The exponentiated-phase misfit (Yuan et al. 2020, eq. 7) is the L2 distance
between the two unit-modulus complex signals:

$$
\chi_{\mathrm{EP}}(m) = \tfrac{1}{2}\sum \int_0^T |\Re\tilde s - \Re\tilde d|^2 + |\Im\tilde s - \Im\tilde d|^2 \,\mathrm dt.
$$

## Why "better than instantaneous phase"

`InstantaneousPhaseLoss` (Bozdağ et al. 2011) takes
$\phi = \arctan(\mathcal H[d]/d)$ and subtracts; this has a branch cut at
$\phi = \pm\pi$ that must be wrapped (we use `atan2(sin, cos)`).  The
exponentiated phase replaces this with a smooth division and never
crosses the branch cut: it is everywhere $C^\infty$.

## Properties

* **Amplitude invariant** (exactly when `eps=0`): $\tilde{(\alpha s)} = \tilde s$ for any $\alpha > 0$.
  With `eps > 0` the invariance is broken by $O(\epsilon/E_s^2)$ — set
  `eps=0` if you can guarantee no zero-amplitude samples enter the loss.
* **Bounded**: $|\Delta\Re|^2 + |\Delta\Im|^2 \le 4$ per sample, so the
  total misfit is at most $2N_{\mathrm{samples}}$.
* **Polarity sensitive**: flipping $s\to -s$ rotates $\tilde s$ by $\pi$, so
  the loss is large (this distinguishes it from the envelope misfit).

## API

```python
from sweep_loss import ExponentiatedPhaseLoss
ExponentiatedPhaseLoss(eps=1e-8)(syn, obs)
```

## Tests

`tests/test_exponentiated_phase.py` checks:

* zero for identical signals,
* amplitude invariance with `eps=0`,
* per-sample upper bound of 2,
* monotone growth with shift,
* polarity sensitivity,
* **no jump across the $\pm\pi$ branch cut** (this is the main motivation
  for the loss — verified on a sinusoid swept through $\pi$),
* gradients flow,
* parameter validation.

## References

* Yuan, Y. O., Bozdağ, E., Ciardelli, C., Gao, F. & Simons, F. J. (2020).
  *The exponentiated phase measurement, and objective-function
  hybridisation for adjoint waveform tomography.* **Geophys. J. Int.**
  221 (2), 1145-1164.
  doi:[10.1093/gji/ggaa063](https://doi.org/10.1093/gji/ggaa063)
* Gao, F., Yuan, Y. O., Ciardelli, C., Simons, F. J., Bozdağ, E. &
  Tromp, J. (2023). *Review of misfit functions for adjoint full
  waveform inversion in seismology.* **Geophys. J. Int.** 235 (3),
  2794-2820.
  doi:[10.1093/gji/ggad372](https://doi.org/10.1093/gji/ggad372)
