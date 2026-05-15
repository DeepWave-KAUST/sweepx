# Instantaneous-phase and envelope+phase misfits

Let the analytic signal be $a(t) = d(t) + i\,\mathcal H[d](t) = E(t)\,e^{i\phi(t)}$
with envelope $E$ and instantaneous phase $\phi$.

## Definitions

**Instantaneous-phase misfit** (Bozdağ, Trampert & Tromp 2011, eq. 22):

$$
\mathcal J_\phi(m) = \tfrac{1}{2}\sum \bigl[w_\phi(t)\,\Delta\phi(t)\bigr]^2,
$$

where $\Delta\phi = \operatorname{wrap}(\phi_{\mathrm s}-\phi_{\mathrm o})$ and
$\operatorname{wrap}(x)=\arctan2(\sin x,\cos x)\in(-\pi,\pi]$ is the
**differentiable** wrapped phase difference.  The optional envelope weight
$w_\phi(t) = E_{\mathrm o}(t)/\max_t E_{\mathrm o}$ damps samples where the
envelope is tiny (and the phase is meaningless).

**Envelope + phase combined** (Fichtner 2008; Yuan, Simons & Tromp 2016):

$$
\mathcal J_{E+\phi}(m) = (1-\alpha)\,\mathcal J_E + \alpha\,\mathcal J_\phi.
$$

With $\alpha=0$ → pure envelope misfit (Wu 2014); with $\alpha=1$ → pure
phase misfit; both endpoints reproduce the corresponding stand-alone
losses exactly (asserted in the tests).

## Properties

* **Amplitude scaling of `syn`**: phase is invariant under a positive
  scalar multiplication; the envelope-weighted variant inherits this
  invariance (up to round-off).
* **Polarity sensitive**: flipping the sign of `syn` shifts the phase by
  $\pi$, so $\mathcal J_\phi$ grows by a non-trivial amount — unlike the
  envelope misfit.
* **Bounded residual**: $|\Delta\phi| \le \pi$ ⇒ $\mathcal J_\phi \le
  \tfrac12 \pi^2 \cdot N_{\mathrm{samples}}$.

## When to use

* Mitigating cycle-skipping while still using waveform information.
* Combined envelope+phase inversion: $\alpha$ controls a "low-frequency
  first, then high-frequency" hierarchy.

## API

```python
from fwiloss import InstantaneousPhaseLoss, EnvelopePhaseLoss
InstantaneousPhaseLoss(envelope_weight=True)(syn, obs)
EnvelopePhaseLoss(alpha=0.5, envelope_log=False)(syn, obs)
```

## Tests

`tests/test_inst_phase.py` checks:

* perfect match → 0,
* amplitude scaling of `syn` leaves the envelope-weighted misfit invariant
  (round-off tolerance only),
* monotone growth with small time shifts,
* polarity sensitivity (vs. the envelope misfit, which is *not* polarity
  sensitive),
* wrapped-phase residual is always in $(-\pi,\pi]$,
* $\alpha=0$ matches the pure envelope loss,
* $\alpha=1$ matches the pure phase loss,
* linear combination at intermediate $\alpha$,
* gradients flow,
* `module == functional` aliases.

## References

* Bozdağ, E., Trampert, J. & Tromp, J. (2011). *Misfit functions for full
  waveform inversion based on instantaneous phase and envelope
  measurements.* **Geophys. J. Int.** 185 (2), 845-870.
  doi:[10.1111/j.1365-246X.2011.04970.x](https://doi.org/10.1111/j.1365-246X.2011.04970.x)
* Fichtner, A., Kennett, B. L. N., Igel, H. & Bunge, H.-P. (2008).
  *Theoretical background for continental- and global-scale full-waveform
  inversion in the time-frequency domain.*  **Geophys. J. Int.** 175 (2),
  665-685.
  doi:[10.1111/j.1365-246X.2008.03923.x](https://doi.org/10.1111/j.1365-246X.2008.03923.x)
* Yuan, Y. O., Simons, F. J. & Tromp, J. (2016). *Double-difference
  adjoint seismic tomography.* **Geophys. J. Int.** 206 (3), 1599-1618.
  doi:[10.1093/gji/ggw233](https://doi.org/10.1093/gji/ggw233)
