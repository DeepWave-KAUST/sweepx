# Time-Frequency phase/envelope misfit (Fichtner-Kristeková family)

## Definition

Compute a Gabor STFT $G_s(t, \omega), G_o(t, \omega)$ of the two traces
(Gaussian-windowed short-time Fourier transform).  With magnitude $A$
and phase $\phi$ such that $G = A\,e^{i\phi}$,

$$
\mathcal J_{\mathrm{TF-env}} = \tfrac12 \sum_{t, \omega} (|G_s|-|G_o|)^2,
$$

$$
\mathcal J_{\mathrm{TF-phi}} = \tfrac12 \sum_{t, \omega} w^2(t, \omega)\,\bigl|e^{i\phi_s} - e^{i\phi_o}\bigr|^2,
$$

with the amplitude weight $w(t, \omega) = |G_o(t,\omega)|/\max|G_o|$ so
noise-floor bins do not contribute.  The unit-circle residual
$|e^{i\phi_s}-e^{i\phi_o}|^2 = 2(1-\cos\Delta\phi)$ equals $(\Delta\phi)^2$
to leading order — same physics as Kristeková et al. (2009) but with a
**gradient that is well defined everywhere** (including the zero-amplitude
bins), unlike a naive `atan2(sin, cos)` formulation.

The class exposes both terms via a single $\alpha$ mixing coefficient:

$$
\mathcal J_{\mathrm{TF}}(m) = (1-\alpha)\mathcal J_{\mathrm{TF-env}} + \alpha\,\mathcal J_{\mathrm{TF-phi}}.
$$

`alpha=0` reproduces the time-frequency envelope misfit; `alpha=1` the
TF phase misfit.

## When to use

* Time-evolving wavetrains (surface waves, long codas) where the
  *spectro-temporal* phase is the meaningful kinematic quantity.
* Continental- / global-scale FWI in the spirit of Fichtner et al.
  (2008) — the original use case.

## API

```python
from sweep_loss import TimeFrequencyPhaseLoss
TimeFrequencyPhaseLoss(alpha=0.5, n_fft=128, sigma_samples=20)(syn, obs)
```

`n_fft` must not exceed the trace length.  Larger `n_fft` ⇒ finer
frequency resolution but coarser time resolution.

## Tests

`tests/test_tf_phase.py` checks:

* zero for identical signals, all $\alpha$,
* $\alpha=0$ vs $\alpha=1$ vs $\alpha=0.5$ linear decomposition,
* TF-envelope is *polarity-invariant* (sign flip → 0), while TF-phase is *polarity-sensitive*,
* monotone growth with shift,
* per-sample phase loss bounded by 2 (unit-circle residual bound),
* gradients flow (note: switching to unit-circle residual prevents the
  atan2-style gradient blow-up at zero-amplitude bins),
* parameter / shape validation.

## References

* Fichtner, A., Kennett, B. L. N., Igel, H. & Bunge, H.-P. (2008).
  *Theoretical background for continental- and global-scale full-waveform
  inversion in the time-frequency domain.* **Geophys. J. Int.** 175 (2),
  665-685.
  doi:[10.1111/j.1365-246X.2008.03923.x](https://doi.org/10.1111/j.1365-246X.2008.03923.x)
* Kristeková, M., Kristek, J. & Moczo, P. (2009). *Time-frequency misfit
  and goodness-of-fit criteria for quantitative comparison of time
  signals.* **Geophys. J. Int.** 178 (2), 813-825.
  doi:[10.1111/j.1365-246X.2009.04177.x](https://doi.org/10.1111/j.1365-246X.2009.04177.x)
* Kristeková, M., Kristek, J., Moczo, P. & Day, S. M. (2006). *Misfit
  criteria for quantitative comparison of seismograms.* **Bull. Seismol.
  Soc. Am.** 96 (5), 1836-1850.
  doi:[10.1785/0120060012](https://doi.org/10.1785/0120060012)
