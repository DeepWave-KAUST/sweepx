# Frequency- and Laplace-domain misfits

Five complementary misfits operating on the **temporal Fourier transform**
$D(\omega) = \sum_t d(t) e^{-i\omega t}$ are bundled here.  All of them
accept an inclusive frequency-index band `freq_band=(k0, k1)` so they
double as **multi-scale FWI** building blocks (Bunks et al. 1995).

## 1.  Frequency-domain L2 (Pratt-Shin-Hicks 1998)

$$
\mathcal J_\Omega = \tfrac12\sum_{\omega\in\Omega}|D_{\mathrm s}(\omega)-D_{\mathrm o}(\omega)|^2.
$$

## 2.  Phase-only frequency-domain (Bednar-Shin-Pyun 2007)

$$
\mathcal J_\phi = \tfrac12\sum_{\omega\in\Omega}|\Delta\Phi(\omega)|^2,
$$

with $\Delta\Phi = \arctan2(\sin(\Phi_{\mathrm s}-\Phi_{\mathrm o}),
\cos(\Phi_{\mathrm s}-\Phi_{\mathrm o}))$ wrapped to $(-\pi,\pi]$.
Amplitude weighting ($w(\omega)=|D_{\mathrm o}(\omega)|/\max|D_{\mathrm o}|$)
suppresses bins where the amplitude is so small that the phase is
meaningless; toggled with `amplitude_weight=True/False`.

## 3.  Amplitude-only frequency-domain (Shin-Min 2006)

$$
\mathcal J_A = \tfrac12\sum_{\omega\in\Omega}(|D_{\mathrm s}|-|D_{\mathrm o}|)^2.
$$

Because $|D(\omega)|$ is invariant under a time shift $t\to t+\tau$, this
misfit is **completely insensitive** to traveltime errors — useful when
you want to invert the amplitude separately from kinematics.

## 4.  Shin–Min log misfit (Shin & Min 2006)

$$
\mathcal J_{\log} = \tfrac12\sum_{\omega\in\Omega}|\log D_{\mathrm s}(\omega)-\log D_{\mathrm o}(\omega)|^2.
$$

Using $\log D = \log|D|+i\,\arg D$, this decomposes cleanly as

$$
\mathcal J_{\log} = \underbrace{\tfrac12\sum(\log|D_{\mathrm s}|-\log|D_{\mathrm o}|)^2}_{\text{log-amplitude misfit}}
\;+\; \underbrace{\tfrac12\sum|\Delta\Phi|^2}_{\text{phase misfit}}.
$$

We wrap the phase difference to $(-\pi,\pi]$ to avoid the unwrapping
pathologies of the naive complex logarithm (Choi & Alkhalifah 2013).

## 5.  Laplace-domain L2 (Shin & Cha 2008)

The data are first damped by $e^{-st}$ ($s>0$ a real damping rate) and
then compared in time-domain L2.  Equivalent to a time-domain L2 on
$\hat d(t) = e^{-st} d(t)$.  Strongly weighting early arrivals enables
*macro-velocity* recovery from cycle-skipped data (Shin & Cha 2008).

The Laplace–Fourier variant (Shin & Cha 2009) uses complex damping
$s=\sigma+i\omega$; in this library that is recovered by combining
`LaplaceL2Loss(s=σ, dt=dt)` with `FrequencyDomainL2Loss(freq_band=...)`
on the damped data.

## API

```python
from sweep_loss import (
    FrequencyDomainL2Loss, FrequencyPhaseLoss, FrequencyAmplitudeLoss,
    LogarithmicShinMinLoss, LaplaceL2Loss,
)

FrequencyDomainL2Loss(freq_band=(5, 30))(syn, obs)
FrequencyPhaseLoss(amplitude_weight=True)(syn, obs)
FrequencyAmplitudeLoss()(syn, obs)
LogarithmicShinMinLoss()(syn, obs)
LaplaceL2Loss(s=2.0, dt=1e-3)(syn, obs)
```

## Tests

`tests/test_frequency.py` checks:

* `FrequencyDomainL2Loss` matches the explicit `0.5 * sum |rfft(s)-rfft(o)|^2`,
* band restriction $\Omega \subsetneq$ all-frequencies gives a strictly
  smaller misfit,
* all four loss classes give 0 for identical inputs,
* `LogarithmicShinMinLoss` equals `(log_amp)^2 + (wrap dphi)^2` numerically,
* `FrequencyAmplitudeLoss` is essentially invariant to a small time shift,
* phase-only misfit grows monotonically with shift magnitude,
* `LaplaceL2Loss` reproduces the explicit damped-L2 formula,
* gradients flow through all five misfits,
* invalid arguments raise.

## References

* Pratt, R. G., Shin, C. & Hicks, G. J. (1998). *Gauss-Newton and full
  Newton methods in frequency-space seismic waveform inversion.*
  **Geophys. J. Int.** 133 (2), 341-362.
  doi:[10.1046/j.1365-246X.1998.00498.x](https://doi.org/10.1046/j.1365-246X.1998.00498.x)
* Shin, C. & Min, D.-J. (2006). *Waveform inversion using a logarithmic
  wavefield.* **Geophysics** 71 (3), R31-R42.
  doi:[10.1190/1.2194523](https://doi.org/10.1190/1.2194523)
* Bednar, J. B., Shin, C. & Pyun, S. (2007). *Comparison of waveform
  inversion, part 2: phase approach.* **Geophys. Prospect.** 55 (4),
  465-475.
  doi:[10.1111/j.1365-2478.2007.00618.x](https://doi.org/10.1111/j.1365-2478.2007.00618.x)
* Shin, C. & Cha, Y. H. (2008). *Waveform inversion in the Laplace
  domain.* **Geophys. J. Int.** 173 (3), 922-931.
  doi:[10.1111/j.1365-246X.2008.03768.x](https://doi.org/10.1111/j.1365-246X.2008.03768.x)
* Shin, C. & Cha, Y. H. (2009). *Waveform inversion in the
  Laplace-Fourier domain.* **Geophys. J. Int.** 177 (3), 1067-1079.
  doi:[10.1111/j.1365-246X.2009.04102.x](https://doi.org/10.1111/j.1365-246X.2009.04102.x)
* Bunks, C., Saleck, F. M., Zaleski, S. & Chavent, G. (1995).
  *Multiscale seismic waveform inversion.* **Geophysics** 60 (5),
  1457-1473.
  doi:[10.1190/1.1443880](https://doi.org/10.1190/1.1443880)
* Choi, Y. & Alkhalifah, T. (2013). *Frequency-domain waveform inversion
  using the phase derivative.* **Geophys. J. Int.** 195 (3), 1904-1916.
  doi:[10.1093/gji/ggt351](https://doi.org/10.1093/gji/ggt351)
