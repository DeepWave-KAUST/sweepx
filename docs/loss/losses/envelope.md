# Envelope misfits

Let $E(t)=|a(t)|=\sqrt{d(t)^2+\mathcal H[d](t)^2}$ be the instantaneous
envelope (modulus of the analytic signal).  `sweep_loss` provides four
envelope-based variants, all triggered through a single class
`EnvelopeLoss` with boolean / integer switches.

## Definitions

* **p = 2** (default), Wu, Luo & Wu (2014):
  $\mathcal J_E^{(2)} = \tfrac12 \sum (E_{\mathrm s}-E_{\mathrm o})^2$.
* **p = 1**:
  $\mathcal J_E^{(1)} = \sum |E_{\mathrm s}-E_{\mathrm o}|$.
* **log** (Bozdağ, Trampert & Tromp 2011, eq. 14):
  $\mathcal J_E^{\log} = \tfrac12 \sum (\log E_{\mathrm s} - \log E_{\mathrm o})^2$.
* **squared** (Chi, Dong & Liu 2014):
  $\mathcal J_{E^2} = \tfrac12 \sum (E_{\mathrm s}^2 - E_{\mathrm o}^2)^2$.

All four operate sample-wise after Hilbert-transforming along the time
axis.

## Why it helps FWI

The envelope is *non-oscillatory* — it carries the low-frequency
"modulation" of the signal even when the carrier waveform itself is
heavily cycle-skipped.  Inverting the envelope first gives a kinematic
update that bypasses cycle-skipping; the subsequent waveform inversion can
then converge in a much narrower basin (Wu, Luo & Wu 2014).

## Sign invariance

$E(t)=|a(t)|$ is invariant under $d \mapsto -d$, so the envelope misfit
cannot tell two polarity-flipped waveforms apart.  This is a feature for
amplitude-only inversion but a foot-gun for polarity-sensitive problems —
add a polarity-aware term (or use the instantaneous-phase misfit) if you
need it.

## API

```python
from sweep_loss import EnvelopeLoss
EnvelopeLoss(p=2)(syn, obs)                  # Wu 2014
EnvelopeLoss(log=True)(syn, obs)             # Bozdağ 2011 eq. 14
EnvelopeLoss(squared=True)(syn, obs)         # Chi 2014
```

## Tests

`tests/test_envelope.py` checks:

* `envelope()` matches `scipy.signal.hilbert` to float64 precision
  (with `eps=0`),
* perfect match → 0 for all four variants,
* numerical equality with the explicit per-sample formulas,
* polarity invariance,
* analytic-signal orthogonality property `<Re(a), Im(a)> ≈ 0`,
* gradients flow,
* invalid parameter combinations raise.

## References

* Bozdağ, E., Trampert, J. & Tromp, J. (2011). *Misfit functions for full
  waveform inversion based on instantaneous phase and envelope
  measurements.* **Geophys. J. Int.** 185 (2), 845-870.
  doi:[10.1111/j.1365-246X.2011.04970.x](https://doi.org/10.1111/j.1365-246X.2011.04970.x)
* Wu, R.-S., Luo, J. & Wu, B. (2014). *Seismic envelope inversion and
  modulation signal model.* **Geophysics** 79 (3), WA13-WA24.
  doi:[10.1190/geo2013-0294.1](https://doi.org/10.1190/geo2013-0294.1)
* Chi, B., Dong, L. & Liu, Y. (2014). *Full waveform inversion method using
  envelope objective function without low frequency data.*  **J. Appl. Geophys.** 109, 36-46.
  doi:[10.1016/j.jappgeo.2014.07.010](https://doi.org/10.1016/j.jappgeo.2014.07.010)
