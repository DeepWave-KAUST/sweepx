# Normalised Integration Method (NIM)

## Definition

NIM (Liu, Hu & Wang 2012; Donno, Chauris & Calandra 2013) maps a signed
seismic trace $d(t)$ to a probability density and then compares
**cumulative distributions** in L2:

$$
f(t) = \sigma\bigl(d(t)\bigr),\qquad
F(t) = \frac{\int_0^t f(\tau)\,\mathrm d\tau}{\int_0^T f(\tau)\,\mathrm d\tau},
$$

$$
\mathcal J_{\mathrm{NIM}}(m) \;=\; \tfrac12 \sum_{\text{trace}}\int_0^T \bigl(F_{\mathrm s}(t)-F_{\mathrm o}(t)\bigr)^2\,\mathrm dt.
$$

The positive transform $\sigma$ is selectable:

| `positive=` | $\sigma(d)$           | First proposed in |
|-------------|-----------------------|-------------------|
| `"square"`  | $d^2$                 | Liu et al. (2012) |
| `"abs"`     | $\lvert d\rvert$      | Donno et al. (2013) |
| `"linear"`  | $d + c$ ($c \ge \max\lvert d\rvert$) | Engquist & Froese (2014) |
| `"exp"`     | $\exp(d)$             | Engquist, Froese & Yang (2016) |

## Why it works

Comparing CDFs is equivalent to the **1-Wasserstein** distance on the
density (Bonneel et al. 2011) — a shift of a wavelet by $\Delta t$
contributes $\sim (\Delta t)^2$ to NIM regardless of cycle-skipping.  L2
saturates once the shift exceeds half a wavelength; NIM does not.

## API

```python
from fwiloss import NIMLoss, nim_loss
NIMLoss(positive="square", dt=1e-3)(syn, obs)
nim_loss(syn, obs, positive="linear", dt=1e-3)
```

## Tests

`tests/test_nim.py` checks:

* zero for identical signals across all four positive transforms,
* monotone growth in the shift of a Gaussian wavelet,
* equality with the explicit CDF-difference formula,
* **anti-cycle-skipping property**: NIM keeps growing past the
  half-wavelength shift where L2 starts to drop,
* gradients flow,
* parameter validation.

## References

* Liu, F., Hu, X. & Wang, J. (2012). *An optimised waveform inversion
  method based on a non-quadratic misfit function and the normalised
  integration method.* **Geophys. Prospect.** 60 (3), 386-394.
  doi:[10.1111/j.1365-2478.2011.00993.x](https://doi.org/10.1111/j.1365-2478.2011.00993.x)
* Donno, D., Chauris, H. & Calandra, H. (2013). *Estimating the background
  velocity model with the normalised integration method.*  75th EAGE
  Conference & Exhibition.
  doi:[10.3997/2214-4609.20130411](https://doi.org/10.3997/2214-4609.20130411)
* Bonneel, N., van de Panne, M., Paris, S. & Heidrich, W. (2011).
  *Displacement interpolation using Lagrangian mass transport.*
  ACM Trans. Graph. 30 (6).
  doi:[10.1145/2070781.2024192](https://doi.org/10.1145/2070781.2024192)
