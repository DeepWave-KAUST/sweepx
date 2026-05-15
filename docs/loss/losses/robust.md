# Cauchy / Tukey / Geman–McClure robust misfits

All three losses are *redescending* M-estimators: their influence function
saturates (Cauchy, Geman–McClure) or vanishes completely (Tukey) for large
residuals, providing strong outlier resistance.

## Definitions

Let $r = d_{\mathrm{syn}} - d_{\mathrm{obs}}$ and $c > 0$ the scale.

**Cauchy / Lorentzian** (Crase et al. 1990; Black & Anandan 1996):

$$
\rho^{\mathrm{C}}_c(r) = \tfrac{c^2}{2}\log\bigl(1 + (r/c)^2\bigr).
$$

**Tukey biweight** (Beaton & Tukey 1974):

$$
\rho^{\mathrm{T}}_c(r) =
\begin{cases}
\tfrac{c^2}{6}\Bigl(1-\bigl[1-(r/c)^2\bigr]^3\Bigr) & |r| \le c,\\
\tfrac{c^2}{6} & |r| > c.
\end{cases}
$$

The gradient $\rho_{\mathrm{T}}'(r)= r\,(1-(r/c)^2)^2 \mathbf 1_{|r|\le c}$ is
**identically zero** for $|r|>c$ - hence "outlier rejection".

**Geman–McClure** (Geman & McClure 1985):

$$
\rho^{\mathrm{GM}}_c(r) = c^2\,\frac{(r/c)^2}{1+(r/c)^2}.
$$

All three reduce to a quadratic for $|r|\ll c$.

## When to use

* Strong, isolated outliers (bad picks, mistraced gathers).
* Tukey is the most aggressive and works best when you can already estimate
  a reasonable noise scale `c`.

## API

```python
from fwiloss import CauchyLoss, TukeyLoss, GemanMcClureLoss
CauchyLoss(c=0.5)(syn, obs)
TukeyLoss(c=0.5)(syn, obs)
GemanMcClureLoss(c=0.5)(syn, obs)
```

## Tests

`tests/test_robust.py` checks:

* point-wise formulas on hand-picked residuals,
* Cauchy log-growth for $|r|\gg c$,
* Tukey clamps to $c^2/6$ for $|r|>c$ and has zero gradient there,
* Geman–McClure stays below $c^2$ everywhere and is quadratic near zero,
* leading-order agreement with L2 in the small-residual regime.

## References

* Beaton, A. E. & Tukey, J. W. (1974). *The fitting of power series, meaning
  polynomials, illustrated on band-spectroscopic data.* Technometrics 16, 147-185.
* Black, M. J. & Anandan, P. (1996). *The robust estimation of multiple motions.*
  Computer Vision and Image Understanding 63 (1), 75-104.
* Bube, K. P. & Nemeth, T. (2007). *Fast line searches for the robust solution of
  linear systems in the hybrid l1/l2 and Huber norms.* **Geophysics** 72 (2),
  A13-A17.
* Crase, E., Pica, A., Noble, M., McDonald, J. & Tarantola, A. (1990).  *Robust
  elastic nonlinear waveform inversion: application to real data.*
  **Geophysics** 55 (5), 527-538.
* Geman, S. & McClure, D. E. (1985).  *Bayesian image analysis: An application to
  single photon emission tomography.*  Proc. Stat. Comp. Sect., ASA, 12-18.
