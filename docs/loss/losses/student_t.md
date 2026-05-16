# Student-t negative-log-likelihood misfit

## Definition

Modelling the residual $r = d_{\mathrm{syn}} - d_{\mathrm{obs}}$ as
Student's-t with $\nu$ degrees of freedom and scale $\sigma$, the
data-dependent part of the negative log-likelihood is

$$
\rho^{\nu,\sigma}(r) \;=\; \frac{\nu+1}{2}\,\log\!\Bigl(1 + \tfrac{1}{\nu}(r/\sigma)^2\Bigr).
$$

Heavy tails ⇒ the influence function

$$
\rho'(r) = \frac{(\nu+1)\,r}{\nu\sigma^2 + r^2}
$$

saturates as $|r|\to\infty$, hence strong outlier resistance.

## Limits

* $\nu = 1$: Cauchy/Lorentzian (matches `CauchyLoss(c=sigma)` up to the
  fixed factor $2/\sigma^2$).
* $\nu \to \infty$: Gaussian / L2 (matches `L2Loss(half=True)/sigma^2`).
* `full_nll=True` adds the data-independent normalisation constants of the
  log-pdf so the loss is the *complete* negative log-likelihood — useful
  if you want to compare runs at different $(\nu,\sigma)$ on an absolute
  scale.

## When to use

Field data with heavy-tailed noise and an unknown number of bad traces.
Tune $\nu$ to control how aggressively outliers are down-weighted ($\nu=1$
is most aggressive; $\nu=5$–$20$ is a common compromise; large $\nu$ degrades
to L2).

## API

```python
from sweep_loss import StudentTLoss, student_t_loss
StudentTLoss(nu=4.0, sigma=0.5)(syn, obs)
student_t_loss(syn, obs, nu=4.0, sigma=0.5, full_nll=False)
```

## Tests

`tests/test_student_t.py` checks:

* point-wise formula on hand-picked residuals,
* zero at zero,
* the $\nu=1$ ↔ Cauchy correspondence,
* the $\nu\to\infty$ ↔ L2 correspondence (float64),
* `full_nll=True` adds exactly the analytic constant,
* gradient saturates at large $|r|$,
* invalid parameters raise.

## References

* Aravkin, A. Y., van Leeuwen, T. & Herrmann, F. J. (2012). *Robust FWI
  using Student-t distribution.*  SEG Technical Program Expanded Abstracts,
  pp. 1-5.
  doi:[10.1190/segam2012-1010.1](https://doi.org/10.1190/segam2012-1010.1)
* Aravkin, A., Burke, J. V. & Friedlander, M. P. (2013). *Variational
  properties of value functions.* **SIAM J. Optim.** 23 (3), 1689-1717.
  doi:[10.1137/120899157](https://doi.org/10.1137/120899157)
