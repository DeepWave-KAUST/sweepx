# Huber and pseudo-Huber misfits

## Definition

Let $r = d_{\mathrm{syn}} - d_{\mathrm{obs}}$.  The **Huber norm** of
Huber (1964) is

$$
h_{\delta}(r) =
\begin{cases}
\tfrac{1}{2} r^2 & |r| \le \delta\\
\delta\bigl(|r| - \tfrac{1}{2}\delta\bigr) & |r| > \delta
\end{cases}
$$

so the gradient $h_{\delta}'(r)=\operatorname{clip}(r,-\delta,\delta)$ is bounded.
Guitton & Symes (2003) advocated using this norm for the FWI data misfit.

The **pseudo-Huber** smoothing of Charbonnier et al. (1997) is everywhere
$C^{\infty}$:

$$
\tilde h_{\delta}(r) = \delta^2 \Bigl(\sqrt{1 + (r/\delta)^2}-1\Bigr).
$$

It satisfies $\tilde h_\delta(r) \approx \tfrac{1}{2} r^2$ for $|r| \ll \delta$
and $\tilde h_\delta(r) \approx \delta |r| - \delta^2$ for $|r| \gg \delta$.

## When to use

* Field data with a long-tailed noise distribution.
* When you need a smooth gradient (`PseudoHuberLoss`) but still want
  outlier resistance.

`delta` plays the role of the L1/L2 transition.  A common rule of thumb is
$\delta \approx 1.345\,\sigma$ where $\sigma$ is the robust noise scale
(Huber 1981).

## API

```python
from sweep_loss import HuberLoss, PseudoHuberLoss
huber = HuberLoss(delta=0.5, reduction="mean")(syn, obs)
psh   = PseudoHuberLoss(delta=0.5, reduction="mean")(syn, obs)
```

## Tests

`tests/test_huber.py` checks:

* point-wise formula on hand-picked residuals,
* agreement with `torch.nn.functional.huber_loss`,
* recovery of L2 as $\delta\to\infty$,
* recovery of L1 (up to a constant) as $\delta\to 0$,
* analytic bound `|grad| <= delta` of the Huber norm,
* the leading-order quadratic / linear asymptotics of the pseudo-Huber.

## References

* Huber, P. J. (1964). *Robust estimation of a location parameter.*
  Ann. Math. Stat. 35 (1), 73-101.
  doi:[10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)
* Guitton, A. & Symes, W. W. (2003). *Robust inversion of seismic data using
  the Huber norm.* **Geophysics** 68 (4), 1310-1319.
  doi:[10.1190/1.1598124](https://doi.org/10.1190/1.1598124)
* Bube, K. P. & Langan, R. T. (1997). *Hybrid L1/L2 minimisation with
  applications to tomography.* **Geophysics** 62 (4), 1183-1195.
  doi:[10.1190/1.1444219](https://doi.org/10.1190/1.1444219)
* Charbonnier, P., Blanc-Feraud, L., Aubert, G. & Barlaud, M. (1997).
  *Deterministic edge-preserving regularisation in computed imaging.* IEEE
  Trans. Image Process. 6 (2), 298-311.
  doi:[10.1109/83.551699](https://doi.org/10.1109/83.551699)
