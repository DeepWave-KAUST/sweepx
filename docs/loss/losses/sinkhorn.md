# Sinkhorn (entropic OT) misfit

## Definition

For two discrete probability measures $p, q$ on the time grid and cost
matrix $C_{ij}=(t_i-t_j)^2$, the entropy-regularised OT problem
(Cuturi 2013) is

$$
\mathrm{OT}_\varepsilon(p, q) = \min_{\Pi\in\Pi(p,q)} \sum_{ij} C_{ij}\Pi_{ij} + \varepsilon\sum_{ij}\Pi_{ij}(\log\Pi_{ij}-1),
$$

solved by Sinkhorn iterations.  We implement the log-domain stabilised
form of Schmitzer (2019).  Default output (`debiased=True`) is the
**Sinkhorn divergence** of Feydy et al. (2019)

$$
S_\varepsilon(p, q) = \mathrm{OT}_\varepsilon(p, q) - \tfrac12 \mathrm{OT}_\varepsilon(p,p) - \tfrac12 \mathrm{OT}_\varepsilon(q,q),
$$

which is **non-negative, zero iff $p=q$**, and interpolates between the
maximum-mean-discrepancy ($\varepsilon\to\infty$) and the exact
$W_2^2$ ($\varepsilon\to 0$) as the regularisation goes to zero.

## When to use

* You want $W_2^2$-like behaviour but on a CPU-friendly budget.
  Sinkhorn is $O(N\cdot n_t^2 \cdot$`n_iter`$)$ per evaluation.
* You can pick the regularisation $\varepsilon$ to match the scale of
  the problem (typically $\varepsilon \sim (k \cdot dt)^2$ with $k$ a
  small number of samples).

## API

```python
from fwiloss import SinkhornLoss, sinkhorn_loss
SinkhornLoss(epsilon=(10*dt)**2, n_iter=64, positive="square",
             dt=dt, debiased=True)(syn, obs)
```

## Tests

`tests/test_sinkhorn.py` checks:

* debiased Sinkhorn → 0 for identical inputs,
* debiased / biased differ for distinct inputs (sanity),
* debiased ≥ 0 for distinct signals,
* monotone growth with shift,
* convergence to $W_2^2$ as $\varepsilon\to 0$ (within 25%),
* **monotone across L2 cycle-skipping range**,
* gradients flow,
* parameter validation.

## References

* Cuturi, M. (2013). *Sinkhorn distances: lightspeed computation of
  optimal transport.* **NeurIPS** 26.
  arXiv:[1306.0895](https://arxiv.org/abs/1306.0895)
* Feydy, J., Séjourné, T., Vialard, F.-X., Amari, S., Trouvé, A. &
  Peyré, G. (2019). *Interpolating between optimal transport and MMD
  using Sinkhorn divergences.* **AISTATS** 22, 2681-2690.
  arXiv:[1810.08278](https://arxiv.org/abs/1810.08278)
* Schmitzer, B. (2019). *Stabilised sparse scaling algorithms for entropy
  regularised transport problems.* **SIAM J. Sci. Comput.** 41 (3),
  A1443-A1481.
  doi:[10.1137/16M1106018](https://doi.org/10.1137/16M1106018)
* Sun, B. & Alkhalifah, T. (2020). *ML-misfit: a neural network-based
  misfit function for full-waveform inversion.*
  arXiv:[2002.03163](https://arxiv.org/abs/2002.03163)
