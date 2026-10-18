# Graph-space Optimal Transport (GSOT) misfit

## Definition

Métivier et al. (2018) embed each sampled trace in the
**(time, amplitude)** plane and solve an assignment problem with the
2-D ground cost

$$
c\bigl((t_i, d_{\mathrm s,i}),(t_j, d_{\mathrm o,j})\bigr) = \eta\,(t_i-t_j)^2 + (d_{\mathrm s,i}-d_{\mathrm o,j})^2.
$$

The GSOT misfit is the minimum-cost permutation:

$$
\mathcal J_{\mathrm{GSOT}}(m) \;=\; \min_{\sigma\in S_{n_t}}\sum_i c\bigl((t_i, d_{\mathrm s,i}),(t_{\sigma(i)}, d_{\mathrm o,\sigma(i)})\bigr).
$$

The weighting $\eta = (\sigma_d/T_{\mathrm{shift,max}})^2$ balances time
vs. amplitude axes; if `eta=None` we estimate it per trace as
$(\max|d| / T_{\mathrm{shift,max}})^2$ with $T_{\mathrm{shift,max}} = n_t \cdot dt / 4$
(Métivier et al. 2018, sec. 2.3).

### Banded variant

For long traces the $O(n_t^3)$ Hungarian solver is expensive.
`max_shift_samples=k` forbids assignments with $|i-j|>k$ (Métivier et
al. 2018, sec. 3.3), which prunes the cost matrix to a 2$k$+1-wide
band and is sufficient when the maximum plausible traveltime error
$\le k\cdot dt$.

## Solver / differentiability

* Hungarian / Jonker-Volgenant via `scipy.optimize.linear_sum_assignment`
  (Jonker & Volgenant 1987).
* **Envelope theorem**: once the optimal $\sigma^\star$ is known we
  recompute the loss in PyTorch with that permutation fixed; gradients
  flow through.

## When to use

* Severely cycle-skipped data with substantial amplitude information.
* The cleanest published 1-D OT misfit for **signed** traces — no
  positive transform is needed because the amplitude axis is part of
  the ground cost.

## API

```python
from sweep_loss import GSOTLoss, gsot_loss
GSOTLoss(eta=None, dt=1e-3, max_shift_samples=None)(syn, obs)
gsot_loss(syn, obs, eta=1e3, dt=1e-3, max_shift_samples=20)
```

## Tests

`tests/test_gsot.py` checks:

* zero for identical inputs,
* monotone growth with small shifts,
* **monotone across an L2 cycle-skipping range**,
* `max_shift_samples=nt` matches the unrestricted GSOT,
* a narrow band gives a *larger* (or equal) misfit than a wide one,
* gradients flow,
* parameter validation.

## References

* Métivier, L., Allain, A., Brossier, R., Mérigot, Q., Oudet, E. &
  Virieux, J. (2018). *Optimal transport for mitigating cycle skipping
  in FWI: a graph-space transform approach.* **Geophysics** 83 (5),
  R515-R540.
  doi:[10.1190/geo2017-0807.1](https://doi.org/10.1190/geo2017-0807.1)
* Métivier, L., Brossier, R., Mérigot, Q. & Oudet, E. (2019). *A graph
  space optimal transport distance as a generalisation of Lp distances:
  application to a seismic imaging inverse problem.* **Inverse
  Problems** 35 (8), 085001.
  doi:[10.1088/1361-6420/ab206f](https://doi.org/10.1088/1361-6420/ab206f)
* Jonker, R. & Volgenant, A. (1987). *A shortest augmenting path
  algorithm for dense and sparse linear assignment problems.*
  **Computing** 38 (4), 325-340.
  doi:[10.1007/BF02278710](https://doi.org/10.1007/BF02278710)
