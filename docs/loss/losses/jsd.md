# Jensen–Shannon divergence misfit (Yan et al. 2024)

## Definition

Map each trace to a probability density $p$ / $q$ via a positive
transform $\sigma$ (`"square"` / `"abs"` / `"linear"` / `"exp"`) and
the standard sum-to-one normalisation.  With $m = \tfrac12(p+q)$,

$$
\mathrm{JSD}(p, q) = \tfrac12\,\mathrm{KL}(p\,\|\,m) + \tfrac12\,\mathrm{KL}(q\,\|\,m),
$$

where $\mathrm{KL}(p\|m) = \sum_t p(t) \log\!\bigl(p(t)/m(t)\bigr)$.

## Properties

* **Symmetric**: $\mathrm{JSD}(p, q) = \mathrm{JSD}(q, p)$ — unlike plain KL.
* **Bounded**: $0 \le \mathrm{JSD}(p, q) \le \log 2$.
* **Saturates** at $\log 2$ for disjoint $p$ and $q$ — this is the *opposite*
  of the OT-style anti-cycle-skipping property: JSD is informative for
  signals that overlap, useless for signals that do not.
* Set `normalize_by_log2=True` to rescale the loss to $[0, 1]$.

## When to use

Multi-parameter / multi-physics inversions where the kinematics is good
and you want a probability-theoretic comparison of the energy
distribution.  Yan et al. (2024) recommend it for shallow-seismic
multiparameter FWI with surface-wave-dominated gathers.

## API

```python
from sweep_loss import JensenShannonLoss, jensen_shannon_loss
JensenShannonLoss(positive="square")(syn, obs)
```

## Tests

`tests/test_jsd.py` checks:

* zero for identical signals across positive transforms,
* symmetry $\mathrm{JSD}(p, q) = \mathrm{JSD}(q, p)$,
* upper bound $\le \log 2$ per trace,
* `normalize_by_log2=True` scales to $[0, 1]$,
* saturation at $\log 2$ for disjoint signals,
* monotone growth with shift,
* gradients flow,
* parameter validation.

## References

* Yan, Z., Mostefai, F., Ouattara, K., et al. (2024). *Multiparameter
  shallow-seismic waveform inversion based on the Jensen-Shannon
  divergence.* **Geophys. J. Int.** 238 (1), 132-148.
  doi:[10.1093/gji/ggae131](https://doi.org/10.1093/gji/ggae131)
* Endres, D. M. & Schindelin, J. E. (2003). *A new metric for
  probability distributions.* **IEEE Trans. Inf. Theory** 49 (7),
  1858-1860.
  doi:[10.1109/TIT.2003.813506](https://doi.org/10.1109/TIT.2003.813506)
* Lin, J. (1991). *Divergence measures based on the Shannon entropy.*
  **IEEE Trans. Inf. Theory** 37 (1), 145-151.
  doi:[10.1109/18.61115](https://doi.org/10.1109/18.61115)
