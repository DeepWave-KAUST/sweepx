# Hybrid L1/L2 misfit (Bube–Langan)

## Definition

$$
\rho_{\delta}(r) \;=\; \delta^2 \Bigl(\sqrt{1 + (r/\delta)^2}-1\Bigr),
\qquad r = d_{\mathrm{syn}} - d_{\mathrm{obs}}.
$$

Asymptotics:

* $|r|\ll\delta$ &nbsp;&nbsp; $\Rightarrow$ $\rho_\delta(r)\approx \tfrac{1}{2}r^2$
  (L2 regime).
* $|r|\gg\delta$ &nbsp;&nbsp; $\Rightarrow$ $\rho_\delta(r)\approx \delta|r|-\delta^2$
  (L1 regime).

This is numerically identical to the *pseudo-Huber* smoothing but is named
after Bube & Langan (1997) who introduced it for **seismic tomography**.  Ha,
Chung & Shin (2009) used it for elastic FWI and showed that it can recover
inversions where pure L2 fails on noisy field data.

## API

```python
from sweep_loss import HybridL1L2Loss
loss = HybridL1L2Loss(delta=0.5)(syn, obs)
```

## Tests

`tests/test_hybrid_l1l2.py` checks:

* numerical agreement with `PseudoHuberLoss` for several $\delta$,
* L2 limit for $|r|\ll\delta$,
* L1 limit for $|r|\gg\delta$,
* gradient is uniformly bounded by $\delta$,
* `module == functional` alias.

## References

* Bube, K. P. & Langan, R. T. (1997). *Hybrid L1/L2 minimisation with
  applications to tomography.* **Geophysics** 62 (4), 1183-1195.
  doi:[10.1190/1.1444219](https://doi.org/10.1190/1.1444219)
* Ha, T., Chung, W. & Shin, C. (2009). *Waveform inversion using a
  back-propagation algorithm and a Huber function norm.*
  **Geophysics** 74 (3), R15-R24.
  doi:[10.1190/1.3112572](https://doi.org/10.1190/1.3112572)
