# Deconvolution-based misfit (Luo–Sava)

## Definition

For each trace, deconvolve the observation by the synthetic:

$$
\Psi(\tau) \;=\; \mathcal F^{-1}\!\Bigl(\tfrac{D_{\mathrm o}(\omega)}{D_{\mathrm s}(\omega)+\epsilon}\Bigr).
$$

If the model is perfect, $\Psi$ is a unit impulse at zero lag.  The
**Luo–Sava** misfit penalises departures from that with $P(\tau)=\tau^2$:

$$
\mathcal J_{\mathrm{Decon}}(m) \;=\; \tfrac{1}{2}\sum_{\text{trace}}\sum_\tau \tau^2\,\Psi(\tau)^2.
$$

The optional `normalize=True` mode divides by $\sum_\tau \Psi^2$ to make
the misfit amplitude-invariant, recovering the "AWI-style" variant of
Choi & Alkhalifah (2018).

## Relation to AWI

* **AWI** convolves $d_{\mathrm s}$ with a Wiener filter to match
  $d_{\mathrm o}$, then penalises the lag spread of that filter
  *normalised by its own L2-norm*.
* **Luo–Sava** *deconvolves* the observation by the synthetic, then
  penalises the lag spread of that deconvolution.  Without
  normalisation it is amplitude-sensitive; with normalisation it is
  amplitude-invariant and closer to AWI in spirit.

## API

```python
from sweep_loss import DeconvolutionLoss
DeconvolutionLoss(dt=1e-3, epsilon=1e-4, normalize=True)(syn, obs)
```

## Tests

`tests/test_deconvolution.py` checks:

* `normalize=True` is invariant under positive amplitude scaling,
* `normalize=False` is *not* (sanity check that the option matters),
* monotone growth with small shifts,
* **monotone across a range where L2 cycle-skips**,
* gradients flow,
* parameter validation.

## References

* Luo, S. & Sava, P. (2011). *A deconvolution-based objective function
  for wave-equation inversion.* SEG Tech. Progr. Expanded Abstracts,
  pp. 2788-2792.
  doi:[10.1190/1.3627766](https://doi.org/10.1190/1.3627766)
* Choi, Y. & Alkhalifah, T. (2018). *Time-domain full-waveform inversion
  of exponentially damped wavefield using the deconvolution-based
  objective function.* **Geophysics** 83 (2), R77-R88.
  doi:[10.1190/geo2017-0057.1](https://doi.org/10.1190/geo2017-0057.1)
* Zhu, H. & Fomel, S. (2016). *Building good starting models for FWI
  using adaptive matching filtering misfit.* SEG Tech. Progr. Expanded
  Abstracts, pp. 1421-1425.
  doi:[10.1190/segam2016-13865936.1](https://doi.org/10.1190/segam2016-13865936.1)
