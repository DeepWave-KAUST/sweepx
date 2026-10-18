# Adaptive Waveform Inversion (AWI)

## Definition

For each trace, fit a **Wiener filter** $w(\tau)$ that best convolves
the synthetic data into the observed data:

$$
w \;=\; \arg\min_{w}\;\bigl\|\,d_{\mathrm s}\ast w - d_{\mathrm o}\,\bigr\|_2^{\,2}
        + \epsilon\,\|w\|_2^{\,2}.
$$

Then the AWI misfit is

$$
\mathcal J_{\mathrm{AWI}}(m) \;=\;
\tfrac{1}{2}\sum_{\text{trace}}
\frac{\sum_\tau \bigl(T(\tau)\,w(\tau)\bigr)^2}{\sum_\tau w(\tau)^2},
\qquad T(\tau) = \tau\,\Delta t,
$$

i.e. the *time-weighted spread* of the Wiener filter, normalised so the
loss is invariant under positive scaling of either input.  A perfect
model collapses $w$ onto a zero-lag delta and yields zero misfit (up to
the inevitable broadening from the Tikhonov stabiliser).

## Numerical recipe

The Wiener filter is solved analytically in the frequency domain
(Warner & Guasch 2016, eq. 12):

$$
W(\omega) \;=\; \frac{\overline{D_{\mathrm s}(\omega)}\,D_{\mathrm o}(\omega)}
                    {|D_{\mathrm s}(\omega)|^2 + \epsilon\,\max_\omega |D_{\mathrm s}|^2}.
$$

The regulariser $\epsilon$ is dimensionless and scaled by the peak
spectral amplitude so out-of-band bins (where $|D_{\mathrm s}|^2 \approx 0$)
are correctly damped.  After `irfft`, the filter is centred via
`torch.fft.fftshift` and the temporal penalty is applied with
$T(\tau) = \tau\cdot dt$ in seconds.

## Why it works (anti-cycle-skipping)

For a pure time shift $\Delta t$, the Wiener filter is a delta at lag
$\Delta t$, so

$$
\mathcal J_{\mathrm{AWI}} \;=\; \tfrac{1}{2}(\Delta t)^2
$$

which is **quadratic and monotone in the shift**.  By contrast, plain L2
on the same setup is non-monotone in $\Delta t$ (oscillates with the
waveform period) — this is the cycle-skipping pathology that AWI was
designed to circumvent.  The behaviour is exercised by
`test_awi_monotone_across_l2_cycle_skipping`.

## API

```python
from sweep_loss import AWILoss
AWILoss(dt=1e-3, epsilon=1e-4)(syn, obs)
```

The default `epsilon=1e-4` is the Warner-Guasch (2016) default.  For very
clean synthetic data, use a smaller `epsilon` (down to ~1e-10) for a
sharper Wiener filter.

## Tests

`tests/test_awi.py` checks:

* invariance under positive amplitude scaling of either input,
* monotone growth with small time shifts,
* **AWI is monotone over a shift range where L2 cycle-skips**,
* gradients flow through the Wiener-filter division,
* parameter validation,
* `module == functional` alias.

## References

* Warner, M. & Guasch, L. (2014). *Adaptive waveform inversion: theory.*
  SEG Tech. Progr. Expanded Abstracts, pp. 1089-1093.
  doi:[10.1190/segam2014-0371.1](https://doi.org/10.1190/segam2014-0371.1)
* Warner, M. & Guasch, L. (2016). *Adaptive waveform inversion: theory.*
  **Geophysics** 81 (6), R429-R445.
  doi:[10.1190/geo2015-0387.1](https://doi.org/10.1190/geo2015-0387.1)
* Guasch, L., Warner, M. & Ravaut, C. (2019). *Adaptive waveform
  inversion: practice.* **Geophysics** 84 (3), R447-R461.
  doi:[10.1190/geo2018-0377.1](https://doi.org/10.1190/geo2018-0377.1)
