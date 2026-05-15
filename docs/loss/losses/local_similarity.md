# Local-similarity (Fomel) misfit

## Definition

Fomel (2007) introduces the *local similarity* attribute: a windowed
normalised cross-correlation that varies smoothly in time:

$$
\gamma_\sigma(\tau) \;=\;
\frac{\sum_t w_\sigma(t-\tau)\,d_{\mathrm s}(t)\,d_{\mathrm o}(t)}
     {\sqrt{\sum_t w_\sigma(t-\tau)\,d_{\mathrm s}^2(t)\,\cdot\,\sum_t w_\sigma(t-\tau)\,d_{\mathrm o}^2(t)}},
$$

with $w_\sigma$ a Gaussian window of half-width $\sigma$ samples.  The
implementation evaluates the three sliding sums by 1-D convolution with
the same Gaussian kernel (replicate-padded so the result has the input
length).

The FWI misfit (Zhang, Sirgue & Zhang 2018) is

$$
\mathcal J_{\mathrm{LS}}(m) \;=\; \tfrac12 \sum_{\text{trace}}\sum_\tau w_\sigma^E(\tau)\,(1-\gamma_\sigma(\tau))^2,
$$

with an *energy weight* $w_\sigma^E(\tau) = \tfrac{(d_{\mathrm s}^2\!\ast\!w_\sigma)(\tau)\cdot(d_{\mathrm o}^2\!\ast\!w_\sigma)(\tau)}{\max_\tau\!(\cdot)}$
that down-weights silent regions of the trace (so the boundaries do not
contribute spurious (1-0)² = 1 to the misfit).

## Properties

* **Amplitude-invariant per window**: scaling either signal by any
  positive constant leaves $\gamma_\sigma(\tau)$ unchanged.  Tested
  explicitly.
* **Time-localised**: a brief flipped-polarity window contributes
  strongly to the misfit, whereas the *global* NCC averages it down to
  almost nothing.

## When to use

* Time-lapse FWI where you want to localise the residual in time
  (Zhang, Sirgue & Zhang 2018).
* As an attribute / monitoring quantity to *visualise* the misfit
  alongside the inversion.

## API

```python
from fwiloss import LocalSimilarityLoss, local_similarity_loss
LocalSimilarityLoss(sigma_samples=8.0)(syn, obs)
local_similarity_loss(syn, obs, sigma_samples=8.0)
```

## Tests

`tests/test_local_similarity.py` checks:

* zero for identical inputs (with energy-weighted boundary handling),
* invariance under per-trace amplitude scaling,
* monotone growth with small shifts,
* a brief polarity flip is *more* visible to local than to global NCC,
* gradients flow,
* parameter validation.

## References

* Fomel, S. (2007). *Local seismic attributes.* **Geophysics** 72 (3),
  A29-A33.
  doi:[10.1190/1.2437573](https://doi.org/10.1190/1.2437573)
* Zhang, P., Sirgue, L. & Zhang, R. (2018). *Local-similarity-based FWI
  for a time-lapse application.*  80th EAGE Conference & Exhibition.
  doi:[10.3997/2214-4609.201801006](https://doi.org/10.3997/2214-4609.201801006)
