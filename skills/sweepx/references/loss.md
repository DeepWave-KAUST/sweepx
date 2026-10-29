# Misfits

Two different things:

1. **Task YAML `loss.kind`** — a fixed set the runner implements.
2. **`sweepx.loss`** — a library of ~35 `torch.nn.Module` misfits for your own
   Python loop. Only the kinds in table 1 are reachable from a YAML.

## Contents

- [Task YAML loss kinds](#task-yaml-loss-kinds)
- [sweepx.loss library](#sweepxloss-library)
- [Conventions and traps](#conventions-and-traps)

## Task YAML loss kinds

| `loss.kind` | use when | options read |
|---|---|---|
| `mse` (default) | synthetic data, exact wavelet/amplitudes | — (sum of `(syn-obs)^2`, no 1/2) |
| `l1` | outliers, spiky noise | — |
| `huber` | mixed noise | `huber_delta` |
| `trace_cosine` | field data: unknown source scale / per-trace amplitude; `1 - cos` per trace | `trace_cosine_demean`, `trace_cosine_eps` |
| `envelope` | large traveltime errors (cycle skipping) | — |
| `ot` | W1 on per-trace energy CDF; large shifts, smooth updates | — |
| `cc_traveltime` | cross-correlation traveltime (soft-argmax shift), tomographic updates of a smooth background | `cc_max_lag_samples` (0 = nt//4), `cc_beta` |
| `steady_gcn` | frequency selection only (set automatically there; refused elsewhere) | — |

Options read by every kind: `time_mute_samples` / `time_mute_late_samples`
(zero samples before/after an index in both syn and obs), `data_mask_path`
(`.npy` broadcastable to the obs; post-hoc weight unless
`data_mask_window_mode: true`), `diving_window_db` + `diving_*` (per-batch
diving-wave window).

- An option set for another kind (e.g. `huber_delta` with `mse`) is not
  refused and does nothing.
- `trace_cosine` + `data_mask_path` needs `data_mask_window_mode: true` to
  window; otherwise the per-trace value is only re-weighted.
- RTM ignores `loss.kind`; it uses `imaging.loss_kind` (`trace_cosine`
  default, `mse`, `l1`) and `imaging.trace_cosine_demean`.
- `receiver_smoothing:` (FWI) applies the same lateral Gaussian to obs and
  syn (`target: both`), or obs only (`target: obs`).

## sweepx.loss library

Every class takes `reduction` (`mean` | `sum` | `none`) and `mask` (broadcastable;
zeros are excluded) plus the arguments below; each has a functional twin
(`L2Loss` ↔ `l2_loss`, ...). Inputs: `(syn, obs)`.

| family | class (key args) | when |
|---|---|---|
| Lp / robust | `L2Loss(half=True)`, `L1Loss`, `HuberLoss(delta)`, `PseudoHuberLoss(delta)`, `HybridL1L2Loss(delta)`, `CauchyLoss(c)`, `TukeyLoss(c)`, `GemanMcClureLoss(c)`, `StudentTLoss(nu, sigma)` | good start model; robust ones for outliers / bursts |
| amplitude-normalised | `GlobalCorrelationLoss(offset_one, demean)`, `TraceNormalizedL2Loss(demean)`, `LocalSimilarityLoss(sigma_samples)` | unknown source scale / coupling (field data) |
| envelope / phase | `EnvelopeLoss(p, log, squared)`, `InstantaneousPhaseLoss(envelope_weight)`, `EnvelopePhaseLoss(alpha)`, `ExponentiatedPhaseLoss`, `TimeFrequencyPhaseLoss(alpha, n_fft, hop_length)` | cycle skipping; envelope recovers low wavenumbers |
| frequency / Laplace | `FrequencyDomainL2Loss(freq_band)`, `FrequencyPhaseLoss(freq_band)`, `FrequencyAmplitudeLoss(freq_band)`, `LogarithmicShinMinLoss(freq_band)`, `LaplaceL2Loss(s, dt)` | selected frequency bins (`freq_band=(k0, k1)`, rFFT bin indices); damped early arrivals |
| traveltime / warping | `CrossCorrelationTraveltimeLoss(dt, power, sigma)`, `SoftDTWLoss(gamma)` | kinematic misfit, large shifts |
| matching filter | `AWILoss(dt, epsilon)`, `DeconvolutionLoss(dt, epsilon, normalize)`, `OTMFLoss(dt, epsilon, order)` | far-from-truth start, wavelet-tolerant |
| integration / OT | `NIMLoss(positive, dt)`, `Wasserstein1Loss(positive, dt)`, `Wasserstein2Loss(positive, dt, n_quantiles)`, `SinkhornLoss(epsilon, n_iter)`, `GSOTLoss(eta, dt, max_shift_samples)`, `JensenShannonLoss(positive)` | convex in shifts; need a positive transform of the signed trace: `positive=` `square`, `abs`, `linear` (x + c), `exp` |

```python
import torch
from sweepx.loss import L2Loss, EnvelopeLoss, GlobalCorrelationLoss, Wasserstein1Loss

ns, nt, nr = 2, 300, 16
t = torch.arange(nt) * 0.004
def ricker(t0, f=8.0):
    a = (torch.pi * f * (t - t0)) ** 2
    return (1 - 2 * a) * torch.exp(-a)
obs = torch.stack([ricker(0.4)] * nr, dim=-1).expand(ns, nt, nr).unsqueeze(-1).contiguous()  # (ns, nt, nr, 1)
syn = (2.0 * torch.stack([ricker(0.5)] * nr, dim=-1)).expand(ns, nt, nr).unsqueeze(-1).clone().requires_grad_(True)

for name, fn in [("l2", L2Loss()), ("envelope", EnvelopeLoss()),
                 ("ncc", GlobalCorrelationLoss(demean=True)),   # blind to the factor 2
                 ("w1", Wasserstein1Loss(positive="square", dt=0.004))]:
    val = fn(syn, obs)
    (g,) = torch.autograd.grad(val, syn)
    print(f"{name:8s} {val.item():.4g}  grad-norm {g.norm().item():.3g}")

mask = torch.ones(ns, nt, nr, 1); mask[:, :50] = 0          # mute the first 50 samples
print(L2Loss(mask=mask)(syn, obs).item())
```

## Conventions and traps

- Canonical layout `(nshots, nt, nrec, nchan)`, time on axis -3. Also accepted:
  `(nt,)`, `(nt, nrec)`, `(nshots, nt, nrec)`. A gather from `PlanReader` /
  SEG-Y is `(n_traces, n_samples)` — passing it as 2-D silently treats traces
  as time. Transpose first.
- `L2Loss` defaults to `half=True` (`0.5 * sum`); the YAML `mse` is the
  plain sum of squares, normalised by the runner.
- Losses that integrate over time (`NIM`, `W1/W2`, `Sinkhorn`, `AWI`,
  `CrossCorrelationTraveltime`, `LaplaceL2`, ...) take `dt`; pass the real
  sample interval or step sizes scale accordingly.
- `positive=` decides how signed traces become distributions for OT/NIM/JSD
  losses; `square` is the common choice for oscillatory data.
- Pick by the start model: L2 only once the start predicts arrivals within
  half a period at the lowest band; otherwise envelope / OT / AWI families or
  a lower first band (see `workflows.md#multiscale-stages`).
