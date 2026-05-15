# Data convention

Every `fwiloss` misfit consumes tensors in the canonical layout

```
(nshots, nt, nreceivers, nchannel)
```

* `nshots`     — number of independent source experiments in the mini-batch.
* `nt`         — number of time samples.  Time is **axis -3**.
* `nreceivers` — number of receivers per shot.
* `nchannel`   — number of recorded components (1 for pressure / 2-3 for
  elastic / 4-9 for tensorial fields, …).

For convenience the base class also accepts:

| Input shape          | Treated as                          |
|----------------------|-------------------------------------|
| `(nt,)`              | one trace, single shot, single chan |
| `(nt, nrec)`         | one shot, single channel            |
| `(nshots, nt, nrec)` | single channel                      |
| `(nshots, nt, nrec, nchan)` | canonical (no change)         |

so `L2Loss()(syn, obs)` works regardless of which of the above the user
prefers.  All per-trace operations (envelope, CDF transport, Wiener filters,
&c.) are implemented after flattening the canonical tensor to
`(nshots*nrec*nchan, nt)` so they vectorize across receivers and components.

## Time step `dt`

Some misfits (NIM, CDF-based OT, instantaneous traveltime, …) depend on the
sampling interval $\Delta t$.  Those losses accept an explicit `dt` kwarg —
see the per-loss page for the convention.

## Optional masks

`BaseFWILoss` accepts a `mask` argument with the same shape as the inputs
(or broadcastable).  Zero entries are excluded from the reduction; this
covers muted shots, dead receivers, time gating, etc.

```python
from fwiloss import L2Loss
loss = L2Loss(mask=mask)(syn, obs)
```
