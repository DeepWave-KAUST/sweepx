# Marmousi 2-D — sweep-tasks getting started

A walkthrough of **how the sweep-tasks pipeline works**, on a synthetic
benchmark. The Marmousi-II velocity model (281 × 1361 cells at 12.5 m) is
embedded in `sweep.datasets`, so there is nothing to download, no SEG-Y, and
no environment-specific path.

```
forward → FWI (easy start) → FWI (cold 1-D start, multiscale)
```

Every step is one YAML and one command. The reference numbers and figures
below come from actually running them, so you can check your own run against
something.

---

## Prerequisites

```bash
pip install sweep-tasks    # also pulls sweep + pydantic + torch + yaml
```

That is the whole setup. A CUDA GPU is recommended — the examples default to
`backend.impl: c` (sweep's fused CUDA kernels). On a CPU-only machine flip to
`backend.impl: eager` and expect roughly 30× the wall time.

## The velocity presets

`ModelRef.dataset` names a `sweep.datasets` entry directly in the YAML and
`preset` picks the model within it:

| preset | role |
|---|---|
| `vp_true` | true Marmousi-II vp — used to synthesise obs |
| `vp_smooth` | low-pass-smoothed true model — the easy FWI start (Step 2) |
| `vp_linear` | 1-D gradient, zero lateral structure |

Step 3 deliberately does **not** use `vp_linear`. It builds its own 1-D ramp
with `ModelRef.linear_gradient`, because the preset gets the water column
wrong (it ramps to 1797 m/s where Marmousi is a flat 1500) and caps at
3812 m/s, below the true deep section. `sweep datasets list` shows the full
catalogue.

---

## Step 1 — forward modelling

```bash
sweep-tasks run examples/synthetic/01_forward_marmousi.yaml
```

Propagates an 8 Hz Ricker through `vp_true` with 114 shots at 150 m and a
1361-channel fixed spread, recording 10 s at dt = 1 ms.

**Reference (RTX 6000 Ada, seed 0): 34 s.** `output/record.npy` is
`(114, 10000, 1361, 1)`.

Two properties of this acquisition drive the choices in Steps 2–3:

* **obs dominant frequency is 7.10 Hz**, not the source's 8 Hz — propagation
  and geometric spreading shift the peak down. Half a period is 70 ms.
* **the latest first break lands at 8.59 s** of the 10 s record. A 7 s record
  (the obvious first guess) cut the far offsets off *right after* their first
  arrival: 31.7 % of the edge shots' traces kept under 1 s of coda. At 10 s
  that figure is 0 %.

---

## Step 2 — FWI from a smoothed start

```bash
sweep-tasks run examples/synthetic/02_fwi_marmousi_single.yaml
```

`vp_smooth` keeps the right depth trend, so a **single broadband band**
converges without cycle-skipping. Read this one first: it shows the FWI task
shape, `obs.synthetic_from`, the boundary-saving adjoint and the per-epoch QC
without the multiscale machinery on top.

**Reference (RTX 6000 Ada, seed 0), 200 epochs, 29.6 min:**

| | |
|---|---|
| misfit | 1.592e-02 → 1.564e-04 (min, epoch 115) → 2.575e-04 |
| RMSE vs true | 357.3 (init) → **270.6** |
| perturbation correlation `r` | 0.700 |

![inverted vp](figures/02_vp_final.png)

![misfit](figures/02_loss.png)

The runner's obs/syn QC panel — interleaved gathers, the acquisition map, and
a receiver-averaged amplitude spectrum where obs and syn sit on top of each
other:

![obs vs syn](figures/02_shot_gather.png)

> **`batchsize` is a fraction, not a count.** It is how many shots are drawn
> at random per optimizer step, so what matters is `batchsize / nshots`. Keep
> it near ~15 %. Measured here: 114 shots with `batchsize: 8` (7 %) gives RMSE
> 297, and `batchsize: 16` (14 %) gives 271 — denser shots only pay off if the
> batch grows with them.

---

## Step 3 — FWI from a cold 1-D start, with frequency continuation

```bash
sweep-tasks run examples/synthetic/03_fwi_marmousi_multiscale.yaml
```

Same grid, geometry and wavelet as Steps 1–2. What changes is the **start**: a
1-D ramp with no lateral structure whatsoever, so nothing about the answer
leaks in through the model. A single broadband band cycle-skips from there;
the four-stage ladder (2 → 5 → 10 Hz → broadband, 50 epochs each) is what
makes it work.

**Reference (RTX 6000 Ada, seed 0), 4 × 50 epochs, 11.5 min:**

| after | RMSE vs true |
|---|---|
| init (1-D ramp 1500→4000, water pinned) | 473.7 |
| stage 1 — 0.5–2 Hz | 402.6 |
| stage 2 — 0.5–5 Hz | 347.6 |
| stage 3 — 0.5–10 Hz | 325.3 |
| stage 4 — broadband | **317.3** |

perturbation correlation `r` = 0.752; low-wavenumber RMSE (both smoothed by
σ = 8 cells) 184.9, a **42.5 % improvement** on the starting model.

![per-stage evolution](figures/03_by_stage.png)

![inverted vp](figures/03_vp_final.png)

### Why those rungs

The ladder is not a magic sequence. A stage cycle-skips where the two-way
traveltime error exceeds half a period, so the next rung must satisfy

```
f  <  1 / (2 |Δt|)
```

Measured on this setup, the 1-D start is **−160 ms** off at 2 km depth in the
flat-layered left half of the model. That forces the first rung below 3.1 Hz —
2 Hz here. One 2 Hz stage pulls the error down to about −50 ms, which clears
5 Hz (100 ms half-period), and so on up. Jumping straight from the 1-D start
to 5 Hz reproducibly wrecks the left half, while the right half — which starts
only +65 ms off — converges fine either way.

Two consequences worth internalising:

* **RMSE against the true model is a poor progress metric here.** It is
  dominated by thin-layer amplitudes FWI cannot resolve. Traveltime error and
  the perturbation correlation `r` track what the inversion is actually doing;
  RMSE rose through the stages in several configurations that were visibly
  improving the background.
* **A single 1-D gradient cannot suit both halves of Marmousi.** The left half
  wants a slower start, the right half a faster one, and `|Δt_left| +
  |Δt_right|` at 2 km is invariant (255 ms) whatever `vmax` you choose.
  Raising it only moves error from one half to the other.

---

## Checking your own run

Everything above is reproducible: `seed` is set explicitly in each YAML, the
models come from the embedded dataset, and two runs of the same file on the
same machine come out **bit-identical** (verified — matching misfit to every
digit, max model difference 0.0 m/s).

Each run directory is self-describing:

```
sweep_runs/<task_id>/
  config_resolved.yaml     every default filled in — the exact spec that ran
  run_meta.json            host, CUDA, package versions, git state
  output/initial_vp.npy    the resolved STARTING model
  output/inverted_vp.npy   the result
  output/loss.npy          misfit per epoch
  output/epochs/           per-`show_every` model snapshots
  qc/                      vp, vp_diff, shot_gather, loss_curve figures
```

`output/initial_vp.npy` matters when the start is built in memory
(`dataset:` or `linear_gradient:`): there is no input file to point at
afterwards, so the runner writes the resolved array before training begins.

**Scope of "reproducible":** bit-identical is a same-machine, same-build
claim. A different GPU model or a rebuilt CUDA extension can reorder floating
point atomics, so expect last-digit differences there. The numbers above
should still land within a fraction of a percent, and none of the conclusions
move.
