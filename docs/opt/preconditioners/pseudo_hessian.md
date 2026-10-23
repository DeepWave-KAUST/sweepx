# Pseudo-Hessian preconditioner

`sweepx.opt.precond.PseudoHessianPrecond` is the diagonal preconditioner from
Plessix & Mulder (2004) - the de-facto industrial preconditioner for FWI.
It approximates the diagonal of the Hessian by the product of source and
receiver illumination at each model grid point, then divides the gradient by
that diagonal.

## What it does

Given source illumination $S(x)$ and receiver illumination $R(x)$ on the
model grid (sweep accumulates both during `backward()`, see below), the
preconditioner forms the diagonal

$$
P_{ii} = \frac{1}{S_i R_i + \epsilon}
$$

and applies it to the raw gradient: $\tilde{g} = P\, g$. The construction
approximately compensates for **geometrical spreading** and **uneven
acquisition coverage** - both of which otherwise bias FWI updates toward
the shallow / dense parts of the model.

If `normalize=True` (the default), the diagonal is rescaled so its maximum
is 1, keeping the optimizer's `lr` interpretation comparable to the
unpreconditioned run.

`eps` is added to the product $S_i R_i$ as it is, so it only means something
relative to the scale of the maps: pass maps normalized to a peak of 1 (as
below) and `eps` is a fraction of that peak. At the starting model of the
[setup](../getting_started/quickstart.md#setup-used-on-these-pages) the raw source map
peaks near $3 \times 10^{13}$, so any `eps` up to $10^{-3}$ is negligible
there and the diagonal spans more than six orders of magnitude. Normalized,
the product is still far below 1 almost everywhere: a few cells hold the
peak (0.4), 99 % of the cells are below $10^{-4}$ and the median is about
$5 \times 10^{-6}$. `eps=1e-3` then leaves the diagonal at 0.92 or more on
99 % of the grid, while `eps=1e-5`, used in the examples, gives a median of
about 0.7 and a 10th percentile of about 0.3.

## API

```python
sweepx.opt.precond.PseudoHessianPrecond(
    eps=1e-3,           # stabilizer added inside the inverse
    normalize=True,     # rescale so max(diag) = 1
)
```

The standard preconditioner contract:

```python
class PseudoHessianPrecond:
    def update(self, params, *, source_illum=None, receiver_illum=None) -> None:
        """Refresh diagonal from new illumination tensors."""

    def __call__(self, grad, param) -> torch.Tensor:
        """Return P @ grad."""
```

## Hooking it into a sweep run

sweep computes the illumination maps on its native backend (`impl='c'`, the
default on a GPU) when `solver.compute_illumination = True` (off by default,
because it adds work to the backward pass). `backward()` then fills
`solver.source_illumination` and `solver.receiver_illumination`, shaped like
the model; after a forward without gradient they are `None`, and between a
forward and its `backward()` they are zeros. The eager backend never fills
them: setting `compute_illumination` on it has no effect.

Refresh the preconditioner in the closure, right after `backward()`: every
`step()` calls the closure at the current model first, so the direction is
preconditioned with the maps of the model it starts from.

```python
import sweepx.opt

precond = sweepx.opt.precond.PseudoHessianPrecond(eps=1e-5)
solver.compute_illumination = True

opt = sweepx.opt.optim.SteepestDescent(
    [inv_vp], line_search="strong_wolfe", bounds=(1500.0, 5500.0),
    preconditioner=precond, initial_step=20.0,
)

def closure():
    opt.zero_grad()
    syn = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    S, R = solver.source_illumination, solver.receiver_illumination
    precond.update([inv_vp], source_illum=S / S.max(), receiver_illum=R / R.max())
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

With several parameters (e.g. `vp`, `vs`, `rho` in elastic FWI), pass one map
per parameter as dicts keyed by `id(param)`:
`precond.update([vp, vs, rho], source_illum={id(vp): s_vp, id(vs): s_vs, id(rho): s_rho}, receiver_illum={...})`.

## When (not) to use

The pseudo-Hessian needs no extra propagation (sweep accumulates the maps in
the backward it runs anyway, at the cost of extra work in that backward) and
is the standard FWI preconditioner. Skip it if your forward solver does not
expose illumination (sweep's eager backend, e.g. for the Newton-type
optimizers), or if you are debugging an algorithm and want to see its raw
behaviour.

## References

- Plessix, R.-E., and Mulder, W. A. (2004). "Frequency-domain
  finite-difference amplitude-preserving migration." _Geophysical Journal
  International_ 157(3), 975-987.
  [DOI:10.1111/j.1365-246X.2004.02289.x](https://doi.org/10.1111/j.1365-246X.2004.02289.x).
