# Pseudo-Hessian preconditioner

`sweepx.opt.precond.PseudoHessianPrecond` is the diagonal preconditioner from
Plessix & Mulder (2004) - the de-facto industrial preconditioner for FWI.
It approximates the diagonal of the Hessian by the product of source and
receiver illumination at each model grid point, then divides the gradient by
that diagonal.

## What it does

Given source illumination $S(x)$ and receiver illumination $R(x)$ on the
model grid (both produced by the propagator after the forward / adjoint
pass), the preconditioner forms the diagonal

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

    def __call__(self, grad, param) -> Tensor:
        """Return P @ grad."""
```

## Hooking it into a sweep run

The `sweep` propagator exposes the source / receiver illumination on the
solver after each forward pass. Plumb them into the preconditioner before
each optimizer step:

```python
import sweepx.opt

precond = sweepx.opt.precond.PseudoHessianPrecond(eps=1e-3)
opt = sweepx.opt.optim.SteepestDescent([inv_vp], lr=1.0, preconditioner=precond)

for _ in range(cfg["epochs"]):
    opt.zero_grad()
    syn = solver(wavelet, sources, receivers, models=[inv_vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()

    precond.update(
        [inv_vp],
        source_illum=solver.source_illumination.detach(),
        receiver_illum=solver.receiver_illumination.detach(),
    )
    opt.step()
```

For multi-parameter elastic FWI (`vp`, `vs`, `rho`), pass dicts keyed by
`id(param)`:

```python
precond.update(
    [vp, vs, rho],
    source_illum={id(vp): s_vp, id(vs): s_vs, id(rho): s_rho},
    receiver_illum={id(vp): r_vp, id(vs): r_vs, id(rho): r_rho},
)
```

## When (not) to use

The pseudo-Hessian is essentially free to compute (the propagator already has
the wavefields) and almost always helps. Skip it only if your forward solver
does not expose illumination, or if you are debugging an algorithm and want
to see its raw behaviour.

## References

- Plessix, R.-E., and Mulder, W. A. (2004). "Frequency-domain
  finite-difference amplitude-preserving migration." _Geophysical Journal
  International_ 157(3), 975-987.
  [DOI:10.1111/j.1365-246X.2004.02289.x](https://doi.org/10.1111/j.1365-246X.2004.02289.x).
