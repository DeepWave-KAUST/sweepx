# Nonlinear Conjugate Gradient (PR+)

`sweepx.opt.optim.NCG` is a Nonlinear Conjugate Gradient optimizer using the
**Polak-Ribiere+** update with strong-Wolfe line search. PR+ is the
practical workhorse choice for nonlinear CG: the `max(0, beta_PR)`
projection automatically restarts the iteration when the standard PR formula
goes negative, fixing both the "jamming" failure of Fletcher-Reeves and the
lack of guaranteed descent of plain Polak-Ribiere.

## Update rule

At iteration $k$, with current gradient $g_k$ and preconditioned gradient
$z_k = P g_k$:

$$
\beta_k^{\text{PR+}} = \max\!\left(0,\; \frac{z_k^\top (g_k - g_{k-1})}{g_{k-1}^\top z_{k-1}}\right),
\qquad
d_k = -z_k + \beta_k^{\text{PR+}}\, d_{k-1}
$$

with $d_0 = -z_0$. The step size $\alpha_k$ is chosen by strong-Wolfe line
search; default $c_2 = 0.1$ matches the Nocedal & Wright recommendation for
CG-type methods.

## API

```python
sweepx.opt.optim.NCG(
    params,                          # iterable of torch.Tensor
    *,
    bounds=None,                     # (lo, hi) box; clamp after each step
    preconditioner=None,             # callable (grad, param) -> grad
    line_search="strong_wolfe",      # currently the only choice
    c1=1e-4, c2=0.1,                 # Wolfe constants (CG-tuned default for c2)
    alpha_init=1.0, alpha_max=10.0,
    max_ls_iter=25,
    restart_every=None,              # optional periodic safety restart
)
```

A closure is **required** for every `step()` call (the line search needs to
evaluate $\phi(\alpha)$ and $\phi'(\alpha)$).

Diagnostic attributes:
- `opt.last_beta` - the PR+ coefficient used most recently (0.0 means restart).
- `opt.last_restart` - True iff the last step was a restart (`d = -P g`).

## Drop-in usage

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
opt = sweepx.opt.optim.NCG(
    [vp],
    bounds=(1500.0, 5500.0),
    preconditioner=sweepx.opt.precond.PseudoHessianPrecond(eps=1e-3),
)

def closure():
    opt.zero_grad()
    syn = solver(wavelet, sources, receivers, models=[vp])
    loss = (syn - obs).pow(2).sum()
    loss.backward()
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

## When to use

NCG is the natural step up from Steepest Descent: same memory cost (just
keep `g_{k-1}`, `z_{k-1}`, `d_{k-1}`), much better convergence on
ill-conditioned problems. For FWI specifically, it is competitive with
L-BFGS at small history sizes (2-3 stored vectors) and avoids L-BFGS's
storage of multiple `(s, y)` pairs.

When you have memory to spare and want better convergence, switch to
L-BFGS (Tier-1) or Truncated Newton (Tier-2). Both are added in later
commits.

## References

- Polak, E., and Ribiere, G. (1969). "Note sur la convergence de methodes de
  directions conjuguees." _Revue francaise d'informatique et de recherche
  operationnelle_ 3(R1), 35-43.
  [DOI:10.1051/m2an/196903R100351](https://doi.org/10.1051/m2an/196903R100351).
- Powell, M. J. D. (1977). "Restart procedures for the conjugate gradient
  method." _Mathematical Programming_ 12, 241-254.
  [DOI:10.1007/BF01593790](https://doi.org/10.1007/BF01593790).
- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer, Section 5.2.
  [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).
