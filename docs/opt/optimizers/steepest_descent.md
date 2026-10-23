# Steepest Descent

`sweepx.opt.optim.SteepestDescent` is the simplest first-order optimizer in the
library. It exists for two reasons:

1. **Baseline.** Compare every other optimizer against SD - you need to know
   what "no curvature information" looks like before you can claim a speed-up.
2. **Reusable Wolfe line-search machinery.** The strong-Wolfe routine that
   ships alongside SD is the same routine used by the Nonlinear CG and L-BFGS
   optimizers later.

## Update rule

Without line search:

$$
\theta_{k+1} = \theta_k - \alpha\, P\, g_k
$$

where $g_k = \nabla f(\theta_k)$, $P$ is an optional preconditioner (identity
by default), and $\alpha$ is the fixed `lr`. `lr` multiplies the raw
gradient, so it is in (parameter units) / (gradient units): for a velocity
model in m/s, whose gradients are tiny, it is a large number.

With `line_search="strong_wolfe"`:

$$
\alpha_k = \arg\min_{\alpha > 0}\; f\!\bigl(\theta_k + \alpha d_k\bigr)
\quad \text{subject to strong-Wolfe},\qquad d_k = -P g_k
$$

solved by Algorithms 3.5 + 3.6 of Nocedal & Wright. `initial_step=` sizes
the first trial of the first search in parameter units (it changes the
largest component by that much); later searches start from the previous
step (Nocedal & Wright eq. 3.60), and no trial changes a parameter by more
than `alpha_max / alpha_init` (10) times `initial_step`. Without it every
search starts at
`alpha_init=1`, which does not move an m/s model; a search that ends at
`alpha_max` or leaves the parameters unchanged emits a `RuntimeWarning`.
Bounds, when supplied as `bounds=(lo, hi)`, are enforced by an in-place
clamp after each step (projected gradient); line-search trials are projected
before they are evaluated.

## API

```python
sweepx.opt.optim.SteepestDescent(
    params,                         # iterable of torch.Tensor
    lr=1e-3,                        # used when line_search is None
    *,
    bounds=None,                    # (lo, hi) box; clamp after each step
    preconditioner=None,            # callable (grad, param) -> grad
    line_search=None,               # None or "strong_wolfe"
    c1=1e-4, c2=0.9,                # Wolfe constants
    alpha_init=1.0, alpha_max=10.0, # Wolfe step bracket, multiplies d = -P g
    max_ls_iter=25,                 # Wolfe iteration cap
    initial_step=None,              # first trial, parameter units (line search only)
)
```

## Drop-in usage

(a) Fixed-`lr` SD, no closure required. `lr` multiplies the (preconditioned)
raw gradient, so it is in (m/s) / (gradient units). Without a preconditioner
the stable `lr` is set by the stiffest part of the model: on the setup,
`lr=300` already diverges and `lr=100` lowers the misfit by less than 1 % in
10 steps. With the [pseudo-Hessian](../preconditioners/pseudo_hessian.md),
`lr=3e4` is stable and moves the model by about 8 m/s on the first step:

```python
import sweepx.opt

vp = init_vp.clone().requires_grad_(True)
precond = sweepx.opt.precond.PseudoHessianPrecond(eps=1e-5)
solver.compute_illumination = True   # impl='c': backward() fills the illumination maps

opt = sweepx.opt.optim.SteepestDescent([vp], lr=3e4, bounds=(1500.0, 5500.0),
                                       preconditioner=precond)
for _ in range(cfg["epochs"]):
    opt.zero_grad()
    loss = (solver(wavelet, sources, receivers, [vp]) - obs).pow(2).sum()
    loss.backward()
    S, R = solver.source_illumination, solver.receiver_illumination
    precond.update([vp], source_illum=S / S.max(), receiver_illum=R / R.max())
    opt.step()
```

(b) SD with the strong-Wolfe line search, closure required; `initial_step`
sizes the first trial in m/s:

```python
vp = init_vp.clone().requires_grad_(True)
opt = sweepx.opt.optim.SteepestDescent([vp], line_search="strong_wolfe",
                                       bounds=(1500.0, 5500.0), initial_step=20.0)

def closure():
    opt.zero_grad()
    loss = (solver(wavelet, sources, receivers, [vp]) - obs).pow(2).sum()
    loss.backward()
    return loss

for _ in range(cfg["epochs"]):
    opt.step(closure)
```

## When (not) to use

SD is **slow** on ill-conditioned problems - its convergence rate degrades as
$O((\kappa-1)/(\kappa+1))^k$ where $\kappa$ is the condition number of the
Hessian. FWI Hessians have $\kappa \sim 10^4$ in realistic settings, so SD is
mostly useful as a baseline. Reach for Nonlinear CG, L-BFGS, or Truncated
Newton for production work.

## References

- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer. Chapters 3.1-3.5 (line search) and 3.4 (Wolfe conditions).
  [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
