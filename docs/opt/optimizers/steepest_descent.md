# Steepest Descent

`sweep_opt.optim.SteepestDescent` is the simplest first-order optimizer in the
library. It exists for two reasons:

1. **Baseline.** Every other optimizer is benchmarked against SD on the
   Marmousi 2-D acoustic FWI problem - you need to know what "no curvature
   information" looks like before you can claim a speed-up.
2. **Reusable Wolfe line-search machinery.** The strong-Wolfe routine that
   ships alongside SD is the same routine used by the Nonlinear CG and L-BFGS
   optimizers later.

## Update rule

Without line search:

$$
\theta_{k+1} = \theta_k - \alpha\, P\, g_k
$$

where $g_k = \nabla f(\theta_k)$, $P$ is an optional preconditioner (identity
by default), and $\alpha$ is the fixed `lr`.

With `line_search="strong_wolfe"`:

$$
\alpha_k = \arg\min_{\alpha > 0}\; f\!\bigl(\theta_k + \alpha d_k\bigr)
\quad \text{subject to strong-Wolfe},\qquad d_k = -P g_k
$$

solved by Algorithms 3.5 + 3.6 of Nocedal & Wright. Bounds, when supplied as
`bounds=(lo, hi)`, are enforced by an in-place clamp after each step
(projected gradient).

## API

```python
sweep_opt.optim.SteepestDescent(
    params,                         # iterable of torch.Tensor
    lr=1e-3,                        # used when line_search is None
    *,
    bounds=None,                    # (lo, hi) box; clamp after each step
    preconditioner=None,            # callable (grad, param) -> grad
    line_search=None,               # None or "strong_wolfe"
    c1=1e-4, c2=0.9,                # Wolfe constants
    alpha_init=1.0, alpha_max=10.0, # Wolfe step bracket
    max_ls_iter=25,                 # Wolfe iteration cap
)
```

## Drop-in usage

```python
import sweep_opt

vp = init_vp.clone().requires_grad_(True)

# (a) Plain SD - no closure required.
opt = sweep_opt.optim.SteepestDescent([vp], lr=cfg["lr"])
for _ in range(cfg["epochs"]):
    opt.zero_grad()
    loss = (solver(wavelet, sources, receivers, [vp]) - obs).pow(2).sum()
    loss.backward()
    opt.step()

# (b) SD with strong-Wolfe line search - closure required.
opt = sweep_opt.optim.SteepestDescent([vp], line_search="strong_wolfe",
                                       bounds=(1500.0, 5500.0))

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
Newton for production work (added in later commits).

## References

- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer. Chapters 3.1-3.5 (line search) and 3.4 (Wolfe conditions).
  [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
