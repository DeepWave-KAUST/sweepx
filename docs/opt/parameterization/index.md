# Parameterization

`sweep-opt`'s parameterizations let you optimize over a latent variable while
the FWI solver still sees the physical model.

```python
class Parameterization:
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent -> physical model the solver consumes."""

    def inverse(self, physical: torch.Tensor) -> torch.Tensor:
        """Initial physical model -> latent state to optimize."""
```

This decouples the optimization variable from the physical bounds, which is
useful for:

- enforcing velocity bounds via a `sigmoid` map
- working in `log(velocity)` space for stability
- inverting for a perturbation `dv` rather than absolute `v`
- replacing the entire model by the output of a neural network (DIP-FWI)

## Available parameterizations

| Name | Class | Tier | Notes |
|---|---|---|---|
| _none yet_ | - | - | - |
