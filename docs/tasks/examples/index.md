# Examples

The repo ships two example families under `examples/`:

- **`examples/synthetic/`** — runs with **no downloads**. Models come from
  `sweep.datasets`, so the YAMLs are self-contained. Start here:
  ```bash
  sweep-tasks run examples/synthetic/01_forward_marmousi.yaml
  sweep-tasks run examples/synthetic/02_fwi_marmousi_single.yaml
  ```
- **`examples/field/`** — works on **real SEG-Y**, driving the full
  build-index → wavelet → invert pipeline.

See `examples/README.md` (in the repo) for the full config index.

## Worked walkthroughs

End-to-end walkthroughs with commands, figures, and measured numbers:

- **[Marmousi](../datasets/marmousi/README.md)** — synthetic forward → FWI,
  single-scale and multi-stage.
- **[Viking](../datasets/viking/README.md)** — real SEG-Y, the full
  build-index → wavelet → FWI → RTM / SIREN pipeline.
