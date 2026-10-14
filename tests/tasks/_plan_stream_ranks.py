"""One rank of a plan-streaming FWI run, for test_plan_streaming_rank_sync.py.

Launched under ``python -m torch.distributed.run`` (gloo, CPU), or as a plain
process for the one-rank reference. ``--fake-dd`` swaps sweep's ModelParallel
for :class:`FakeModelParallel`, a CPU stand-in with the same gradient contract,
so the domain-decomposed loop runs without GPUs.

Writes ``<out>/rank<r>.npz``: the model this rank ends the run with (the leaf a
grid run steps, or the network's render), the network's parameters, and every
model the solver was called with; plus, on rank 0, the epoch-0 gradient
(``SWEEP_DUMP_GRAD``) and the run's own outputs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class _FakeDDForward(torch.autograd.Function):
    """ModelParallel's gradient contract, without the decomposition.

    Every rank models the whole domain and returns the traces of the receivers
    its tile owns. The backward sums every rank's adjoint source (each rank
    holds its own receivers' residuals) and hands back the gradient on this
    rank's tile interior only: in the model's own shape for a replicated model,
    the tile for a per-tile one -- what ``sweep.parallel.dd_propagator``
    returns.
    """

    @staticmethod
    def forward(ctx, dd, wavelet, sources, receivers, model):
        with torch.enable_grad():
            full = dd.global_model(model).detach().requires_grad_(True)
            rec = dd.prop(wavelet, sources, receivers, models=[full])
        ctx.dd, ctx.full, ctx.rec, ctx.shape = dd, full, rec, tuple(model.shape)
        own = dd._own_rec_idx
        # No receiver on this tile: one dummy trace, as ModelParallel records.
        return (rec[:, :, own] if own else rec[:, :, :1] * 0.0).detach().clone()

    @staticmethod
    def backward(ctx, grad_own):
        dd = ctx.dd
        g_rec = torch.zeros_like(ctx.rec)
        if dd._own_rec_idx:
            g_rec[:, :, dd._own_rec_idx] = grad_own
        dist.all_reduce(g_rec)
        (g,) = torch.autograd.grad(ctx.rec, ctx.full, g_rec)
        tile = g[dd.tile_slices()]
        if ctx.shape == dd.global_shape:
            out = torch.zeros_like(g)
            out[dd.tile_slices()] = tile
        else:
            out = tile.clone()
        return None, None, None, None, out


class FakeModelParallel:
    """CPU stand-in for ``sweep.parallel.dd_propagator.ModelParallel``."""

    def __init__(self, prop, mesh, seen):
        self.prop = prop
        self.global_shape = tuple(int(s) for s in prop._shape_phys)
        local, offsets = mesh.local_extent(self.global_shape)
        self.nyp, self.nxp = int(local[1]), int(local[2])
        self.y0, self.x0 = int(offsets[1]), int(offsets[2])
        self.local_shape = tuple(int(s) for s in local)
        self._own_rec_idx = []
        self._seen = seen

    def tile_slices(self):
        return (slice(None), slice(self.y0, self.y0 + self.nyp),
                slice(self.x0, self.x0 + self.nxp))

    def global_model(self, model):
        if tuple(model.shape) == self.global_shape:
            return model
        parts = [torch.empty_like(model) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, model.detach().contiguous())
        bounds = [None] * dist.get_world_size()
        dist.all_gather_object(bounds, (self.y0, self.x0))
        full = model.new_empty(self.global_shape)
        for (y0, x0), p in zip(bounds, parts):
            full[:, y0:y0 + self.nyp, x0:x0 + self.nxp] = p
        return full

    def __call__(self, wavelet, sources, receivers, models):
        rec = np.asarray(receivers)[0]                         # (nrec, 3): x, y, z
        own = ((rec[:, 0] >= self.x0) & (rec[:, 0] < self.x0 + self.nxp)
               & (rec[:, 1] >= self.y0) & (rec[:, 1] < self.y0 + self.nyp))
        self._own_rec_idx = np.flatnonzero(own).tolist()
        self._seen.append(models[0].detach().cpu().numpy().copy())
        return _FakeDDForward.apply(self, wavelet, sources, receivers, models[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--fake-dd", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    os.environ["SWEEP_DUMP_GRAD"] = "1"          # rank 0, epoch 0
    os.environ["SWEEP_DUMP_DIR"] = str(out)

    import sweep_tasks
    import sweep_tasks.tasks.fwi_plan_streaming as ps
    from sweep.propagator.torch import PropTorch
    from sweep_tasks import TaskRunner, load_task

    seen, opts, nets = [], [], []
    for name in ("_build_optimizer", "_build_reparam_optimizer"):
        def build(*a, _orig=getattr(ps, name), **k):
            opts.append(_orig(*a, **k))
            return opts[-1]
        setattr(ps, name, build)

    def build_net(*a, _orig=ps._build_reparam_net, **k):
        nets.append(_orig(*a, **k))
        return nets[-1]
    ps._build_reparam_net = build_net

    if args.fake_dd:
        ps._dd_wrap = lambda solver, mesh: FakeModelParallel(solver, mesh, seen)
    else:
        forward = PropTorch.forward

        def spy(self, wavelet, sources, receivers, *a, **k):
            seen.append(k["models"][0].detach().cpu().numpy().copy())
            return forward(self, wavelet, sources, receivers, *a, **k)
        PropTorch.forward = spy

    result = TaskRunner().run(load_task(args.spec))
    if result.status.state != "success":
        raise SystemExit(f"rank {rank}: {result.status.error}")
    if nets:
        with torch.no_grad():
            model = nets[-1]().detach().cpu().numpy()
        params = np.concatenate([p.detach().cpu().numpy().ravel()
                                 for p in nets[-1].parameters()])
    else:
        model = opts[-1].param_groups[0]["params"][0].detach().cpu().numpy()
        params = np.zeros(0, dtype=np.float32)
    np.savez(out / f"rank{rank}.npz", model=model, params=params,
             tree=sweep_tasks.__file__, task_dir=str(result.task_dir),
             **{f"seen{i}": s for i, s in enumerate(seen)})


if __name__ == "__main__":
    main()
