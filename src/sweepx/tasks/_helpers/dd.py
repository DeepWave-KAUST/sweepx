"""Domain-decomposition (DD) tile helpers + local-window crop. Verbatim from runner.py."""
import os

import numpy as np
import torch


def _dd_config():
    """Returns (dd_on, py, px, render_chunk) from env."""
    import os
    return (os.environ.get("SWEEP_DD_ENABLE") == "1",
            int(os.environ.get("SWEEP_DD_PY", "1")),
            int(os.environ.get("SWEEP_DD_PX", "1")),
            int(os.environ.get("SWEEP_DD_RENDER_CHUNK", "8")))


def _dd_wrap(solver, mesh):
    """Wrap a PropTorch in ModelParallel for domain-decomposed fwd/adjoint."""
    from sweep.parallel.dd_propagator import ModelParallel
    return ModelParallel(solver, mesh)


def _dd_tile_bounds(ddp):
    """This rank's tile INTERIOR bounds in GLOBAL coords, as render_window args.
    Reads ddp.global_shape so it is correct after any per-stage mesh rebuild."""
    shape = ddp.global_shape
    nz = int(shape[0])
    if len(shape) == 3:
        return (0, nz, int(ddp.y0), int(ddp.y0 + ddp.nyp),
                int(ddp.x0), int(ddp.x0 + ddp.nxp))
    return (0, nz, int(ddp.x0), int(ddp.x0 + ddp.nxp))


def _dd_render_tile(reparam_net, bounds, rc):
    """Detached per-tile vp via z-chunked render_window (bounds render memory)."""
    import torch
    tz0, tz1, rest = bounds[0], bounds[1], bounds[2:]
    slabs = []
    with torch.no_grad():
        for z0 in range(tz0, tz1, rc):
            z1 = min(tz1, z0 + rc)
            slabs.append(reparam_net.render_window(z0, z1, *rest))
    return torch.cat(slabs, dim=0).detach().requires_grad_(True)


def _dd_backward_tile(reparam_net, model_leaf, bounds, rc):
    """Push the tile velocity grad through net params (z-chunked), then
    all_reduce net-param grads across tiles. The all_reduce runs
    UNCONDITIONALLY (zero-filling missing grads) so the collective stays
    consistent even for tiles whose model_leaf.grad is None."""
    import torch
    import torch.distributed as _td
    tz0, tz1, rest = bounds[0], bounds[1], bounds[2:]
    g = model_leaf.grad
    if g is not None:
        for z0 in range(tz0, tz1, rc):
            z1 = min(tz1, z0 + rc)
            reparam_net.render_window(z0, z1, *rest).backward(g[z0 - tz0:z1 - tz0])
    if _td.is_available() and _td.is_initialized():
        for p in reparam_net.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            _td.all_reduce(p.grad, op=_td.ReduceOp.SUM)


def _dd_render_tile_multi(reparam_net, bounds, rc, n_params):
    """Multi-parameter DD tile render (MultiParamINR): return a list of
    ``n_params`` detached per-tile leaves. render_window yields ``(n, z-chunk, …)``;
    z-chunk over the tile, cat along z (axis 1), split into one leaf per channel."""
    import torch
    tz0, tz1, rest = bounds[0], bounds[1], bounds[2:]
    slabs = []
    with torch.no_grad():
        for z0 in range(tz0, tz1, rc):
            z1 = min(tz1, z0 + rc)
            slabs.append(reparam_net.render_window(z0, z1, *rest))   # (n, z1-z0, …)
    full = torch.cat(slabs, dim=1)                                   # (n, tile_z, …)
    return [full[i].detach().requires_grad_(True) for i in range(int(n_params))]


def _dd_backward_tile_multi(reparam_net, model_leaves, bounds, rc):
    """Push EACH channel's tile grad through the shared trunk (z-chunked
    render_window), then all_reduce net-param grads once across tiles. all_reduce
    runs unconditionally (zero-fill) so the collective stays consistent even when
    a tile's leaf grads are all None."""
    import torch
    import torch.distributed as _td
    tz0, tz1, rest = bounds[0], bounds[1], bounds[2:]
    grads = [lf.grad for lf in model_leaves]
    idx = [i for i, gg in enumerate(grads) if gg is not None]
    if idx:
        for z0 in range(tz0, tz1, rc):
            z1 = min(tz1, z0 + rc)
            fields = reparam_net.render_window(z0, z1, *rest)        # (n, z1-z0, …) w/ grad
            torch.autograd.backward([fields[i] for i in idx],
                                    [grads[i][z0 - tz0:z1 - tz0] for i in idx])
    if _td.is_available() and _td.is_initialized():
        for p in reparam_net.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            _td.all_reduce(p.grad, op=_td.ReduceOp.SUM)


def _compute_local_window(sources_chunk, receivers_chunk, full_shape, dh, win_spec):
    """Bounding-box crop enclosing the batch's sources + receivers.

    2-D returns ``(z0, z1, x0, x1)`` on a ``(nz, nx)`` grid with the last
    coord axis ``[x, z]``.

    3-D returns ``(z0, z1, y0, y1, x0, x1)`` on a ``(nz, ny, nx)`` grid
    with the last coord axis ``[x, y, z]`` (matching sweep's 3-D
    propagator convention). The y dimension is padded by
    ``win_spec.padding_y_m`` if set, else falls back to
    ``win_spec.padding_x_m``. ``min_width_m`` enforces a floor on the x
    extent in both 2-D and 3-D; it is not applied to y.

    Both branches clamp to the full-model shape. Callers detect the
    grid dimensionality via ``len(full_shape)`` or by the length of the
    returned tuple (4 vs. 6).
    """
    import numpy as _np
    ndim = len(full_shape)
    pad_x = int(float(win_spec.padding_x_m) / float(dh))
    pad_z = int(float(win_spec.padding_z_m) / float(dh))
    min_w = int(float(win_spec.min_width_m) / float(dh))

    if ndim == 2:
        nz, nx = int(full_shape[0]), int(full_shape[1])
        src_x = sources_chunk[:, 0]; src_z = sources_chunk[:, 1]
        rec_x = receivers_chunk[..., 0].reshape(-1); rec_z = receivers_chunk[..., 1].reshape(-1)
        all_x = _np.concatenate([src_x, rec_x])
        all_z = _np.concatenate([src_z, rec_z])
        x0 = max(0, int(all_x.min()) - pad_x)
        x1 = min(nx, int(all_x.max()) + pad_x + 1)
        if win_spec.full_depth:
            z0, z1 = 0, nz
        else:
            z0 = max(0, int(all_z.min()) - pad_z)
            z1 = min(nz, int(all_z.max()) + pad_z + 1)
        if min_w > 0 and (x1 - x0) < min_w:
            extra = min_w - (x1 - x0)
            x0 = max(0, x0 - extra // 2)
            x1 = min(nx, x0 + min_w)
            if (x1 - x0) < min_w:
                x0 = max(0, x1 - min_w)
        return int(z0), int(z1), int(x0), int(x1)

    if ndim == 3:
        nz, ny, nx = (int(v) for v in full_shape)
        pad_y_m = win_spec.padding_y_m if win_spec.padding_y_m is not None else win_spec.padding_x_m
        pad_y = int(float(pad_y_m) / float(dh))
        src_x = sources_chunk[:, 0]; src_y = sources_chunk[:, 1]; src_z = sources_chunk[:, 2]
        rec_x = receivers_chunk[..., 0].reshape(-1)
        rec_y = receivers_chunk[..., 1].reshape(-1)
        rec_z = receivers_chunk[..., 2].reshape(-1)
        all_x = _np.concatenate([src_x, rec_x])
        all_y = _np.concatenate([src_y, rec_y])
        all_z = _np.concatenate([src_z, rec_z])
        x0 = max(0, int(all_x.min()) - pad_x)
        x1 = min(nx, int(all_x.max()) + pad_x + 1)
        y0 = max(0, int(all_y.min()) - pad_y)
        y1 = min(ny, int(all_y.max()) + pad_y + 1)
        if win_spec.full_depth:
            z0, z1 = 0, nz
        else:
            z0 = max(0, int(all_z.min()) - pad_z)
            z1 = min(nz, int(all_z.max()) + pad_z + 1)
        if min_w > 0 and (x1 - x0) < min_w:
            extra = min_w - (x1 - x0)
            x0 = max(0, x0 - extra // 2)
            x1 = min(nx, x0 + min_w)
            if (x1 - x0) < min_w:
                x0 = max(0, x1 - min_w)
        return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)

    raise NotImplementedError(
        f"_compute_local_window: only 2-D and 3-D grids supported; got shape {full_shape}."
    )


def _rebase_geometry_to_window(sources_chunk, receivers_chunk, z0, x0, *, y0=None):
    """Shift grid-index source / receiver coords to window-local origin.

    Args:
        sources_chunk    : ``(B, 2)`` or ``(B, 3)`` grid indices.
        receivers_chunk  : ``(B, nrec, 2)`` or ``(B, nrec, 3)`` grid indices.
        z0, x0           : window origin in z and x (always required).
        y0               : window origin in y. Pass for 3-D inputs; leave
                           ``None`` for 2-D. The function picks the axis
                           layout from ``y0`` rather than from the input
                           shape so 2-D callers can keep their existing
                           kwargs (``z0=..., x0=...``) unchanged.

    Returns fresh int64 arrays (not views); axis convention matches the
    rest of the runner: ``[x, z]`` (2-D) / ``[x, y, z]`` (3-D) on the
    last coord axis.
    """
    import numpy as _np
    s = _np.asarray(sources_chunk, dtype=_np.int64).copy()
    r = _np.asarray(receivers_chunk, dtype=_np.int64).copy()
    if y0 is None:
        s[:, 0] -= int(x0); s[:, 1] -= int(z0)
        r[..., 0] -= int(x0); r[..., 1] -= int(z0)
    else:
        s[:, 0] -= int(x0); s[:, 1] -= int(y0); s[:, 2] -= int(z0)
        r[..., 0] -= int(x0); r[..., 1] -= int(y0); r[..., 2] -= int(z0)
    return s, r
