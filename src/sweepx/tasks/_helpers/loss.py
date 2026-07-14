"""Misfit / loss helpers.

Extracted verbatim from ``runner.py`` — behaviour is bit-identical; only the
location changed. Consumed by the fwi / multisource / freqsel / rtm / lsrtm
task paths, so it lives here as a single source of truth.
"""
import numpy as np


def _compute_loss(syn, obs, loss_spec):
    """Elementwise misfit via `sweep_loss` functional API.

    Returns the pointwise loss tensor; the caller is responsible for the
    final reduction (so multi-rank averaging stays in the runner's hands).

    For ``trace_cosine`` (a per-trace amplitude-normalised correlation misfit
    rather than a pointwise function), we still return a pointwise tensor
    by broadcasting the per-trace value back across the time axis. This
    keeps the caller's ``.sum() / global_norm`` recipe yielding the correct
    per-trace mean, regardless of the trace count or sample count.
    """
    from sweep_loss import huber_loss, l1_loss, l2_loss

    kind = loss_spec.kind
    if kind == "mse":
        # half=False matches the legacy `(syn-obs)**2` (not `0.5 * ...`).
        return l2_loss(syn, obs, reduction="none", half=False)
    if kind == "l1":
        return l1_loss(syn, obs, reduction="none")
    if kind == "huber":
        return huber_loss(syn, obs, delta=float(loss_spec.huber_delta), reduction="none")
    if kind == "trace_cosine":
        # sweep-loss expects canonical (ns, nt, nrec, nchan). Both backends
        # (eager + c, after geophyai 21041c5) deliver this layout natively
        # so no permutation is needed. The historical 3-D fallback
        # ``(nt, nrec, 1)`` is dropped: sweep no longer produces 3-D, and
        # the axis-0=time assumption disagreed with ``sweep_loss``'s 3-D
        # convention ``(nshots, nt, nrec)`` (time on axis 1) — a 3-D
        # input would have been silently misinterpreted.
        from sweep_loss import global_correlation_loss
        eps = float(getattr(loss_spec, "trace_cosine_eps", 1.0e-8))
        demean = bool(getattr(loss_spec, "trace_cosine_demean", True))

        if syn.ndim != 4:
            raise ValueError(
                f"trace_cosine: expected canonical 4-D syn "
                f"(ns, nt, nrec, nchan); got {syn.ndim}-D shape "
                f"{tuple(syn.shape)}. sweep backends always return 4-D."
            )

        per_trace = global_correlation_loss(
            syn, obs,
            offset_one=True, demean=demean, eps=eps, reduction="none",
        )  # (N,) where N = ns * nrec * nchan after sweep-loss flatten

        # Reshape back to per-trace canonical and broadcast across time so
        # the caller's .sum() / global_norm yields mean(1-cos) over traces.
        ns_c, nt_c, nr_c, nc_c = syn.shape
        return per_trace.view(ns_c, 1, nr_c, nc_c).expand(ns_c, nt_c, nr_c, nc_c)
    raise ValueError(f"Unknown loss kind '{kind}'.")


def _loss_sum(syn, obs_chunk, loss_spec, mask_chunk=None, *, window_mode=False):
    """Pointwise misfit summed, optionally weighted by a per-sample data mask.
    ``mask_chunk=None`` is bit-identical to the legacy ``_compute_loss(...).sum()``.

    ``window_mode=True`` applies the mask MUTE-THEN-MISFIT: syn & obs are
    multiplied by the mask BEFORE the misfit. For ``trace_cosine`` this makes the
    cosine a true windowed correlation (over the kept samples) instead of a
    per-trace weight applied AFTER the full-trace correlation (which only
    reweights and never windows). Used by the diving-wave window."""
    if window_mode and mask_chunk is not None:
        return _compute_loss(syn * mask_chunk, obs_chunk * mask_chunk, loss_spec).sum()
    pw = _compute_loss(syn, obs_chunk, loss_spec)
    if mask_chunk is not None:
        pw = pw * mask_chunk
    return pw.sum()


def _diving_window_mask(chunk_src, chunk_rec, node_asinh, nt, dt, dh, loss_spec, dev):
    """On-the-fly diving-wave mute mask for one shot-chunk.

    CRG reciprocity: ``chunk_src`` is the OBN node ("shot"), ``chunk_rec`` are
    the survey sources acting as the solver's receivers. For every (shot,
    receiver) pair the horizontal+depth offset is read from the grid geometry
    and the window is ``[asinh(off)-pre, off/water_vel]`` (bottom hugs the water
    direct), min width ``minwin``, cosine ``taper``, shifted by ``obs_delay``.

    Args:
      chunk_src: (nsh, npts, ndim) int grid indices of the shot node(s).
      chunk_rec: (nsh, nrec, ndim) int grid indices of the receivers (sources).
      node_asinh: (nsh, 3) float32 per-shot moveout params (t0, v0, k) matched
                  to this chunk's nodes; NaN row -> that shot gets an all-ones
                  (no-op) mask so it is not silently muted.
      nt, dt: time samples / step (s) of the syn/obs axis.
    Returns float32 mask (nsh, nt, nrec, 1) on ``dev``.
    """
    import torch

    pre = float(loss_spec.diving_pre_s)
    minwin = float(loss_spec.diving_minwin_s)
    taper = max(float(loss_spec.diving_taper_s), dt)
    vw = float(loss_spec.diving_water_vel)
    delay = float(loss_spec.diving_obs_delay_s)
    dh = float(dh)

    src = torch.as_tensor(np.asarray(chunk_src), dtype=torch.float64)   # (nsh,npts,ndim)
    rec = torch.as_tensor(np.asarray(chunk_rec), dtype=torch.float64)   # (nsh,nrec,ndim)
    nsh, nrec = rec.shape[0], rec.shape[1]
    # node position = mean of the shot points (single-point shot -> itself)
    node = src.mean(dim=1, keepdim=True)                               # (nsh,1,ndim)
    off = torch.linalg.norm((rec - node), dim=2) * dh                  # (nsh,nrec) metres
    par = torch.as_tensor(np.asarray(node_asinh), dtype=torch.float64) # (nsh,3)
    ok = torch.isfinite(par).all(dim=1)                               # (nsh,) valid rows
    # Replace NaN/invalid rows with a finite dummy BEFORE the math so 0*NaN
    # can't poison the taper; the NaN shots are overridden to all-ones below.
    par = torch.where(ok.view(nsh, 1), par, torch.tensor([0.1, 1800.0, 0.8], dtype=torch.float64))
    t0 = par[:, 0:1]; v0 = par[:, 1:2].clamp(min=1.0); k = par[:, 2:3].clamp(min=1e-3)
    center = t0 + (2.0 / k) * torch.arcsinh(k * off / (2.0 * v0))      # (nsh,nrec)
    twd = off / vw
    ttop = center - pre + delay
    tbot = torch.maximum(twd, center + minwin) + delay
    t = (torch.arange(int(nt), dtype=torch.float64) * dt)             # (nt,)
    tt = t.view(1, int(nt), 1)
    a = ttop.view(nsh, 1, nrec); b = tbot.view(nsh, 1, nrec)
    core = ((tt >= a) & (tt <= b)).to(torch.float64)
    up = ((tt >= a - taper) & (tt < a)).to(torch.float64)
    core = core + up * 0.5 * (1 - torch.cos(np.pi * (tt - (a - taper)) / taper))
    dn = ((tt > b) & (tt <= b + taper)).to(torch.float64)
    core = core + dn * 0.5 * (1 + torch.cos(np.pi * (tt - b) / taper))
    core = torch.where(ok.view(nsh, 1, 1), core, torch.ones_like(core))  # NaN-param shot -> all ones (no-op)
    return core.to(torch.float32).unsqueeze(-1).to(dev)               # (nsh,nt,nrec,1)


def _mask_chunk(data_mask, chunk, dev):
    """Slice the per-shot data mask for a chunk (or broadcast it when its
    leading dim is 1). Returns None when no mask is configured."""
    if data_mask is None:
        return None
    m = data_mask if data_mask.shape[0] == 1 else data_mask[chunk]
    return m.to(dev)
