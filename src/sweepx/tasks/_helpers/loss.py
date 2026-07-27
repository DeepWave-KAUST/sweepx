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
    # Optional early-time mute (mute-then-misfit on BOTH syn & obs): zero the
    # first ``time_mute_samples`` time samples so the strong early diving-wave
    # energy is removed and the misfit is dominated by the LATE wide-angle /
    # reservoir-reflection event (which carries the LVZ ~1 s delay). Canonical
    # 4-D layout is (ns, nt, nrec, nchan) with time on axis 1.
    _tmute = int(getattr(loss_spec, "time_mute_samples", 0) or 0)
    _tmute_late = int(getattr(loss_spec, "time_mute_late_samples", 0) or 0)
    if (_tmute > 0 or _tmute_late > 0) and syn.ndim == 4:
        import torch
        nt_ = syn.shape[1]
        _m = torch.ones(nt_, device=syn.device, dtype=syn.dtype)
        if _tmute > 0 and _tmute < nt_:
            _m[:_tmute] = 0.0                 # mute early (diving/first-arrival)
        if 0 < _tmute_late < nt_:
            _m[_tmute_late:] = 0.0            # mute late -> WINDOW [early, late]
        _m = _m.view(1, -1, 1, 1)
        syn = syn * _m
        obs = obs * _m
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
    if kind == "envelope":
        # Envelope (instantaneous-amplitude) misfit — cycle-skip robust for large
        # traveltime shifts. |analytic signal| via Hilbert transform (FFT along the
        # time axis = axis 1 of canonical (ns, nt, nrec, nchan)). Differentiable.
        import torch
        nt = syn.shape[1]
        f = torch.fft.fftfreq(nt, d=1.0).to(syn.device)
        step = torch.zeros(nt, device=syn.device, dtype=torch.float32)
        step[f > 0] = 2.0
        step[f == 0] = 1.0                      # DC (and Nyquist stays 0; negligible)

        def _env(x):
            X = torch.fft.fft(x.to(torch.float32), dim=1)
            a = torch.fft.ifft(X * step.view(1, nt, 1, 1), dim=1)
            return torch.abs(a)

        return (_env(syn) - _env(obs)) ** 2
    if kind == "ot":
        # 1-D optimal-transport (Wasserstein-1) misfit on the per-trace energy
        # distribution: robust to LARGE traveltime shifts (a shift tau -> CDF
        # offset -> |Fp-Fq| grows ~linearly, no cycle-skipping) and gives a
        # low-wavenumber (tomographic) gradient — the right tool for a smooth
        # velocity feature (LVZ) whose signature is a ~1 s wide-angle delay.
        import torch
        eps = 1.0e-12
        p = syn.to(torch.float32) ** 2                 # non-negative energy
        q = obs.to(torch.float32) ** 2
        p = p / (p.sum(dim=1, keepdim=True) + eps)      # per-trace distribution
        q = q / (q.sum(dim=1, keepdim=True) + eps)
        Fp = torch.cumsum(p, dim=1)                     # CDF along time
        Fq = torch.cumsum(q, dim=1)
        return (Fp - Fq).abs()                          # W1 = sum_t |Fp-Fq|
    if kind == "cc_traveltime":
        # Cross-correlation traveltime misfit (Luo & Schuster 1991). Per trace,
        # measure the time shift dt that aligns syn to obs via a DIFFERENTIABLE
        # soft-argmax of the (demeaned, unit-energy) cross-correlation, then
        # return 0.5*dt^2 broadcast across time (so the caller's .sum()/
        # global_norm yields the per-trace mean, like trace_cosine). Autodiff
        # produces the Luo-Schuster adjoint source ( ~ dt * d/dt syn ) exactly.
        #
        # Why this and not waveform FWI: the residual traveltime of the reservoir
        # reflection (already present in the model) back-projects as a smooth,
        # low-wavenumber TRANSMISSION update along its wavepath -> it moves the
        # background velocity (the LVZ) instead of stamping high-wavenumber
        # reflectivity, and it is immune to the ~0.4 s cycle-skip that kills
        # amplitude misfits. Canonical layout (ns, nt, nrec, nchan), time axis 1.
        import torch
        ns_c, nt_c, nr_c, nc_c = syn.shape
        s = syn.to(torch.float32)
        o = obs.to(torch.float32)
        # demean + unit-energy per trace (along time) -> CC is a correlation coeff
        s = s - s.mean(dim=1, keepdim=True)
        o = o - o.mean(dim=1, keepdim=True)
        s = s / (s.pow(2).sum(dim=1, keepdim=True).sqrt() + 1.0e-12)
        o = o / (o.pow(2).sum(dim=1, keepdim=True).sqrt() + 1.0e-12)
        # full cross-correlation CC[tau] = sum_t s(t) o(t+tau) via FFT.
        nfft = int(2 * nt_c)
        S = torch.fft.rfft(s, n=nfft, dim=1)
        O = torch.fft.rfft(o, n=nfft, dim=1)
        cc = torch.fft.irfft(torch.conj(S) * O, n=nfft, dim=1)  # (ns, nfft, nr, nc)
        # reorder to lags tau = -(nt-1)..(nt-1); center at tau=0
        cc = torch.cat([cc[:, -(nt_c - 1):], cc[:, :nt_c]], dim=1)  # (ns, 2nt-1, nr, nc)
        lags = torch.arange(-(nt_c - 1), nt_c, device=syn.device, dtype=torch.float32)
        L = int(getattr(loss_spec, "cc_max_lag_samples", 0) or (nt_c // 4))
        keep = lags.abs() <= L
        cc = cc[:, keep]                                       # (ns, 2L+1, nr, nc)
        lag_k = lags[keep].view(1, -1, 1, 1)
        beta = float(getattr(loss_spec, "cc_beta", 30.0))
        w = torch.softmax(beta * cc, dim=1)                    # soft-argmax weights
        dt_samp = (w * lag_k).sum(dim=1, keepdim=True)         # (ns,1,nr,nc) in samples
        dt_s = dt_samp * float(_LOSS_DT[0])                    # -> seconds
        per_trace = 0.5 * dt_s.pow(2)                          # (ns,1,nr,nc)
        return per_trace.expand(ns_c, nt_c, nr_c, nc_c)
    raise ValueError(f"Unknown loss kind '{kind}'.")


# The cross-correlation traveltime misfit needs dt (s) to report the shift in
# physical units. The runner sets this once per run before the misfit is called
# (a module-level 1-element list avoids threading dt through every _loss_sum
# call site). Defaults to 1.0 -> dt is then measured in SAMPLES, still a valid
# (rescaled) misfit; the runner override makes the lr physically meaningful.
_LOSS_DT = [1.0]


def set_loss_dt(dt):
    """Register the time step (s) used by the cc_traveltime misfit."""
    _LOSS_DT[0] = float(dt)


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


def _diving_window_mask(chunk_src, chunk_rec, node_asinh, nt, dt, dh, loss_spec, dev,
                        frame_delay_s=0.0):
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
      frame_delay_s: MECHANICAL offset between the pick database's time datum
                  and the misfit's time axis, which the caller knows and the
                  user should not have to re-type. Concretely: a SIREN-pipeline
                  wavelet carries a ``source_delay_s`` zero-prepad, so the runner
                  rolls obs LATER by ``obs_prepad_samples`` to line it up with
                  syn — while the picks (and hence the asinh fit) are on the RAW
                  field-data datum. Pass ``obs_prepad_samples * dt``.
                  Without it the window sits that far too early: it still brackets
                  the first break (the window is wide enough to absorb the error,
                  so a picks-inside-window check CANNOT see this bug), but it is
                  de-centred — on a production field wavelet the delay error
                  shrinks the post-arrival window from ``minwin`` 0.5 s to 0.3 s,
                  which at 2-4 Hz is barely one cycle of the wavelet.
                  ``loss_spec.diving_obs_delay_s`` is added ON TOP as the user's
                  own correction.
    Returns float32 mask (nsh, nt, nrec, 1) on ``dev``.
    """
    import torch

    pre = float(loss_spec.diving_pre_s)
    minwin = float(loss_spec.diving_minwin_s)
    taper = max(float(loss_spec.diving_taper_s), dt)
    vw = float(loss_spec.diving_water_vel)
    delay = float(loss_spec.diving_obs_delay_s) + float(frame_delay_s)
    dh = float(dh)

    # Built on ``dev`` in float32. The (nt, nrec) grid is the whole cost, and
    # doing it in float64 on the CPU measured 1.6 s per node = ~37 s/iter at
    # batchsize 24 — enough to dominate the solver, for what is a geometry-only
    # constant. Times here are O(10 s) at millisecond dt, comfortably inside
    # float32's resolution.
    f32 = torch.float32
    src = torch.as_tensor(np.asarray(chunk_src), dtype=f32, device=dev)  # (nsh,npts,ndim)
    rec = torch.as_tensor(np.asarray(chunk_rec), dtype=f32, device=dev)  # (nsh,nrec,ndim)
    nsh, nrec = rec.shape[0], rec.shape[1]
    # node position = mean of the shot points (single-point shot -> itself)
    node = src.mean(dim=1, keepdim=True)                               # (nsh,1,ndim)
    off = torch.linalg.norm((rec - node), dim=2) * dh                  # (nsh,nrec) metres
    par = torch.as_tensor(np.asarray(node_asinh), dtype=f32, device=dev)  # (nsh,3)
    ok = torch.isfinite(par).all(dim=1)                               # (nsh,) valid rows
    # Replace NaN/invalid rows with a finite dummy BEFORE the math so 0*NaN
    # can't poison the taper; the NaN shots are overridden to all-ones below.
    par = torch.where(ok.view(nsh, 1), par,
                      torch.tensor([0.1, 1800.0, 0.8], dtype=f32, device=dev))
    t0 = par[:, 0:1]; v0 = par[:, 1:2].clamp(min=1.0); k = par[:, 2:3].clamp(min=1e-3)
    center = t0 + (2.0 / k) * torch.arcsinh(k * off / (2.0 * v0))      # (nsh,nrec)
    ttop = center - pre + delay
    tbot = torch.maximum(off / vw, center + minwin) + delay
    t = torch.arange(int(nt), dtype=f32, device=dev).view(1, int(nt), 1) * dt
    a = ttop.view(nsh, 1, nrec); b = tbot.view(nsh, 1, nrec)
    # Cosine-tapered box as the product of two clamped cosine ramps. Identical
    # to the piecewise core+up+dn form (each ramp is flat 1 across the core and
    # 0 beyond the taper) but with ~3x fewer (nt, nrec) temporaries.
    up = ((t - (a - taper)) / taper).clamp_(0.0, 1.0)
    dn = ((b + taper - t) / taper).clamp_(0.0, 1.0)
    w = (0.5 - 0.5 * torch.cos(np.pi * up)) * (0.5 - 0.5 * torch.cos(np.pi * dn))
    w = torch.where(ok.view(nsh, 1, 1), w, torch.ones_like(w))  # NaN-param shot -> no-op
    return w.unsqueeze(-1)                                            # (nsh,nt,nrec,1)


def _diving_asinh_for_nodes(db_path, node_model_xy, tol_m=25.0, min_picks=50):
    """Match a diving-wave pick database's per-node moveout params onto this
    run's nodes, ready for ``_diving_window_mask``'s ``node_asinh``.

    The database stores ``node_rec`` in the SAME rotated model frame as
    ``frame.to_model(plan.group_xyz)``, so the match is a plain 2-D lookup in
    metres — no grid/origin conversion, hence immune to the model_plan crop and
    the in-window filter's group renumbering.

    Among the candidates inside ``tol_m`` this takes the one with the MOST
    picks, NOT the nearest. A survey node can appear TWICE in the database (a
    full deployment plus a small remnant, 0-4 m apart), and the remnant's asinh
    fit is degenerate — too few picks, so v0 pins at the fit bound and the
    centre lands ~1 s off. They sit closer together than the match is precise,
    so nearest-neighbour picks between them by coin flip; on the production
    field plan that handed a few nodes a single-digit-pick fit, which would
    have muted their diving wave away entirely. ``node_fitok`` does NOT catch
    this (it reads 667/667 ok), hence the explicit ``min_picks`` floor.

    Nodes with no usable fit get a NaN row, which ``_diving_window_mask`` turns
    into an all-ones no-op mask; the caller is expected to report that count,
    since an unwindowed node contributes its FULL record (reflections +
    multiples) to a diving-wave-only misfit.

    Args:
      node_model_xy: (G, 2) rotated model-frame xy of this run's nodes.
    Returns (float32 (G, 3) of (t0, v0, k), n_matched).
    """
    from scipy.spatial import cKDTree

    db = np.load(db_path)
    rec = np.asarray(db["node_rec"], dtype=np.float64)      # (N,3) rotated metres
    par = np.asarray(db["node_asinh"], dtype=np.float64)    # (N,3) (t0, v0, k)
    if "node_npick" in db.files:
        npick = np.asarray(db["node_npick"], dtype=np.int64)
    else:
        npick = np.bincount(np.asarray(db["node_inv"]), minlength=len(par))
    good = np.flatnonzero(np.isfinite(par).all(axis=1) & (npick >= int(min_picks)))
    out = np.full((len(node_model_xy), 3), np.nan, dtype=np.float32)
    if good.size == 0:
        return out, 0
    xy = np.asarray(node_model_xy, dtype=np.float64)
    cand = cKDTree(rec[good, :2]).query_ball_point(xy, r=float(tol_m))
    n_hit = 0
    for g, c in enumerate(cand):
        if not c:
            continue
        best = good[c[int(np.argmax(npick[good[c]]))]]
        out[g] = par[best].astype(np.float32)
        n_hit += 1
    return out, n_hit


def _mask_chunk(data_mask, chunk, dev):
    """Slice the per-shot data mask for a chunk (or broadcast it when its
    leading dim is 1). Returns None when no mask is configured."""
    if data_mask is None:
        return None
    m = data_mask if data_mask.shape[0] == 1 else data_mask[chunk]
    return m.to(dev)
