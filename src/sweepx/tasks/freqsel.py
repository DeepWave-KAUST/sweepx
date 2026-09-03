"""Frequency-selection (frequency-division) source encoding for OBN FWI.

Implements the steady-state comb method (Tromp & Bachmann, 2019, GJI) on the
sweep mode-B super-shot path: every CRG node in the active pool continuously
emits ONE exclusive DFT-comb frequency; the last ``n_p`` samples of the record
form an integer-period window in which the bins are orthogonal, so the window
DFT separates the nodes exactly (deterministic zero crosstalk — no ±1 signs,
no reseeding, no shared-shot intersection sampling).

The observed side is a set of pre-extracted DTFT coefficients (one complex
number per trace per comb bin, fold-averaged onto surface cells by the
extraction job) — after extraction the inversion never touches SEG-Y.

Loss: per-node complex-cosine coherence (GCN),
``J_s = 1 - |<u_s, d_s>| / (|u_s| |d_s|)``, invariant to any per-node complex
scale — the source wavelet spectrum, excitation delay and sensor
coupling/polarity all cancel; NO wavelet input exists in this mode.

Validated end-to-end on a field OBN dataset in the low-frequency band
(multi-node DD):
see the project notes for the validation run.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "FrequencyComb",
    "FreqSelTargets",
    "PoolScheduler",
    "encoded_wavelet",
    "SteadyGCNLoss",
    "extract_shard",
    "extract_shard_gathers",
    "synthesize_shard",
]


def extract_shard(out_path: str, record, node_grid_xyz: np.ndarray,
                  trace_grid_xyz: np.ndarray, comb: FrequencyComb,
                  *, fold=None, chunk: int = 8,
                  verbose: bool = False, dt_record: float | None = None) -> str:
    """DTFT already-recorded common-node gathers onto the comb and write a shard.

    The observed-data half of the method, with no solver in it: ``record`` is
    whatever the acquisition (or a forward run) produced, one conventional
    gather per node, and the output is the same npz schema
    :class:`FreqSelTargets` reads and :func:`synthesize_shard` writes.

    ``record``          ``(n_nodes, nt, n_rec)`` or ``(n_nodes, nt, n_rec, 1)``.
    ``node_grid_xyz``   ``(n_nodes, ndim)`` grid indices, (x, z) or (x, y, z).
    ``trace_grid_xyz``  ``(n_rec, ndim)`` shared across nodes, or
                        ``(n_nodes, n_rec, ndim)`` per node.
    ``fold``            optional ``(n_nodes * n_rec,)`` stack count per item;
                        defaults to 1 (one trace per cell, the synthetic case).
    ``dt_record``       sample interval OF THE RECORD, when it differs from the
                        solver's ``comb.dt`` — field data is routinely 2 or
                        4 ms while the solver runs at 1 ms. Comb frequencies
                        are physical (``ks / (n_p * comb.dt)``) and do not
                        change; only the time axis the DTFT integrates over
                        does. Defaults to ``comb.dt``.

    Nodes need not share a trace count: see :func:`extract_shard_gathers` for
    the ragged field case (streamer CRGs, OBN nodes with varying live fold).

    Absolute scale and the time origin do not matter: the GCN misfit is
    invariant to a per-(node, bin) complex factor, which is exactly what a
    source spectrum, an excitation delay or a coupling constant contribute.
    What DOES matter is that ``comb`` here is the comb the inversion will
    configure — :class:`FreqSelTargets` refuses a shard whose ``n_p``, ``dt``
    or ``ks`` disagree.
    """
    rec = np.asarray(record)
    if rec.ndim == 4 and rec.shape[-1] == 1:
        rec = rec[..., 0]
    if rec.ndim != 3:
        raise ValueError(
            "record must be (n_nodes, nt, n_rec) or (n_nodes, nt, n_rec, 1); "
            f"got shape {tuple(np.shape(record))}")
    n_nodes, nt, n_rec = rec.shape

    nodes = np.asarray(node_grid_xyz, np.int64)
    if nodes.ndim != 2 or len(nodes) != n_nodes:
        raise ValueError(
            f"node_grid_xyz must be (n_nodes={n_nodes}, ndim); "
            f"got {nodes.shape}")
    ndim = nodes.shape[1]
    if ndim not in (2, 3):
        raise ValueError(f"node_grid_xyz ndim must be 2 or 3; got {ndim}")

    # Collisions are a property of THIS grid: a pair separated by less than dh
    # shares a cell here and may not on a finer band. Say so while the shard is
    # being built -- the inversion fires one source per cell, so a duplicate's
    # coefficients are computed and stored (these shards run to 100+ GB) for
    # rows that will then be dropped. See PoolScheduler.
    _dupe = len(nodes) - len(np.unique(nodes, axis=0))
    if _dupe:
        print(f"[freqsel] extract: {len(nodes)} nodes occupy "
              f"{len(nodes) - _dupe} distinct cells on this grid; {_dupe} "
              "node(s) share a cell and will be dropped at inversion time",
              flush=True)

    traces = np.asarray(trace_grid_xyz, np.int64)
    if traces.ndim == 2:
        if len(traces) != n_rec:
            raise ValueError(
                f"trace_grid_xyz must be (n_rec={n_rec}, ndim); "
                f"got {traces.shape}")
        traces = np.tile(traces, (n_nodes, 1))
    elif traces.ndim == 3:
        if traces.shape[:2] != (n_nodes, n_rec):
            raise ValueError(
                f"per-node trace_grid_xyz must be ({n_nodes}, {n_rec}, ndim); "
                f"got {traces.shape}")
        traces = traces.reshape(n_nodes * n_rec, traces.shape[-1])
    else:
        raise ValueError(
            f"trace_grid_xyz must be 2-D or 3-D; got {traces.shape}")
    if traces.shape[1] != ndim:
        raise ValueError(
            f"trace_grid_xyz ndim {traces.shape[1]} != node_grid_xyz ndim {ndim}")

    E = _comb_kernel(comb, nt, dt_record)
    D = np.empty((n_nodes * n_rec, comb.n_bins), np.complex64)
    for a in range(0, n_nodes, chunk):
        b = min(n_nodes, a + chunk)
        arr = np.asarray(rec[a:b], dtype=np.float64)      # (b-a, nt, n_rec)
        for i in range(b - a):
            D[(a + i) * n_rec:(a + i + 1) * n_rec] = \
                (arr[i].T @ E.T).astype(np.complex64)
        if verbose:
            print(f"[freqsel] extracted nodes {b}/{n_nodes}", flush=True)

    if fold is None:
        fold_arr = np.ones(n_nodes * n_rec, np.int32)
    else:
        fold_arr = np.asarray(fold, np.int32).reshape(-1)
        if len(fold_arr) != n_nodes * n_rec:
            raise ValueError(
                f"fold must have {n_nodes * n_rec} entries; got {len(fold_arr)}")

    np.savez(out_path,
             node_ids=np.arange(n_nodes, dtype=np.int32),
             node_grid_xyz=nodes.astype(np.int32),
             node_ptr=np.arange(n_nodes + 1, dtype=np.int64) * n_rec,
             D=D, fold=fold_arr,
             trace_grid_xyz=traces.astype(np.int32),
             freqs=comb.freqs, ks=comb.ks, qc_freqs=np.zeros(0),
             meta=json.dumps(dict(n_p=comb.n_p, dt_solver=comb.dt,
                                  synthetic=False, nt_record=int(nt),
                                  dt_record=float(comb.dt if dt_record is None
                                                  else dt_record))))
    return out_path


def _comb_kernel(comb: FrequencyComb, nt: int,
                 dt_record: float | None) -> np.ndarray:
    """``(n_bins, nt)`` DTFT kernel on the RECORD's clock.

    The comb frequencies are physical; ``dt_record`` only says how the record
    samples time. Conflating the two silently rescales every frequency by
    ``dt_record / comb.dt`` — a 4x error for 4 ms field data on a 1 ms solver.
    """
    dtr = comb.dt if dt_record is None else float(dt_record)
    if dtr <= 0:
        raise ValueError(f"dt_record must be > 0; got {dtr}")
    if comb.freqs.max() > 0.5 / dtr:
        raise ValueError(
            f"comb reaches {comb.freqs.max():.3f} Hz, above the record's "
            f"Nyquist {0.5 / dtr:.3f} Hz (dt_record={dtr} s)")
    t_axis = np.arange(nt, dtype=np.float64) * dtr
    return np.exp(-2j * np.pi * comb.freqs[:, None] * t_axis[None, :])


def extract_shard_gathers(out_path: str, gathers, comb: FrequencyComb,
                          *, dt_record: float | None = None,
                          verbose: bool = False) -> str:
    """Ragged sibling of :func:`extract_shard` — one gather at a time.

    Field acquisition does not hand you a rectangular ``(n_nodes, nt, n_rec)``
    block: a streamer CRG has whatever fold the spread happened to give that
    cell, and OBN nodes lose traces to dead channels. The shard schema has
    always supported this — ``node_ptr`` is a CSR pointer and
    :class:`FreqSelTargets` reads it as one — so nothing downstream changes;
    only the writer needed to stop assuming a common trace count.

    ``gathers`` is an iterable of ``(record, node_xyz, trace_xyz)`` per node:

    ``record``     ``(nt, n_rec_i)`` — time first, one column per trace.
    ``node_xyz``   ``(ndim,)`` grid indices for the node itself.
    ``trace_xyz``  ``(n_rec_i, ndim)`` grid indices, one row per trace.

    Streaming by design: gathers are consumed one at a time and only the
    coefficients are kept, so a survey whose gathers do not fit in host RAM
    still extracts in one pass.
    """
    D_parts, node_rows, trace_parts, ptr = [], [], [], [0]
    ndim = nt = None
    E = None
    for i, (record, node_xyz, trace_xyz) in enumerate(gathers):
        arr = np.asarray(record, np.float64)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(
                f"gather {i}: record must be (nt, n_rec); got {arr.shape}")
        node = np.asarray(node_xyz, np.int64).reshape(-1)
        trc = np.asarray(trace_xyz, np.int64)
        if trc.ndim != 2 or len(trc) != arr.shape[1]:
            raise ValueError(
                f"gather {i}: trace_xyz must be (n_rec={arr.shape[1]}, ndim); "
                f"got {trc.shape}")
        if ndim is None:
            ndim, nt = len(node), arr.shape[0]
            if ndim not in (2, 3):
                raise ValueError(f"node_xyz ndim must be 2 or 3; got {ndim}")
            E = _comb_kernel(comb, nt, dt_record)
        elif arr.shape[0] != nt:
            raise ValueError(
                f"gather {i}: nt={arr.shape[0]} != {nt} of the first gather; "
                "every gather must share one time axis")
        elif len(node) != ndim or trc.shape[1] != ndim:
            raise ValueError(f"gather {i}: inconsistent ndim")
        D_parts.append((arr.T @ E.T).astype(np.complex64))
        node_rows.append(node)
        trace_parts.append(trc)
        ptr.append(ptr[-1] + arr.shape[1])
        if verbose and (i + 1) % 25 == 0:
            print(f"[freqsel] extracted {i + 1} gathers", flush=True)
    if not D_parts:
        raise ValueError("gathers yielded nothing")

    n_nodes = len(D_parts)
    np.savez(out_path,
             node_ids=np.arange(n_nodes, dtype=np.int32),
             node_grid_xyz=np.asarray(node_rows, np.int32),
             node_ptr=np.asarray(ptr, np.int64),
             D=np.concatenate(D_parts, 0),
             fold=np.ones(ptr[-1], np.int32),
             trace_grid_xyz=np.concatenate(trace_parts, 0).astype(np.int32),
             freqs=comb.freqs, ks=comb.ks, qc_freqs=np.zeros(0),
             meta=json.dumps(dict(n_p=comb.n_p, dt_solver=comb.dt,
                                  synthetic=False, nt_record=int(nt),
                                  dt_record=float(comb.dt if dt_record is None
                                                  else dt_record))))
    return out_path


def synthesize_shard(out_path: str, solver, vp_true: torch.Tensor,
                     node_grid_xyz: np.ndarray, shot_grid_xyz: np.ndarray,
                     comb: FrequencyComb, wavelet: np.ndarray,
                     device, batch: int = 8, verbose: bool = False,
                     models: list | None = None) -> str:
    """Forward-model per-node conventional gathers through ``vp_true`` and
    write an extraction shard npz (same schema as the field extractor).

    Synthetic-test path (``synthesize_from_true``): nodes fire one at a time
    (batched, mode A) with the given transient ``wavelet``; each record is
    DTFT'd at the comb bins. Runs on the SAME solver instance as the
    inversion (record length = the encoded nt — wasteful but exact and free
    of a second solver build). Single-device only; the caller must not pass
    a ModelParallel wrapper.
    """
    n_nodes = len(node_grid_xyz)
    n_rec = len(shot_grid_xyz)
    nt = None
    wav_t = torch.tensor(wavelet, dtype=torch.float32, device=device)
    t_axis = None
    D = np.empty((n_nodes * n_rec, comb.n_bins), np.complex64)
    for a in range(0, n_nodes, batch):
        b = min(n_nodes, a + batch)
        src = node_grid_xyz[a:b].astype(np.int32)
        rec = np.repeat(shot_grid_xyz[None].astype(np.int32), b - a, axis=0)
        with torch.no_grad():
            # ``models`` overrides the default single-vp model list — e.g. VRZ
            # synth passes ``[vp_true, z_true]`` so the true obs uses the correct
            # multi-parameter physics.
            out = solver(wav_t, src, rec,
                         models=(models if models is not None else [vp_true]))[..., 0]
        arr = out.double().cpu().numpy()                 # (b-a, nt, n_rec)
        if nt is None:
            nt = arr.shape[1]
            t_axis = np.arange(nt, dtype=np.float64) * comb.dt
            E = np.exp(-2j * np.pi * comb.freqs[:, None] * t_axis[None, :])
        for i in range(b - a):
            D[(a + i) * n_rec:(a + i + 1) * n_rec] = \
                (arr[i].T @ E.T).astype(np.complex64)
        if verbose:
            print(f"[freqsel] synth nodes {b}/{n_nodes}", flush=True)
    ptr = np.arange(n_nodes + 1, dtype=np.int64) * n_rec
    np.savez(out_path,
             node_ids=np.arange(n_nodes, dtype=np.int32),
             node_grid_xyz=node_grid_xyz.astype(np.int32),
             node_ptr=ptr, D=D,
             fold=np.ones(n_nodes * n_rec, np.int32),
             trace_grid_xyz=np.tile(shot_grid_xyz.astype(np.int32),
                                    (n_nodes, 1)),
             freqs=comb.freqs, ks=comb.ks, qc_freqs=np.zeros(0),
             meta=json.dumps(dict(n_p=comb.n_p, dt_solver=comb.dt,
                                  synthetic=True, nt_record=int(nt))))
    return out_path


@dataclass(frozen=True)
class FrequencyComb:
    """Exclusive-bin comb: ``freqs[k] = ks[k] / (n_p * dt)``.

    ``n_p`` is the steady analysis-window length in solver samples; every
    frequency is an exact DFT bin of that window (integer periods -> bin
    orthogonality -> zero crosstalk). Constraints checked here:
    bins unique, ``0 < k < n_p/2`` and ``k_i + k_j != n_p`` (conjugate
    aliasing), pool size never exceeds the number of bins.
    """

    dt: float
    n_p: int
    ks: np.ndarray

    def __post_init__(self):
        ks = np.asarray(self.ks, np.int64)
        object.__setattr__(self, "ks", ks)
        if len(np.unique(ks)) != len(ks):
            raise ValueError("comb bins must be unique")
        if ks.min() <= 0 or ks.max() >= self.n_p // 2:
            raise ValueError("comb bins must satisfy 0 < k < n_p/2")
        s = set(int(k) for k in ks)
        if any((self.n_p - int(k)) in s for k in ks):
            raise ValueError("conjugate-aliasing pair k_i + k_j == n_p in comb")

    @property
    def freqs(self) -> np.ndarray:
        return self.ks / (self.n_p * self.dt)

    @property
    def n_bins(self) -> int:
        return len(self.ks)


class FreqSelTargets:
    """Pre-extracted obs coefficients on union surface cells.

    Loads one or more extraction shards (npz with keys ``node_ids,
    node_grid_xyz, node_ptr, D, trace_grid_xyz, freqs, ks, meta``), rebuilds
    the cross-shard union cell table, and exposes:

    - ``union_xyz`` (Nu, 3) int32 — solver receiver table (grid indices),
      bounded by the grid, independent of fold;
    - per-``(node, cell)`` coefficient items with node ownership — the
      per-node masks of the varying acquisition (nodes need NOT share
      receivers; missing coverage is simply absent from the item list).

    ``bind_ownership(own_cols)`` restricts to the union columns owned by this
    DD tile (from ``ModelParallel._own_rec_idx``) and uploads the local
    coefficient table to ``device`` — call it once after the first forward.
    """

    def __init__(self, pattern: str, comb: FrequencyComb, ny: int,
                 verbose: bool = False):
        paths = sorted(_glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no extraction shards match {pattern!r}")
        sizes, nnodes = [], []
        for pth in paths:
            with np.load(pth) as p:
                sizes.append(int(p["node_ptr"][-1]))
                nnodes.append(len(p["node_ids"]))
        self.n_items, self.n_nodes = sum(sizes), sum(nnodes)
        with np.load(paths[0]) as p:
            freqs = p["freqs"].copy()
            ks = p["ks"].astype(np.int64)
            meta = json.loads(str(p["meta"]))
        if abs(meta["dt_solver"] - comb.dt) > 1e-12:
            raise ValueError(
                "extraction comb does not match the configured comb "
                f"(shards: dt={meta['dt_solver']}; spec: dt={comb.dt})")
        shard_np = int(meta["n_p"])
        if shard_np == comb.n_p and np.array_equal(ks, comb.ks):
            self._bin_cols = None                       # the ordinary case
        else:
            # A configured comb may be an integer DECIMATION of the extracted
            # one. Orthogonality needs ``k * W / n_p`` whole, so the extracted
            # bins whose index is divisible by ``m`` stay mutually orthogonal
            # over a window ``W = n_p / m`` -- and their frequencies are
            # unchanged, ``(k/m) / (W*dt) == k / (n_p*dt)``. So a SHORTER
            # record can reuse shards extracted at the long one, no re-DTFT.
            #
            # This is what makes the bin count a run-time knob: the record
            # length is ``O + N/B``, so halving the comb halves the boundary
            # buffer, which is what actually decides how many cards a band
            # needs. Pick ``probe_samples`` highly composite at EXTRACTION time
            # and the whole ladder (m = 2, 3, 4, ...) opens up later.
            #
            # Amplitude is not a concern: the two windows differ by a real
            # normalisation, and the GCN loss is invariant to any per-node
            # complex scale.
            if comb.n_p <= 0 or shard_np % comb.n_p:
                raise ValueError(
                    "configured comb is neither the extracted comb nor an "
                    f"integer decimation of it (shards: n_p={shard_np}; "
                    f"spec: n_p={comb.n_p}; {shard_np}/{comb.n_p} is not whole)")
            m = shard_np // comb.n_p
            want = np.asarray(comb.ks, np.int64) * m
            order = np.argsort(ks)
            pos = np.searchsorted(ks[order], want)
            if pos.max(initial=-1) >= len(ks) or \
                    not np.array_equal(ks[order][np.clip(pos, 0, len(ks) - 1)], want):
                missing = int(want[0]) if len(want) else -1
                raise ValueError(
                    f"configured comb decimates the extracted one by m={m}, "
                    f"but bin k={missing}*... is absent from the shards "
                    f"(extracted k in [{int(ks.min())}, {int(ks.max())}]); "
                    "every configured k*m must exist in the extraction")
            self._bin_cols = order[pos]
            # Never silent: this branch also accepts m == 1, i.e. a plain
            # SUBSET of the extracted bins, which the old exact-equality check
            # rejected outright. That is physically fine (it is the same window,
            # just fewer nodes fired) but it is exactly the shape a wrong-band
            # config has, so say so rather than let it pass unremarked.
            print(f"[freqsel] comb decimation m={m}: using {len(comb.ks)} of "
                  f"{len(ks)} extracted bins, window {shard_np} -> {comb.n_p} "
                  f"samples", flush=True)
        nb = len(comb.ks) if self._bin_cols is not None else len(freqs)
        # Metadata-only pass — the coefficient table D is NOT loaded here.
        # Every DD rank constructs this object, so holding the full table per
        # rank (tens of GB on a full-survey node set) OOM-kills the host;
        # bind_ownership() streams D shard-by-shard keeping only owned rows.
        self._paths, self._sizes, self._nb = paths, sizes, nb
        # Grid dimensionality comes from the shards themselves: (x, y, z) for
        # 3-D, (x, z) for 2-D. Everything downstream (union table, solver
        # receiver table) follows this width, so a 2-D run needs no flag.
        with np.load(paths[0]) as p:
            self.ndim = int(np.asarray(p["trace_grid_xyz"]).shape[1])
        if self.ndim not in (2, 3):
            raise ValueError(
                f"trace_grid_xyz must be (N,2) for 2-D or (N,3) for 3-D; "
                f"got width {self.ndim} in {paths[0]}")
        xyz = np.empty((self.n_items, self.ndim), np.int32)
        self.node_of_item = np.empty(self.n_items, np.int32)
        self.node_grid = np.empty((self.n_nodes, self.ndim), np.int64)
        oi = on = 0
        for t, pth in enumerate(paths):
            with np.load(pth) as p:
                ni = sizes[t]
                xyz[oi:oi + ni] = p["trace_grid_xyz"]
                ptr = p["node_ptr"]
                for s in range(nnodes[t]):
                    self.node_of_item[oi + ptr[s]:oi + ptr[s + 1]] = on + s
                self.node_grid[on:on + nnodes[t]] = p["node_grid_xyz"]
            oi += ni
            on += nnodes[t]
            if verbose:
                print(f"[freqsel] shard {t + 1}/{len(paths)} indexed", flush=True)
        # Dedup receivers onto surface grid CELLS. 3-D: the lateral cell
        # (x, y), z forced to the surface. 2-D: the lateral coordinate is x
        # alone and the shards carry (x, z) pairs, so ``ny`` is unused.
        if self.ndim == 2:
            key = xyz[:, 0].astype(np.int64)
            ukey, self.union_col = np.unique(key, return_inverse=True)
            self.union_xyz = np.stack(
                [ukey, np.zeros_like(ukey)], -1).astype(np.int32)
        else:
            key = xyz[:, 0].astype(np.int64) * ny + xyz[:, 1]
            ukey, self.union_col = np.unique(key, return_inverse=True)
            self.union_xyz = np.stack(
                [ukey // ny, ukey % ny, np.zeros_like(ukey)], -1).astype(np.int32)
        self.n_union = len(ukey)
        # Receivers are unique by construction (np.unique above) — the solver
        # gets one entry per cell, never the 22.8 M raw (node, trace) pairs.
        # It matters for the same reason it does for sources: the adjoint
        # injects residuals through the same atomicAdd source kernel, so a
        # repeated cell would make the gradient depend on arrival order.
        assert len(np.unique(self.union_xyz, axis=0)) == self.n_union, \
            "receiver cells must be distinct"
        self._bound = None

    def bind_ownership(self, own_cols, device, bin_use=None) -> None:
        own_cols = np.asarray(own_cols, np.int64)
        g2l = np.full(self.n_union, -1, np.int64)
        g2l[own_cols] = np.arange(len(own_cols))
        loc = g2l[self.union_col]
        items = np.nonzero(loc >= 0)[0]
        lrow = loc[items]
        no = self.node_of_item[items]
        o = np.argsort(no, kind="stable")
        bounds = np.searchsorted(no[o], np.arange(self.n_nodes + 1))
        # Does any node have TWO items on the same union cell?  The loop form
        # sums both (two field traces, one modelled sample); a dense scatter
        # would keep only the last one.  So the batched form is only equivalent
        # when this is false -- decide it once, here, on the real index arrays,
        # not by hoping.
        _key = no.astype(np.int64) * np.int64(self.n_union) + lrow.astype(np.int64)
        dup_free = bool(len(np.unique(_key)) == len(_key))
        del _key
        # Stream the shards, keeping only this tile's rows, straight into
        # fp16 (the GCN is scale-invariant per node, so half-float relative
        # precision sits far below the steady-state residual); shards are
        # consecutive item ranges, so per-shard selection preserves the
        # global ``items`` order.
        #
        # Ranks take turns.  ``p["D"]`` decompresses a whole shard member, so
        # the host peak is one shard plus the owned blocks -- but only if one
        # rank is inside the loop at a time.  Four ranks entering together
        # multiply the shard term by four: on a production cascade the shards
        # run to hundreds of GB each, and a 4-rank run was OOM-killed at its host
        # allocation while the GPUs sat nearly idle.
        colmap = None
        if bin_use is not None:
            bu = np.asarray(bin_use, bool)
            if bu.shape != (self.n_nodes, self._nb):
                raise ValueError(f"bin_use must be ({self.n_nodes}, "
                                 f"{self._nb}); got {bu.shape}")
            colmap = np.full((self.n_nodes, self._nb), -1, np.int32)
            k = int(bu.sum(1).max(initial=0))
            for s in range(self.n_nodes):
                cs = np.nonzero(bu[s])[0]
                colmap[s, cs] = np.arange(len(cs), dtype=np.int32)
            Dh = np.zeros((len(items), max(k, 1), 2), np.float16)
            self._load_shards(Dh, loc, colmap=colmap, node_of_row=no)
            _tot = self.n_items * self._nb * 4 / 2 ** 30
            _now = self.n_items * max(k, 1) * 4 / 2 ** 30
            print(f"[freqsel] D column pruning: at most {k}/{self._nb} bins per "
                  f"node are scheduled (median {int(np.median(bu.sum(1)))}); "
                  f"table {_tot:.1f} -> {_now:.1f} GiB across ranks; "
                  f"dup_free={dup_free}", flush=True)
        else:
            Dh = np.empty((len(items), self._nb, 2), np.float16)
            self._load_shards(Dh, loc)
        # The observed table stays on the HOST by default.  The loss reads
        # ``D[dsel[s], bins[j]]`` -- one frequency column of one node's rows --
        # so a whole iteration touches well under 1% of it.  At production comb
        # sizes the resident table costs tens of GB of card and can force DD
        # purely to make it fit.
        # fit.  Set SWEEP_FREQSEL_D_ON_GPU=1 to restore the resident table.
        d_on_gpu = os.environ.get("SWEEP_FREQSEL_D_ON_GPU", "") == "1"
        Dt_ = torch.from_numpy(Dh)
        self._bound = {
            "D": Dt_.to(device) if d_on_gpu else Dt_,
            "D_on_gpu": d_on_gpu,
            # None -> D still has one column per comb bin (bins index it
            # directly).  Otherwise colmap[s, k] is node s's local column for
            # comb bin k, and -1 means "the schedule said this pair never
            # happens" -- a hard error at read time, never a silent zero.
            "colmap": colmap,
            "dup_free": dup_free,
            "groups": [torch.tensor(lrow[o[bounds[s]:bounds[s + 1]]],
                                    dtype=torch.long, device=device)
                       for s in range(self.n_nodes)],
            # dsel indexes D, so it has to live wherever D lives
            "dsel": [torch.tensor(o[bounds[s]:bounds[s + 1]], dtype=torch.long,
                                  device=device if d_on_gpu else "cpu")
                     for s in range(self.n_nodes)],
        }

    def _cols(self, blk):
        """The coefficient columns this run actually uses.

        ``None`` means the configured comb IS the extracted one -- the common
        case, and a plain slice. Otherwise the run decimates the extraction and
        only the divisible bins are read.
        """
        blk = np.asarray(blk)
        if self._bin_cols is None:
            return blk[:, :self._nb]
        return blk[:, self._bin_cols]

    @staticmethod
    def _memmap_member(pth, name):
        """Memory-map an uncompressed .npy member of a .npz.

        ``np.load`` materialises a whole member; at production comb sizes these run to hundreds of GB.  The
        writer leaves them STORED, so the bytes are already a plain .npy at a
        known offset and can be mapped instead -- each rank then touches only
        the rows it owns.
        """
        import zipfile
        from numpy.lib import format as _npf
        zf = zipfile.ZipFile(pth)
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            zf.close()
            return None                       # compressed: caller falls back
        with zf.open(info) as fh:
            ver = _npf.read_magic(fh)
            # Version-keyed public readers.  numpy.lib.format._read_array_header
            # is private and is not present in every numpy the cluster runs --
            # it exists on the box the tests ran on and not on the one the
            # cascade ran on, which is why this crashed 53 s into a 40 h job.
            rd = {(1, 0): getattr(_npf, "read_array_header_1_0", None),
                  (2, 0): getattr(_npf, "read_array_header_2_0", None)}.get(ver)
            if rd is None:
                zf.close()
                return None                   # unknown version: use np.load
            shape, order, dtype = rd(fh)
            hdr = fh.tell()                   # .npy header, inside the member
        zf.close()
        # Parse the local file header that is actually in the file.  Do not use
        # ZipInfo.FileHeader(): it re-synthesises one, and its extra field need
        # not match the stored bytes, which puts the map a few bytes off.
        with open(pth, "rb") as f:
            f.seek(info.header_offset)
            lfh = f.read(30)
            n_name = int.from_bytes(lfh[26:28], "little")
            n_extra = int.from_bytes(lfh[28:30], "little")
        off = info.header_offset + 30 + n_name + n_extra + hdr
        return np.memmap(pth, dtype=dtype, mode="r", offset=off, shape=shape,
                         order="F" if order else "C")

    def _load_shards(self, Dh, loc, colmap=None, node_of_row=None) -> None:
        """Fill ``Dh`` with this rank's rows, reading only those rows.

        The per-node max-abs scale has to come from the node's FULL row block
        so every DD rank derives the same value -- a per-rank max would
        desynchronise the cross-rank partial sums.  Raw field DTFT
        coefficients run past the half-float range (observed max|D| ~ 1.2e5 >
        65504 -> inf -> nan loss), so the scale has to exist before the cast.
        Every rank scans the node blocks assigned to it and the maxima are
        reduced with MAX, so each node is scanned exactly once and the read is
        spread over ``world_size`` ranks.
        """
        _ws, _rk, _dist = 1, 0, None
        try:
            import torch.distributed as _d
            if _d.is_available() and _d.is_initialized():
                _dist, _ws, _rk = _d, _d.get_world_size(), _d.get_rank()
        except Exception:
            pass
        off = oi = 0
        for t, pth in enumerate(self._paths):
            ni = self._sizes[t]
            m = loc[oi:oi + ni] >= 0
            n_own = int(m.sum())
            if n_own:
                Dm = self._memmap_member(pth, "D.npy")
                with np.load(pth) as p:
                    ptr = p["node_ptr"]
                    if Dm is None:                       # compressed fallback
                        Dm = p["D"]
                # Each node is scanned by exactly one rank and the maxima are
                # reduced with MAX, which reproduces the old rank-0 full pass
                # BIT FOR BIT while the read is spread over the ranks.
                #
                # The old shape was an operational problem, not just slow: rank 0
                # read the WHOLE shard (hundreds of GB at large comb sizes) while
                # every other rank sat in the broadcast. NCCL spin-waits occupy
                # SMs, so the idle ranks register as ~85% busy and the one rank
                # actually working registers as idle -- a per-GPU utilisation
                # rule then fires on GPU 0 alone (observed: 73.3% against
                # 84.8/84.8/83.5). It also cut the page-cache burst to 1/N.
                #
                # 0 is the identity for MAX here: blocks this rank did not scan
                # stay 0 and lose the reduction, and the 0 -> 1.0 substitution
                # afterwards is exactly the old ``a if a > 0 else 1.0``.
                sc = np.zeros(ni, np.float32)
                for s in range(_rk, len(ptr) - 1, _ws):
                    sc[ptr[s]:ptr[s + 1]] = np.abs(self._cols(
                        Dm[ptr[s]:ptr[s + 1]])).max(initial=0.0)
                if _ws > 1:
                    import torch as _t
                    # NCCL has no CPU backend, so the scale has to make the
                    # round trip through the device to be reduced at all.
                    _dev = (_t.device("cuda", _t.cuda.current_device())
                            if _t.cuda.is_available() else _t.device("cpu"))
                    _b = _t.from_numpy(sc).to(_dev)
                    _dist.all_reduce(_b, op=_dist.ReduceOp.MAX)
                    sc = _b.cpu().numpy()
                sc = np.where(sc > 0, sc, np.float32(1.0)).astype(np.float32)
                rows = np.nonzero(m)[0]
                rs = sc[rows][:, None]
                # chunked so the gather never holds more than a slice
                CH = 1 << 19
                if colmap is None:
                    for a0 in range(0, len(rows), CH):
                        r = rows[a0:a0 + CH]
                        blk = self._cols(np.asarray(Dm[r]))
                        Dh[off + a0:off + a0 + len(r), :, 0] = blk.real / rs[a0:a0 + len(r)]
                        Dh[off + a0:off + a0 + len(r), :, 1] = blk.imag / rs[a0:a0 + len(r)]
                        del blk
                else:
                    # Pruned: the kept columns differ per node, so walk the
                    # node blocks.  The READ is unchanged (a whole row is one
                    # page either way); what shrinks is what is kept.
                    for s_loc in range(len(ptr) - 1):
                        lo = np.searchsorted(rows, ptr[s_loc])
                        hi = np.searchsorted(rows, ptr[s_loc + 1])
                        if hi <= lo:
                            continue
                        s_glb = int(node_of_row[off + lo])
                        cs = np.nonzero(colmap[s_glb] >= 0)[0]
                        if len(cs) == 0:
                            continue
                        cs = cs[np.argsort(colmap[s_glb][cs])]
                        for a0 in range(lo, hi, CH):
                            r = rows[a0:min(a0 + CH, hi)]
                            blk = self._cols(np.asarray(Dm[r]))[:, cs]
                            d = off + a0
                            Dh[d:d + len(r), :len(cs), 0] = blk.real / rs[a0:a0 + len(r)]
                            Dh[d:d + len(r), :len(cs), 1] = blk.imag / rs[a0:a0 + len(r)]
                            del blk
                del Dm
                off += n_own
            oi += ni
        assert off == len(Dh)

    @property
    def bound(self):
        if self._bound is None:
            raise RuntimeError("call bind_ownership() after the first forward")
        return self._bound


@dataclass
class PoolScheduler:
    """Node batching + per-iteration frequency permutation.

    Two node-selection modes:
      * ``random_batch=None`` (default): deterministic pool rotation. Nodes are
        spatially interleaved (sorted by model-y, strided) into ``n_pools``
        pools, so every pool spans the whole array; iteration ``i`` fires pool
        ``i % n_pools``. Guaranteed uniform coverage every iteration.
      * ``random_batch=k``: fire a FRESH random subset of ``k`` nodes each
        iteration (drawn without replacement within the iteration), matching
        the random ±1 path's per-iter resampling. The fixed pools are still
        built (sized ~k) but only for the one-time capture / steady-state QC.

    Either way, frequencies are drawn WITHOUT replacement each iteration — the
    permutation is the method's one mandatory stochastic ingredient (a fixed
    assignment overfits its spectral lines; see the V3 experiment in the notes).
    ``k`` (or the max pool size) must not exceed ``n_bins`` comb frequencies.

    Nodes sharing a grid cell are collapsed to one: see :attr:`dropped_nodes`.
    """

    node_grid: np.ndarray
    n_pools: int
    n_bins: int
    seed: int
    random_batch: int | None = None
    pools: list = field(init=False)
    dropped_nodes: np.ndarray = field(init=False)

    def __post_init__(self):
        # One source per grid cell, always. Two sources in the same cell are
        # injected into the same u[] element, and the CUDA source kernel does
        # that with atomicAdd -- the summation order then varies between runs,
        # so the wavefield and everything downstream stop being reproducible.
        # It is a geometry statement too: at this dh the grid cannot separate
        # the two, so the second node would be modelled from a position it is
        # not at. Keep the lowest-indexed node of each cell and say what went.
        _, first = np.unique(self.node_grid, axis=0, return_index=True)
        keep = np.sort(first)
        self.dropped_nodes = np.setdiff1d(
            np.arange(len(self.node_grid)), keep)
        if len(self.dropped_nodes):
            _d = self.dropped_nodes
            print(f"[freqsel] {len(self.node_grid)} nodes occupy {len(keep)} "
                  f"distinct grid cells; dropping {len(_d)} duplicate(s): "
                  f"{_d[:20].tolist()}{' ...' if len(_d) > 20 else ''}",
                  flush=True)
        self._cand = keep
        order = keep[np.argsort(self.node_grid[keep, 1], kind="stable")]
        npool = self.n_pools
        if self.random_batch:
            # size the fixed interleaved pools to the random batch (used only
            # for the one-time geometry capture / steady-state QC); draw()
            # ignores them and samples a fresh random subset each iteration.
            npool = max(1, -(-len(order) // int(self.random_batch)))
        self.pools = [np.sort(order[p::npool]) for p in range(npool)]
        self.n_pools = npool
        bs = (int(self.random_batch) if self.random_batch
              else max(len(p) for p in self.pools))
        if bs > self.n_bins:
            raise ValueError(
                f"batch size {bs} exceeds {self.n_bins} comb bins; "
                "raise n_pools, lower random_batch, or widen the comb")
        if self.random_batch and int(self.random_batch) > len(keep):
            raise ValueError(
                f"random_batch {self.random_batch} exceeds the {len(keep)} "
                f"distinct source cells ({len(self.node_grid)} nodes collapse "
                f"onto {len(keep)} cells at this dh); lower random_batch or "
                "refine the grid")
        self._n_nodes = len(order)
        self._rng = np.random.default_rng(self.seed)

    def plan(self, n_iters: int):
        """The (pool, bins) the next ``n_iters`` ``draw`` calls WILL return.

        Side-effect free: the RNG state is snapshotted and restored, so calling
        this changes nothing about the sequence ``draw`` then produces. That is
        the whole point -- the schedule is a pure function of (seed, call
        count), so the coefficient table can be pruned to the columns the run
        will actually read before a single iteration has run.
        """
        st = self._rng.bit_generator.state
        try:
            return [self.draw(i) for i in range(int(n_iters))]
        finally:
            self._rng.bit_generator.state = st

    def draw(self, iteration: int):
        if self.random_batch:
            pool = np.sort(self._cand[self._rng.choice(
                self._n_nodes, size=int(self.random_batch), replace=False)])
        else:
            pool = self.pools[iteration % self.n_pools]
        bins = self._rng.permutation(self.n_bins)[:len(pool)]
        # the invariant this class exists to hold up; cheap at these sizes
        assert len(np.unique(self.node_grid[pool], axis=0)) == len(pool), \
            "source cells must be distinct"
        return pool, bins


def encoded_wavelet(comb: FrequencyComb, bins, nt: int, ramp_s: float,
                    device) -> torch.Tensor:
    """(n_src, nt) rows of ``ramp(t) * cos(2 pi f_b t)`` — the mode-B wavelet."""
    t = np.arange(nt, dtype=np.float64) * comb.dt
    ramp = np.where(t < ramp_s, 0.5 * (1.0 - np.cos(np.pi * t / ramp_s)), 1.0)
    w = ramp[None] * np.cos(2 * np.pi * comb.freqs[bins][:, None] * t[None])
    return torch.tensor(w, dtype=torch.float32, device=device)


class _SumAcrossRanks(torch.autograd.Function):
    # y = sum over ranks of x; dy/dx = identity per rank because every rank
    # evaluates the SAME downstream scalar (do NOT use the all_reduce-backward
    # variant here — it would double-count by world_size).
    @staticmethod
    def forward(ctx, x):
        import torch.distributed as dist
        y = x.clone()
        dist.all_reduce(y)
        return y

    @staticmethod
    def backward(ctx, g):
        return g


def _dcol(b, s, bin_k):
    """Local column of node ``s``'s comb bin ``bin_k`` in the bound table.

    Identity when the table is unpruned.  When it is pruned, a -1 means the
    schedule that sized the table never paired this node with this bin, so
    something advanced the RNG differently than ``plan()`` saw -- loud, because
    reading the wrong column would just quietly invert a different dataset.
    """
    cm = b.get("colmap")
    if cm is None:
        return int(bin_k)
    c = int(cm[int(s), int(bin_k)])
    if c < 0:
        raise RuntimeError(
            f"pruned D has no column for (node {int(s)}, bin {int(bin_k)}); "
            "the draw schedule diverged from the one the table was built for. "
            "Set SWEEP_FREQSEL_D_PRUNE=0 to load the full table.")
    return c


class SteadyGCNLoss:
    """Steady-window extraction + per-node masked complex-cosine loss.

    ``__call__(record, pool, bins)`` expects the RAW per-tile record from
    ModelParallel (time innermost) or the canonical single-domain record;
    extracts ``U(f_s, r)`` over the window ``[start, start + n_p)`` with the
    ``2/n_p`` factor and the phase reference to the window start, then
    reduces the per-node GCN across tiles with a differentiable sum.
    Wavelet-free by construction (complex-scale invariance).
    """

    def __init__(self, comb: FrequencyComb, targets: FreqSelTargets,
                 n_ss: int, slack: int, device, distributed: bool,
                 eps: float = 1e-12):
        self.comb, self.targets = comb, targets
        self.start = n_ss + slack
        self.device = device
        self.distributed = distributed
        self.eps = eps

    def _as_rec_time(self, record: torch.Tensor) -> torch.Tensor:
        # ModelParallel returns the RAW tile record, time innermost:
        # (B, nrec, nt) or (nfield, B, nrec, nt). Single-domain PropTorch
        # returns the canonical (B, nt, nrec, nfield). Distinguish by the
        # construction flag, not by shape heuristics.
        if self.distributed:
            return record.reshape(-1, record.shape[-2], record.shape[-1])[0]
        return record[0, :, :, 0].transpose(0, 1)          # (nrec, nt)

    def _probes(self, bins, start):
        m = np.arange(self.comb.n_p, dtype=np.float64)
        ph = np.exp(-2j * np.pi * self.comb.freqs[bins][:, None]
                    * (start * self.comb.dt + m[None] * self.comb.dt)) \
            * (2.0 / self.comb.n_p)
        return (torch.tensor(ph.real, dtype=torch.float32, device=self.device),
                torch.tensor(ph.imag, dtype=torch.float32, device=self.device))

    def __call__(self, record: torch.Tensor, pool, bins, start=None):
        start = self.start if start is None else start
        y = self._as_rec_time(record)
        ywin = y[:, start:start + self.comb.n_p]
        er, ei = self._probes(bins, start)
        b = self.targets.bound
        D = b["D"]
        part = torch.zeros(len(pool), 4, device=self.device)
        for j, s in enumerate(pool):
            gi = b["groups"][s]
            if len(gi) == 0:
                continue
            ds = b["dsel"][s]
            ys = ywin[gi]
            ur = ys @ er[j]
            ui = ys @ ei[j]
            _c = _dcol(b, s, bins[j])
            if b.get("D_on_gpu", True):
                dr = D[ds, _c, 0].float()
                di = D[ds, _c, 1].float()
            else:
                # Host gather, one small transfer per component (69 KB each
                # (tens of KB).  The two components are indexed separately so the
                # result is contiguous, matching the resident path's layout.
                #
                # Bit-identical to the resident path outside the PML.  This
                # comment used to say the opposite -- that the gradient simply
                # was not reproducible, two runs of one binary differing by
                # 1.3e-5 over 99.96% of cells.  That was duplicate source cells
                # colliding in the atomicAdd source kernel, since fixed; see
                # PoolScheduler.  What is left sits in the absorbing boundary.
                dr = D[ds, _c, 0].to(self.device).float()
                di = D[ds, _c, 1].to(self.device).float()
            part[j, 0] = (ur * dr + ui * di).sum()
            part[j, 1] = (ui * dr - ur * di).sum()
            part[j, 2] = (ur ** 2 + ui ** 2).sum()
            part[j, 3] = (dr ** 2 + di ** 2).sum()
        tot = _SumAcrossRanks.apply(part) if self.distributed else part
        num = torch.sqrt(tot[:, 0] ** 2 + tot[:, 1] ** 2 + self.eps)
        den = torch.sqrt(tot[:, 2] + self.eps) * torch.sqrt(tot[:, 3] + self.eps)
        return (1.0 - num / den).sum(), len(pool)

    @torch.no_grad()
    def two_window_check(self, record: torch.Tensor, pool, bins,
                         n_ss: int, slack: int) -> float:
        """Steady-state QC without a true model: extract U on two windows
        offset by ``slack`` samples; their median relative difference is the
        transient-residual level (target ~<= 1e-2)."""
        y = self._as_rec_time(record)
        b = self.targets.bound
        us = {}
        diffs = []
        for tag, st in (("late", n_ss + slack), ("early", n_ss)):
            er, ei = self._probes(bins, st)
            for j, s in enumerate(pool):
                gi = b["groups"][s]
                if len(gi) == 0:
                    continue
                ys = y[gi][:, st:st + self.comb.n_p]
                u = torch.complex(ys @ er[j], ys @ ei[j])
                if tag == "late":
                    us[j] = u
                else:
                    d = (u - us[j]).abs() / (us[j].abs() + 1e-20)
                    diffs.append(float(d.median()))
        return float(np.median(diffs)) if diffs else float("nan")
