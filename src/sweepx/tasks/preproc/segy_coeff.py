"""Extract frequency-comb coefficient shards straight from SEG-Y.

:func:`sweep_tasks.freqsel.extract_shard_gathers` takes gathers a caller has
already assembled; a :class:`SeismicPlan` supplies them when the survey has
been through the plan machinery. Raw field SEG-Y has neither: the traces of one
common-node gather are scattered across every shot file, so the only way to
build the gathers is to sweep the whole survey once and bin as you go.

That inverts the loop -- files outside, nodes inside -- and with it the memory
question. This module does the sweep and writes the same shard schema:

    files -> headers -> in-box filter -> decode -> mute -> DTFT -> bin by
    (node cell, trace cell) -> average -> streamed npz

Everything survey-specific is a parameter. The module knows nothing about any
particular acquisition: header byte offsets, the coordinate scalar, the grid,
the model transform, the mute law and the comb all arrive from the caller.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from ..freqsel import FrequencyComb, write_npz_streamed

__all__ = ["SegyLayout", "BoxGrid", "TopMute", "extract_coeff_from_segy",
           "decode_ibm32", "source_bbox_manifest", "filter_by_manifest",
           "node_part"]


@dataclass(frozen=True)
class SegyLayout:
    """Where the numbers live in a SEG-Y file.

    Defaults are the SEG-Y rev1 trace-header positions (1-based in the
    standard, 0-based here). ``coord_scale`` is applied as a divisor, which is
    what a header scalar of -100 means; pass 1.0 for coordinates already in
    metres.
    """
    text_header_bytes: int = 3600
    trace_header_bytes: int = 240
    sx: int = 72
    sy: int = 76
    gx: int = 80
    gy: int = 84
    coord_scale: float = 100.0
    sample_format: str = "ibm32"        # only IBM float32 for now

    def trace_bytes(self, nt: int) -> int:
        return self.trace_header_bytes + 4 * int(nt)


@dataclass(frozen=True)
class BoxGrid:
    """The aggregation box, in model coordinates.

    A trace is kept when its SOURCE cell lands in ``[0, nx) x [0, ny)``; cells
    are ``rint((xy - origin) / dh)``, the same rounding the inversion uses to
    place a source, so a trace is binned exactly where it will be fired.
    """
    origin_x: float
    origin_y: float
    dh: float
    nx: int
    ny: int

    def cells(self, x: np.ndarray, y: np.ndarray):
        cx = np.rint((np.asarray(x) - self.origin_x) / self.dh).astype(np.int64)
        cy = np.rint((np.asarray(y) - self.origin_y) / self.dh).astype(np.int64)
        return cx, cy

    def inside(self, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        return (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)

    def accept_window(self, margin: float = 0.0):
        """Physical window a source can fall in, given ``rint``'s half cell."""
        return (self.origin_x - 0.5 * self.dh - margin,
                self.origin_x + (self.nx - 0.5) * self.dh + margin,
                self.origin_y - 0.5 * self.dh - margin,
                self.origin_y + (self.ny - 0.5) * self.dh + margin)


@dataclass(frozen=True)
class TopMute:
    """Offset-dependent top mute, cosine-tapered.

    ``t_cut = t0 + offset / v + a * sqrt(offset) - guard``; samples before it
    are zeroed and the following ``taper`` seconds ramp up as a raised cosine.
    The ``a * sqrt`` term is there because a straight line in (t, x) does not
    follow a refraction whose velocity grows with depth; ``guard`` pulls the
    whole curve up so the mute never eats the event it is protecting.
    """
    t0: float = 0.0
    v: float = 1500.0
    a: float = 0.0
    guard: float = 0.0
    taper: float = 0.0

    def cut_time(self, offset: np.ndarray) -> np.ndarray:
        off = np.maximum(np.asarray(offset, np.float64), 0.0)
        return self.t0 + off / self.v + self.a * np.sqrt(off) - self.guard


def decode_ibm32(raw: np.ndarray) -> np.ndarray:
    """IBM 360 float32 (big-endian) -> IEEE float32.

    ``raw`` is ``(n, nt, 4)`` uint8. The exponent is base 16 biased by 64 and
    the mantissa is a plain fraction, so the value is
    ``sign * frac/2^24 * 16^(exp-64)``. Done in int64 because the shift
    assembly overflows int32 for the top byte.
    """
    b = np.asarray(raw, np.uint8).astype(np.int64)
    u = (b[..., 0] << 24) | (b[..., 1] << 16) | (b[..., 2] << 8) | b[..., 3]
    sign = (u >> 31) & 1
    exp = (u >> 24) & 0x7F
    frac = (u & 0x00FFFFFF).astype(np.float64)
    v = frac / 2.0 ** 24 * np.power(16.0, (exp - 64).astype(np.float64))
    return np.where(sign == 1, -v, v).astype(np.float32)


def _be_i32(raw: np.ndarray, off: int) -> np.ndarray:
    b = raw[:, off:off + 4].astype(np.uint32)
    u = ((b[:, 0].astype(np.int64) << 24) | (b[:, 1].astype(np.int64) << 16)
         | (b[:, 2].astype(np.int64) << 8) | b[:, 3].astype(np.int64))
    return np.where(u >= 2 ** 31, u - 2 ** 32, u)


def _read_file(path: str, layout: SegyLayout, nt: int):
    """Whole file into memory, headers decoded, samples left as bytes.

    ``buf[hdr:]`` would slice the bytes object and COPY it -- a second full
    copy of every file, and twice the transient footprint per worker.
    ``frombuffer`` takes the offset directly and is zero-copy.
    """
    tb = layout.trace_bytes(nt)
    size = os.path.getsize(path)
    ntr = (size - layout.text_header_bytes) // tb
    if ntr < 1:
        return None
    with open(path, "rb") as fh:
        buf = fh.read()
    raw = np.frombuffer(buf, dtype=np.uint8, count=ntr * tb,
                        offset=layout.text_header_bytes).reshape(ntr, tb)
    sc = float(layout.coord_scale)
    sx = _be_i32(raw, layout.sx) / sc
    sy = _be_i32(raw, layout.sy) / sc
    gx = _be_i32(raw, layout.gx) / sc
    gy = _be_i32(raw, layout.gy) / sc
    return raw, sx, sy, gx, gy


def source_bbox_manifest(paths, layout: SegyLayout, nt: int, to_model=None,
                         *, nproc: int = 8, out_path: str | None = None) -> dict:
    """Per-file bounding box of the SOURCE positions, in model coordinates.

    A sweep reads every file even though a survey's shot lines usually reach
    well past any one inversion box, so most files contribute nothing. The
    bbox is band- and shard-independent -- it is a property of the file, not of
    the grid -- so one manifest serves every later pass, and a box test on it
    can only over-include. Over-inclusion costs time; exclusion would lose
    data silently, which is why the test is a bbox and not a sampled subset.
    """
    def one(p):
        r = _read_file(p, layout, nt)
        if r is None:
            return os.path.basename(p), None
        _, sx, sy, _, _ = r
        xy = np.stack([sx, sy], 1)
        if to_model is not None:
            xy = np.asarray(to_model(xy), np.float64)
        return os.path.basename(p), [float(xy[:, 0].min()), float(xy[:, 0].max()),
                                     float(xy[:, 1].min()), float(xy[:, 1].max())]
    man = {}
    with ThreadPoolExecutor(max(1, int(nproc))) as ex:
        for name, bb in ex.map(one, list(paths)):
            man[name] = bb
    if out_path:
        tmp = str(out_path) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(man, fh)
        os.replace(tmp, out_path)
    return man


def filter_by_manifest(paths, manifest: dict, box: BoxGrid, margin: float = 0.0):
    """Drop files whose source bbox cannot touch the box.

    Files missing from the manifest are KEPT: a stale or partial manifest must
    cost speed, never data.
    """
    x0, x1, y0, y1 = box.accept_window(margin)
    out = []
    for p in paths:
        b = manifest.get(os.path.basename(p))
        if b is None or not (b[1] < x0 or b[0] > x1 or b[3] < y0 or b[2] > y1):
            out.append(p)
    return out


def _node_uid(gx: np.ndarray, gy: np.ndarray, quantum: float) -> np.ndarray:
    """Stable integer id for a receiver position.

    Nodes are identified by their own coordinates rather than by a header
    field, because a survey's node numbering is not portable and a redrop
    reuses ids. ``quantum`` is the rounding applied before packing, in the
    coordinate unit: it must be fine enough to keep two real nodes apart and
    coarse enough to absorb header jitter for one node across files.
    """
    qx = np.rint(np.asarray(gx, np.float64) / quantum).astype(np.int64)
    qy = np.rint(np.asarray(gy, np.float64) / quantum).astype(np.int64)
    return (qx << 32) | (qy & 0xFFFFFFFF)


def node_part(uid: np.ndarray, npart: int) -> np.ndarray:
    """Which shard a node belongs to. Hashed, not ``uid % npart``.

    The uid packs two quantized coordinates, so its low bits are one
    coordinate's -- and a survey lays its nodes on a grid. Taking the modulus
    of the raw key then partitions by that coordinate's parity: on a line of
    nodes sharing a y, every uid has the same low half and one shard gets
    everything. Mix the whole key first (murmur3 fmix64) so the split is
    balanced whatever the geometry, and still a pure function of the uid, so
    the same node lands in the same shard on every pass.
    """
    x = np.asarray(uid).astype(np.uint64)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xC4CEB9FE1A85EC53)
    x ^= x >> np.uint64(33)
    return (x % np.uint64(int(npart))).astype(np.int64)


def _dtft_numpy(traces: np.ndarray, E: np.ndarray) -> np.ndarray:
    return (traces.astype(np.float64) @ E.T).astype(np.complex64)


def _process_file(path, *, layout, box, comb_E, nt, layout_nt, mute, dt_record,
                  to_model, node_quantum, part, npart, gpu_chunk, decode):
    """One SEG-Y file -> the coefficients of its in-box traces.

    The in-box filter runs on the HEADERS, before any decode or transform, so a
    file that contributes nothing costs one read and no arithmetic. The node
    shard filter runs next, for the same reason: with ``npart > 1`` each pass
    keeps a disjoint node subset, and a trace the pass will not keep must not
    reach the DTFT.
    """
    r = _read_file(path, layout, layout_nt)
    if r is None:
        return None
    raw, sx, sy, gx, gy = r
    src = np.stack([sx, sy], 1)
    if to_model is not None:
        src = np.asarray(to_model(src), np.float64)
    cx, cy = box.cells(src[:, 0], src[:, 1])
    keep = box.inside(cx, cy)
    bbox = [float(src[:, 0].min()), float(src[:, 0].max()),
            float(src[:, 1].min()), float(src[:, 1].max())]
    if not keep.any():
        return (None, bbox, os.path.basename(path))
    uid = _node_uid(gx, gy, node_quantum)
    if npart > 1:
        keep &= node_part(uid, npart) == part
        if not keep.any():
            return (None, bbox, os.path.basename(path))
    idx = np.nonzero(keep)[0]
    cell = cx[idx] * box.ny + cy[idx]
    offset = np.hypot(sx[idx] - gx[idx], sy[idx] - gy[idx])
    hdr, nsamp = layout.trace_header_bytes, layout_nt
    out = np.empty((len(idx), comb_E.shape[0]), np.complex64)
    for a in range(0, len(idx), gpu_chunk):
        ic = idx[a:a + gpu_chunk]
        block = raw[ic, hdr:hdr + 4 * nsamp].reshape(len(ic), nsamp, 4)
        tr = decode(block)
        if mute is not None and mute.taper > 0.0:
            t_ax = np.arange(nsamp, dtype=np.float64) * dt_record
            cut = mute.cut_time(offset[a:a + gpu_chunk])
            ramp = np.clip((t_ax[None, :] - cut[:, None]) / mute.taper, 0.0, 1.0)
            tr = tr * (0.5 - 0.5 * np.cos(np.pi * ramp))
        out[a:a + gpu_chunk] = _dtft_numpy(tr, comb_E)
    return ((uid[idx], cell, gx[idx], gy[idx], out), bbox,
            os.path.basename(path))


def extract_coeff_from_segy(out_path: str, segy_paths, *, layout: SegyLayout,
                            box: BoxGrid, comb: FrequencyComb, nt: int,
                            dt_record: float | None = None, to_model=None,
                            mute: TopMute | None = None,
                            node_quantum: float = 0.1,
                            part: int = 0, npart: int = 1,
                            nproc: int = 8, gpu_chunk: int = 4096,
                            manifest: dict | None = None,
                            manifest_margin: float = 0.0,
                            meta_extra: dict | None = None,
                            verbose: bool = False) -> str:
    """Sweep a SEG-Y survey once and write one coefficient shard.

    ``npart`` shards the NODES, not the files: every pass still reads every
    file it was given, because a node's traces are spread across all of them.
    Splitting the nodes halves the coefficient table, which is what decides
    whether the sweep fits in host RAM; the cost is re-reading the survey once
    per part, so use ``npart=1`` whenever the table fits.

    Pass ``manifest`` (from :func:`source_bbox_manifest`) to skip files whose
    sources cannot reach ``box`` -- see :func:`filter_by_manifest` for why that
    is safe.
    """
    from ..freqsel import _comb_kernel   # local: keeps the import graph flat

    paths = sorted(segy_paths)
    if manifest:
        paths = filter_by_manifest(paths, manifest, box, manifest_margin)
        if verbose:
            print(f"[segy-coeff] manifest kept {len(paths)} files", flush=True)
    if not paths:
        raise ValueError("no SEG-Y files to read")
    dtr = float(comb.dt if dt_record is None else dt_record)
    E = np.asarray(_comb_kernel(comb, int(nt), dtr))          # (n_bins, nt)
    n_bins = E.shape[0]

    # Exact upper bound on rows: a file cannot yield more traces than it holds.
    # np.empty does not touch pages, so the unused tail of the bound costs no
    # resident memory -- which is what lets the destination be allocated up
    # front instead of concatenating the parts afterwards (that peaks at 2x).
    tb = layout.trace_bytes(int(nt))
    n_max = int(sum(max(0, (os.path.getsize(p) - layout.text_header_bytes) // tb)
                    for p in paths))
    D = np.empty((n_max, n_bins), np.complex64)
    if verbose:
        print(f"[segy-coeff] {len(paths)} files, row bound {n_max} "
              f"({n_max * n_bins * 8 / 2**30:.1f} GiB virtual)", flush=True)

    work = dict(layout=layout, box=box, comb_E=E, nt=nt, layout_nt=int(nt),
                mute=mute, dt_record=dtr, to_model=to_model,
                node_quantum=float(node_quantum), part=int(part),
                npart=int(npart), gpu_chunk=int(gpu_chunk),
                decode=decode_ibm32)

    uid_parts, cell_parts, gx_parts, gy_parts = [], [], [], []
    bboxes, off, done = {}, 0, 0
    copier = ThreadPoolExecutor(max(1, int(nproc)))
    futures = []

    def _stash(dst0, block):
        D[dst0:dst0 + len(block)] = block

    with ThreadPoolExecutor(max(1, int(nproc))) as ex:
        for res in ex.map(lambda p: _process_file(p, **work), paths):
            done += 1
            if res is None:
                continue
            payload, bbox, name = res
            bboxes[name] = bbox
            if payload is None:
                continue
            uid, cell, gxk, gyk, block = payload
            if off + len(block) > n_max:
                raise RuntimeError("row bound too small; trace count is wrong")
            uid_parts.append(uid); cell_parts.append(cell)
            gx_parts.append(gxk); gy_parts.append(gyk)
            futures.append(copier.submit(_stash, off, block))
            off += len(block)
            # The executor queue is unbounded: without a cap the undrained
            # blocks pile up in it and the peak this bound exists to remove
            # comes straight back.
            if len(futures) > 32:
                for f in futures[:-16]:
                    f.result()
                futures = futures[-16:]
            if verbose and done % 50 == 0:
                print(f"[segy-coeff] {done}/{len(paths)} files, {off} traces",
                      flush=True)
    for f in futures:
        f.result()
    copier.shutdown()
    if off == 0:
        raise ValueError("no trace fell inside the box")
    D = D[:off]

    nuid = np.concatenate(uid_parts); cell = np.concatenate(cell_parts)
    gxa = np.concatenate(gx_parts); gya = np.concatenate(gy_parts)
    del uid_parts, cell_parts, gx_parts, gy_parts
    uid_u, first = np.unique(nuid, return_index=True)
    node_of = np.searchsorted(uid_u, nuid)
    n_nodes = len(uid_u)

    gkey = node_of.astype(np.int64) * (box.nx * box.ny) + cell
    order = np.argsort(gkey, kind="stable")
    gs = gkey[order]
    bnd = np.concatenate([[0], np.nonzero(np.diff(gs))[0] + 1, [len(gs)]])
    uk = gs[bnd[:-1]]
    fold = np.diff(bnd).astype(np.int64)
    return _write_folded_shard(out_path, D, order, bnd, uk, fold, box, comb,
                               n_nodes, gxa[first], gya[first], nt, dtr,
                               meta_extra, verbose)


def _folded_blocks(D, order, bnd, fold, chunk, threads, buffers, n_bins):
    """Yield the folded table one block of groups at a time, in order.

    Three things happen per block and all three touch the same rows, so they
    are done together while those rows are still in cache:

    * gather the block's rows through ``order``;
    * sum each group -- except when every group holds exactly one row, where
      ``reduceat`` over unit segments performs no additions at all and the
      gather already IS the result, so it goes straight into the destination;
    * divide by the fold.

    The divide is NOT skipped in the unit case. Complex division by ``1+0j``
    computes ``b - a*0``, which is ``nan`` when ``a`` is inf, so "fold == 1
    means the divide is a no-op" is false for a non-finite sample and skipping
    it would silently change those rows.

    A sliding window, not waves: the writer consumes block j while the pool is
    already building j+1..j+k, so the segment costs max(compute, write) rather
    than their sum.
    """
    from collections import deque

    n_groups = len(bnd) - 1
    unit = bool(fold.max() == 1)
    starts = list(range(0, n_groups, chunk))
    free = deque(np.empty((chunk, n_bins), np.complex64) for _ in range(buffers))
    flight, nxt = deque(), 0

    def build(j, buf):
        g0 = starts[j]; g1 = min(g0 + chunk, n_groups)
        r0, r1 = int(bnd[g0]), int(bnd[g1])
        out = buf[:g1 - g0]
        if unit:
            np.take(D, order[r0:r1], axis=0, out=out)
        else:
            out[:] = np.add.reduceat(D[order[r0:r1]], bnd[g0:g1] - r0, axis=0)
        np.divide(out, fold[g0:g1, None].astype(np.float64), out=out,
                  casting="unsafe")
        return out

    with ThreadPoolExecutor(max(1, int(threads))) as ex:
        def pump():
            nonlocal nxt
            while free and nxt < len(starts):
                flight.append((ex.submit(build, nxt, free.popleft())))
                nxt += 1
        pump()
        while flight:
            block = flight.popleft().result()
            base = block.base if block.base is not None else block
            yield block
            free.append(base)
            pump()


def _write_folded_shard(out_path, D, order, bnd, uk, fold, box, comb, n_nodes,
                        node_gx, node_gy, nt, dt_record, meta_extra, verbose,
                        chunk: int = 25000, threads: int = 16,
                        buffers: int = 24):
    ncell = box.nx * box.ny
    item_node = (uk // ncell).astype(np.int64)
    item_cell = (uk % ncell).astype(np.int64)
    # ``uk`` is ascending, so item_node is non-decreasing and the CSR pointer
    # can be built by counting; assert rather than assume, because a reorder
    # here would silently mis-associate every node's rows.
    if not np.all(np.diff(item_node) >= 0):
        raise RuntimeError("group keys not sorted by node")
    ptr = np.zeros(n_nodes + 1, np.int64)
    np.add.at(ptr, item_node + 1, 1)
    ptr = np.cumsum(ptr)
    trace_xyz = np.stack([item_cell // box.ny, item_cell % box.ny,
                          np.zeros(len(item_cell), np.int64)], -1)
    node_cx, node_cy = box.cells(node_gx, node_gy)
    node_xyz = np.stack([node_cx, node_cy, np.zeros(n_nodes, np.int64)], -1)
    if verbose:
        print(f"[segy-coeff] {n_nodes} nodes, {len(uk)} cells, "
              f"fold mean {fold.mean():.3f} max {int(fold.max())}", flush=True)
    return write_npz_streamed(
        out_path,
        stream_name="D",
        stream_parts=_folded_blocks(D, order, bnd, fold, chunk, threads,
                                    buffers, int(comb.n_bins)),
        stream_shape=(len(uk), int(comb.n_bins)), stream_dtype=np.complex64,
        node_ids=np.arange(n_nodes, dtype=np.int32),
        node_grid_xyz=node_xyz.astype(np.int32),
        node_ptr=ptr,
        fold=fold.astype(np.int32),
        trace_grid_xyz=trace_xyz.astype(np.int32),
        freqs=comb.freqs, ks=comb.ks, qc_freqs=np.zeros(0),
        meta=json.dumps(dict(n_p=comb.n_p, dt_solver=comb.dt, synthetic=False,
                             nt_record=int(nt), dt_record=float(dt_record),
                             dh_agg=float(box.dh),
                             origin=[float(box.origin_x), float(box.origin_y)],
                             source="segy_sweep", **(meta_extra or {}))))
