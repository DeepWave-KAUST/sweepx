"""Fused Triton kernels for the anisotropic multi-resolution hash encoding.

Optional GPU-only acceleration backend for :class:`~sweep_nn.hash_encoding.MultiResHashGrid`
(and thus :class:`~sweep_nn.coarse_to_fine.CoarseToFineHashGrid`). Numerically matches the
pure-PyTorch path — forward and ``dL/dlatent`` agree with the reference at ``cos = 1.0``
(fp32 reduction-order differences ~1e-6). The win is memory: the fused kernel never
materializes the ``(L, N, 2**dim, F)`` gathered-latent / vertex intermediates the PyTorch
path builds, so encoder peak memory drops ~7-20x at large batch, and forward+backward is
~1.6-2x faster (encoder alone).

The kernels are hand-unrolled over ``dim`` in {2, 3} — Triton's compile-time container
support is narrow (no ``list.append`` / comprehensions / ``tuple()``), so explicit per-axis
variables are the robust choice. Per-axis ``scale``/``resolution`` make this anisotropy-native:
different per-axis growth factors cost nothing extra (same code path, different ``scale`` values).

``triton`` is imported lazily; ``import sweep_nn`` works without it. Only constructing an
encoder with ``backend="triton"`` requires triton + a CUDA device.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton is an optional dependency
    _HAVE_TRITON = False


def have_triton() -> bool:
    """True if the Triton backend can be used (triton importable)."""
    return _HAVE_TRITON


if _HAVE_TRITON:
    _MASK = tl.constexpr(0xFFFFFFFF)
    _P1 = tl.constexpr(2654435761)   # Knuth prime for dim 1 (matches sweep-nn primes[1])
    _P2 = tl.constexpr(805459861)    # prime for dim 2 (matches sweep-nn primes[2])

    @triton.jit
    def _hash_fwd_kernel(
        pos_ptr, lat_ptr, scale_ptr, stride_ptr, offset_ptr, dense_ptr, out_ptr,
        N, T, table_size, LF,
        D: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        lvl = tl.program_id(1)
        offs = pid_n * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        dense = tl.load(dense_ptr + lvl)
        off_l = tl.load(offset_ptr + lvl)

        sc0 = tl.load(scale_ptr + lvl * D + 0); st0 = tl.load(stride_ptr + lvl * D + 0)
        ps0 = tl.load(pos_ptr + offs * D + 0, mask=m, other=0.0) * sc0
        fl0 = tl.floor(ps0); fr0 = ps0 - fl0
        sc1 = tl.load(scale_ptr + lvl * D + 1); st1 = tl.load(stride_ptr + lvl * D + 1)
        ps1 = tl.load(pos_ptr + offs * D + 1, mask=m, other=0.0) * sc1
        fl1 = tl.floor(ps1); fr1 = ps1 - fl1
        if D >= 3:
            sc2 = tl.load(scale_ptr + lvl * D + 2); st2 = tl.load(stride_ptr + lvl * D + 2)
            ps2 = tl.load(pos_ptr + offs * D + 2, mask=m, other=0.0) * sc2
            fl2 = tl.floor(ps2); fr2 = ps2 - fl2

        acc = tl.zeros((BLOCK, F), dtype=tl.float32)
        fcols = tl.arange(0, F)

        for corner in range(1 << D):
            c0 = (corner >> (D - 1 - 0)) & 1
            c1 = (corner >> (D - 1 - 1)) & 1
            v0 = (fl0 + c0).to(tl.int64); v1 = (fl1 + c1).to(tl.int64)
            w0 = fr0 if c0 == 1 else (1.0 - fr0)
            w1 = fr1 if c1 == 1 else (1.0 - fr1)
            w = tl.minimum(tl.maximum(w0, 0.0), 1.0) * tl.minimum(tl.maximum(w1, 0.0), 1.0)
            idx_tiled = v0 * st0 + v1 * st1
            h = (v0 & _MASK) ^ ((v1 * _P1) & _MASK)
            if D >= 3:
                c2 = (corner >> (D - 1 - 2)) & 1
                v2 = (fl2 + c2).to(tl.int64)
                w2 = fr2 if c2 == 1 else (1.0 - fr2)
                w = w * tl.minimum(tl.maximum(w2, 0.0), 1.0)
                idx_tiled += v2 * st2
                h = h ^ ((v2 * _P2) & _MASK)
            h = h & _MASK

            idx_local = tl.where(dense != 0, idx_tiled, h)
            idx = (idx_local % T) + off_l
            idx = tl.minimum(tl.maximum(idx, 0), table_size - 1)
            lat = tl.load(lat_ptr + idx[:, None] * F + fcols[None, :], mask=m[:, None], other=0.0)
            acc += w[:, None] * lat

        tl.store(out_ptr + offs[:, None] * LF + (lvl * F + fcols)[None, :], acc, mask=m[:, None])

    @triton.jit
    def _hash_bwd_kernel(
        pos_ptr, gout_ptr, scale_ptr, stride_ptr, offset_ptr, dense_ptr, glat_ptr,
        N, T, table_size, LF,
        D: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        lvl = tl.program_id(1)
        offs = pid_n * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        dense = tl.load(dense_ptr + lvl)
        off_l = tl.load(offset_ptr + lvl)

        sc0 = tl.load(scale_ptr + lvl * D + 0); st0 = tl.load(stride_ptr + lvl * D + 0)
        ps0 = tl.load(pos_ptr + offs * D + 0, mask=m, other=0.0) * sc0
        fl0 = tl.floor(ps0); fr0 = ps0 - fl0
        sc1 = tl.load(scale_ptr + lvl * D + 1); st1 = tl.load(stride_ptr + lvl * D + 1)
        ps1 = tl.load(pos_ptr + offs * D + 1, mask=m, other=0.0) * sc1
        fl1 = tl.floor(ps1); fr1 = ps1 - fl1
        if D >= 3:
            sc2 = tl.load(scale_ptr + lvl * D + 2); st2 = tl.load(stride_ptr + lvl * D + 2)
            ps2 = tl.load(pos_ptr + offs * D + 2, mask=m, other=0.0) * sc2
            fl2 = tl.floor(ps2); fr2 = ps2 - fl2

        fcols = tl.arange(0, F)
        genc = tl.load(gout_ptr + offs[:, None] * LF + (lvl * F + fcols)[None, :], mask=m[:, None], other=0.0)

        for corner in range(1 << D):
            c0 = (corner >> (D - 1 - 0)) & 1
            c1 = (corner >> (D - 1 - 1)) & 1
            v0 = (fl0 + c0).to(tl.int64); v1 = (fl1 + c1).to(tl.int64)
            w0 = fr0 if c0 == 1 else (1.0 - fr0)
            w1 = fr1 if c1 == 1 else (1.0 - fr1)
            w = tl.minimum(tl.maximum(w0, 0.0), 1.0) * tl.minimum(tl.maximum(w1, 0.0), 1.0)
            idx_tiled = v0 * st0 + v1 * st1
            h = (v0 & _MASK) ^ ((v1 * _P1) & _MASK)
            if D >= 3:
                c2 = (corner >> (D - 1 - 2)) & 1
                v2 = (fl2 + c2).to(tl.int64)
                w2 = fr2 if c2 == 1 else (1.0 - fr2)
                w = w * tl.minimum(tl.maximum(w2, 0.0), 1.0)
                idx_tiled += v2 * st2
                h = h ^ ((v2 * _P2) & _MASK)
            h = h & _MASK

            idx_local = tl.where(dense != 0, idx_tiled, h)
            idx = (idx_local % T) + off_l
            idx = tl.minimum(tl.maximum(idx, 0), table_size - 1)
            tl.atomic_add(glat_ptr + idx[:, None] * F + fcols[None, :], w[:, None] * genc, mask=m[:, None])

    class _TritonHashFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pos, latents, scales, strides, offsets, is_dense, T, D, F, BLOCK):
            N = pos.shape[0]
            L = scales.shape[0]
            table_size = latents.shape[0]
            out = torch.empty((N, L * F), device=pos.device, dtype=torch.float32)
            grid = (triton.cdiv(N, BLOCK), L)
            _hash_fwd_kernel[grid](
                pos, latents, scales, strides, offsets, is_dense, out,
                N, T, table_size, L * F, D=D, F=F, BLOCK=BLOCK,
            )
            ctx.save_for_backward(pos, scales, strides, offsets, is_dense)
            ctx.T, ctx.D, ctx.F, ctx.BLOCK = T, D, F, BLOCK
            ctx.L, ctx.table_size = L, table_size
            return out

        @staticmethod
        def backward(ctx, grad_out):
            pos, scales, strides, offsets, is_dense = ctx.saved_tensors
            N = pos.shape[0]
            glat = torch.zeros((ctx.table_size, ctx.F), device=pos.device, dtype=torch.float32)
            grid = (triton.cdiv(N, ctx.BLOCK), ctx.L)
            _hash_bwd_kernel[grid](
                pos, grad_out.contiguous(), scales, strides, offsets, is_dense, glat,
                N, ctx.T, ctx.table_size, ctx.L * ctx.F,
                D=ctx.D, F=ctx.F, BLOCK=ctx.BLOCK,
            )
            return None, glat, None, None, None, None, None, None, None, None


def triton_encode(pos, latents, scales, strides, offsets, is_dense, T, D, F, BLOCK=256):
    """Fused Triton multi-resolution hash encoding. Returns ``(N, L*F)`` level-major.

    Parameters
    ----------
    pos : (N, D) float32, normalized coords in ``[0, 1)``.
    latents : (table_size, F) float32 Parameter — the flat latent table.
    scales : (L, D) float32 — per-level, per-axis ``scale``.
    strides : (L, D) int64 — per-level, per-axis dense stride.
    offsets : (L,) int64 — per-level base offset into ``latents`` (i.e. ``offsets[:-1]``).
    is_dense : (L,) int32 — 1 for dense (tiled) levels, 0 for hashed.
    T : int — global hashmap size ``2**log2_hashmap_size``.
    D, F : int — spatial dims (2 or 3) and features per level.
    """
    if not _HAVE_TRITON:
        raise RuntimeError(
            "backend='triton' requires the `triton` package (and a CUDA device). "
            "Install triton or use backend='pytorch'."
        )
    if not pos.is_cuda:
        raise RuntimeError("The Triton hash backend requires CUDA tensors.")
    return _TritonHashFn.apply(
        pos.contiguous(), latents, scales.contiguous(), strides.contiguous(),
        offsets.contiguous(), is_dense.contiguous(), int(T), int(D), int(F), int(BLOCK),
    )


__all__ = ["triton_encode", "have_triton"]
