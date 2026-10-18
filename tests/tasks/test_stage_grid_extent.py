"""A stage grid must sample the SAME physical model as the native grid.

A stage that sets ``dh_m`` used to get ``round(n * dh_native / dh)`` points
with the native extent stretched onto them end to end, while the solver
stepped with the nominal ``dh`` and the geometry snapped with
``round(x / dh)``: every coarse stage inverted a model compressed against its
own data and acquisition (Marmousi at 100 m: 2.9 % in depth). These tests pin
the replacement in ``sweepx.tasks._helpers.stage_grid``: the extent, exact
values at coincident samples, no resampling artefacts, the edge receiver, the
frozen layer, the INR frame, and -- end to end -- the traveltimes a coarse
stage actually models, for conventional FWI and for the freqsel band grids.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sweepx.tasks._helpers.stage_grid import (
    _resample_to_grid,
    _seabed_to_grid,
    _set_physical_inr_frame,
    _snap_covered,
    _stage_grid_shape,
)

# (native shape, native dh, stage dhs): the two benchmark ladders the bug
# was measured on (axis order is the model's: z first).
MARMOUSI = ((281, 1361), 12.5, (25.0, 37.5, 50.0, 100.0))
OVERTHRUST = ((187, 801, 801), 25.0, (50.0, 100.0))


def _legacy_shape(native_shape, native_dh, dh):
    """The removed ``_shape_for_dh``: round(n * ratio) points."""
    if abs(native_dh - dh) < 1e-12:
        return tuple(native_shape)
    return tuple(max(1, int(round(s * native_dh / dh))) for s in native_shape)


def _legacy_resample(t, src_dh, dst_shape, dst_dh):
    """The removed ``_resample_vp_tensor``: ends aligned, spacing ignored."""
    x = t.detach() if isinstance(t, torch.Tensor) else torch.from_numpy(np.asarray(t))
    mode = "bilinear" if x.ndim == 2 else "trilinear"
    out = torch.nn.functional.interpolate(
        x[None, None], size=tuple(dst_shape), mode=mode, align_corners=True)[0, 0]
    return out if isinstance(t, torch.Tensor) else out.numpy()


# ---------------------------------------------------------------- extent --

@pytest.mark.parametrize("native_shape,native_dh,dhs", [MARMOUSI, OVERTHRUST])
def test_stage_grid_spans_the_native_extent(native_shape, native_dh, dhs):
    for dh in dhs:
        shape = _stage_grid_shape(native_shape, native_dh, dh)
        for n0, n in zip(native_shape, shape):
            native_ext = (n0 - 1) * native_dh
            stage_ext = (n - 1) * dh
            # covers the native extent, overshooting by less than one cell
            assert native_ext <= stage_ext < native_ext + dh, (dh, n0, n)
        # the bug this replaces lost up to 3 % of the extent
        legacy = _legacy_shape(native_shape, native_dh, dh)
        worst = max(((n0 - 1) * native_dh - (n - 1) * dh) / ((n0 - 1) * native_dh)
                    for n0, n in zip(native_shape, legacy))
        assert worst > 0.0, f"legacy grid at {dh} m was not short: {legacy}"
    assert _stage_grid_shape(native_shape, native_dh, native_dh) == tuple(native_shape)


def test_stage_grid_numbers_for_the_report():
    """Pin the concrete shapes (old vs new) quoted for the two ladders."""
    got = {dh: _stage_grid_shape(*MARMOUSI[:2], dh) for dh in MARMOUSI[2]}
    assert got == {25.0: (141, 681), 37.5: (95, 455), 50.0: (71, 341),
                   100.0: (36, 171)}
    got = {dh: _stage_grid_shape(*OVERTHRUST[:2], dh) for dh in OVERTHRUST[2]}
    assert got == {50.0: (94, 401, 401), 100.0: (48, 201, 201)}
    # what the old code built (z extent short by 1.4 % / 2.9 % at 50 / 100 m)
    assert _legacy_shape(*MARMOUSI[:2], 100.0) == (35, 170)
    assert _legacy_shape(*OVERTHRUST[:2], 100.0) == (47, 200, 200)


def test_stage_grid_is_float_noise_proof():
    # 800 * 25 / 100 is exactly 200 cells; float noise must not add one
    assert _stage_grid_shape((801,), 25.0, 100.0) == (201,)
    assert _stage_grid_shape((31,), 0.1, 0.3) == (11,)        # 0.3/0.1 != 3.0
    # a dh_m written to a few decimals: 600.0006 cells is 600, not 601
    assert _stage_grid_shape((187, 801), 25.0, 33.3333) == (141, 601)
    assert _stage_grid_shape((187, 801), 25.0, 100.0 / 3.0) == (141, 601)
    assert _stage_grid_shape((1, 5), 10.0, 20.0) == (1, 3)


# ------------------------------------------------------------ resampling --

@pytest.mark.parametrize("ratio", [2, 3])
@pytest.mark.parametrize("shape", [(38, 53), (13, 17, 22)])
def test_coincident_samples_are_reproduced_exactly(ratio, shape):
    rng = np.random.default_rng(7)
    native = torch.from_numpy(rng.uniform(1500, 4500, shape).astype(np.float32))
    dh0, dh = 12.5, 12.5 * ratio
    st_shape = _stage_grid_shape(shape, dh0, dh)
    coarse = _resample_to_grid(native, dh0, st_shape, dh)
    assert coarse.shape == st_shape and coarse.dtype == native.dtype
    # every stage sample IS a native sample: native index ratio*j, and past
    # the native extent (the partial last cell) the native edge sample
    idx = [np.minimum(np.arange(n) * ratio, n0 - 1) for n, n0 in zip(st_shape, shape)]
    assert any((n - 1) * ratio > n0 - 1 for n, n0 in zip(st_shape, shape)), \
        "case has no partial last cell; pick shapes that exercise it"
    assert torch.equal(coarse, native[np.ix_(*idx)])
    # and back: native samples that are stage samples come back exactly
    back = _resample_to_grid(coarse, dh, shape, dh0)
    sl = tuple(slice(0, n0, ratio) for n0 in shape)
    assert torch.equal(back[sl], native[sl])


def test_resample_is_identity_on_the_same_grid_and_refuses_masks():
    x = torch.randn(9, 11)
    y = _resample_to_grid(x, 10.0, (9, 11), 10.0)
    assert torch.equal(x, y) and y.data_ptr() != x.data_ptr()
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(_resample_to_grid(a, 5.0, (3, 4), 5.0), a)
    with pytest.raises(TypeError):
        _resample_to_grid(torch.zeros(4, 4, dtype=torch.bool), 1.0, (2, 2), 2.0)


@pytest.mark.parametrize("dh0,dh", [(12.5, 25.0), (12.5, 37.5), (25.0, 33.3333),
                                    (12.5, 18.75), (25.0, 12.5), (25.0, 18.75)])
def test_resampling_adds_no_spikes(dh0, dh):
    """Max |dv/dz| and |dv/dx| per metre never exceed the native model's.

    The artefact this guards against is the sparse spike a mis-registered
    resample produces (``scipy.ndimage.zoom`` with ``grid_mode=False`` on a
    non-integer ratio); linear interpolation at explicit coordinates is a
    convex combination of neighbours, so it cannot steepen anything.
    """
    nz, nx = 97, 141
    z = np.arange(nz)[:, None] * dh0
    x = np.arange(nx)[None, :] * dh0
    vp = (1500.0 + 0.7 * z + 120.0 * np.sin(2 * np.pi * x / 900.0)
          + 300.0 * np.exp(-((z - 600) ** 2 + (x - 900) ** 2) / (2 * 150.0 ** 2)))
    vp = vp.astype(np.float32)
    shape = _stage_grid_shape(vp.shape, dh0, dh)
    out = _resample_to_grid(vp, dh0, shape, dh)
    for ax in (0, 1):
        g_native = np.abs(np.diff(vp.astype(np.float64), axis=ax)).max() / dh0
        g_out = np.abs(np.diff(out.astype(np.float64), axis=ax)).max() / dh
        assert g_out <= g_native * (1 + 1e-5), (ax, g_out, g_native)
    assert out.min() >= vp.min() - 1e-3 and out.max() <= vp.max() + 1e-3


def test_seabed_map_follows_the_stage_columns():
    sd = np.linspace(100.0, 900.0, 41, dtype=np.float32)[None, :].repeat(21, 0)
    # native resolution, native grid: untouched
    assert np.array_equal(_seabed_to_grid(sd, (21, 41), 25.0, (21, 41), 25.0), sd)
    # stage grid at 50 m: column j sits at 50 j m = native column 2 j
    st = _stage_grid_shape((21, 41), 25.0, 50.0)
    got = _seabed_to_grid(sd, (21, 41), 25.0, st, 50.0)
    assert np.array_equal(got, sd[::2, ::2])
    # a map at half the model's resolution covers the same extent end to end
    half = sd[::2, ::2]
    got = _seabed_to_grid(half, (21, 41), 25.0, st, 50.0)
    np.testing.assert_allclose(got, sd[::2, ::2], rtol=0, atol=1e-3)


# -------------------------------------------------------------- geometry --

@pytest.mark.parametrize("native_shape,native_dh,dhs", [MARMOUSI, OVERTHRUST])
def test_edge_receiver_maps_inside_every_stage_grid(native_shape, native_dh, dhs):
    """A receiver on the native grid's last cell is on every stage grid.

    Geometry snaps with ``round(x / dh)`` (sweepx.io ``to_grid``). The legacy
    grid was one sample short there (Marmousi at 100 m: index 170 on a
    170-sample axis), so the far receiver recorded the absorbing layer.
    """
    from sweepx.io.geometry import PhysicalGeometry

    nd = len(native_shape)
    last = [(n - 1) * native_dh for n in native_shape][::-1]        # (x[, y], z)
    pg = PhysicalGeometry(sources_xyz_m=np.zeros((1, nd)),
                          receivers_xyz_m=np.asarray(last, float)[None, None],
                          dt=0.001, nt=10)
    for dh in dhs:
        gg, _ = pg.to_grid(dh=(dh,) * nd, dedupe=False)
        bounds = np.asarray(_stage_grid_shape(native_shape, native_dh, dh)[::-1])
        assert np.all(gg.receivers[0, 0] < bounds), (dh, gg.receivers, bounds)
        legacy = np.asarray(_legacy_shape(native_shape, native_dh, dh)[::-1])
        if dh == 100.0:
            assert np.any(gg.receivers[0, 0] >= legacy), "legacy grid held it?"


def test_finer_stage_keeps_edge_positions_on_the_grid():
    """A position within half a native cell outside the extent rounds onto
    the native edge, and can round one cell PAST a finer stage grid's edge;
    it is put on the edge node (moved < half a native cell)."""
    native_shape, dh0, dh = (11, 21), 25.0, 12.5
    st = _stage_grid_shape(native_shape, dh0, dh)               # (21, 41)
    pos = np.array([[[20 * 25.0 + 11.0, 4.0], [-11.0, 4.0], [250.0, 4.0],
                     [20 * 25.0 + 40.0, 4.0]]])               # (1, 4, (x, z))
    nat = np.rint(pos / dh0).astype(np.int64)
    stg = np.rint(pos / dh).astype(np.int64)
    out, moved = _snap_covered(stg, nat, st, native_shape)
    assert moved == 2
    assert out[0, 0, 0] == st[1] - 1 and out[0, 1, 0] == 0
    assert out[0, 2, 0] == 20
    assert out[0, 3, 0] == stg[0, 3, 0]          # outside the native grid too: untouched


# ------------------------------------------------------------- INR frame --

def _siren(base):
    from sweepx.nn import VelocityINR
    torch.manual_seed(3)
    return VelocityINR(base, vp_std=100.0, hidden_features=16, hidden_layers=1,
                       use_hash_encoding=False)


def test_inr_coordinates_stay_in_the_native_frame():
    """The same physical point gets the same network coordinate on a stage
    grid, so the learned field does not stretch at a stage change."""
    native_shape, dh0, dh = (31, 41), 10.0, 20.0
    base = torch.full(native_shape, 2000.0)
    net = _siren(base)
    with torch.no_grad():
        delta_native = net() - base
    st = _stage_grid_shape(native_shape, dh0, dh)                # (16, 21)
    sbase = _resample_to_grid(base, dh0, st, dh)
    net.update_base_velocity(sbase)
    with torch.no_grad():
        stretched = net() - sbase                                # sweep-nn i/n frame
    _set_physical_inr_frame(net, shape=st, dh=dh, frame_shape=native_shape,
                            frame_dh=dh0)
    with torch.no_grad():
        delta_stage = net() - sbase
    want = delta_native[::2, ::2]
    np.testing.assert_allclose(delta_stage.numpy(), want.numpy(), atol=1e-3)
    # negative control: without the frame the field is visibly displaced
    assert (stretched - want).abs().max() > 1.0
    # back on the native grid the network's own coordinates ARE the frame
    net.update_base_velocity(base)
    _set_physical_inr_frame(net, shape=native_shape, dh=dh0,
                            frame_shape=native_shape, frame_dh=dh0)
    with torch.no_grad():
        assert torch.equal(net() - base, delta_native)


def test_inr_frame_clamps_past_the_native_extent_and_refuses_aniso():
    from sweepx.nn import VelocityINR

    native_shape, dh0, dh = (12, 9), 10.0, 20.0     # z extent 110 m -> stage 120 m
    net = _siren(torch.full(native_shape, 2000.0))
    st = _stage_grid_shape(native_shape, dh0, dh)
    assert st == (7, 5)
    net.update_base_velocity(torch.full(st, 2000.0))
    _set_physical_inr_frame(net, shape=st, dh=dh, frame_shape=native_shape,
                            frame_dh=dh0)
    c = net.coords.reshape(*st, 2)
    assert float(c.max()) < 1.0                     # stays inside the encoder box
    assert torch.equal(c[-1, :, 0], c[-2, :, 0] * 0 + 110.0 / 120.0)
    aniso = VelocityINR(torch.full((6, 8, 8), 2000.0), use_hash_encoding=False,
                        hidden_features=8, hidden_layers=1, lateral_downsample=2)
    aniso.update_base_velocity(torch.full((3, 4, 4), 2000.0))
    with pytest.raises(NotImplementedError, match="lateral_downsample"):
        _set_physical_inr_frame(aniso, shape=(3, 4, 4), dh=20.0,
                                frame_shape=(6, 8, 8), frame_dh=10.0)
