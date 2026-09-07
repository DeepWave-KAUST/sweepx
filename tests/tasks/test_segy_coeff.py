"""SEG-Y -> 系数分片: 合成数据端到端。

这里不碰任何真实采集: 测试自己造 SEG-Y(含 IBM 浮点编码), 几何、切除律、
频率梳都是就地构造的, 所以判据是"输出等于对同一批道直接做 DTFT", 而不是
"和某次生产跑得一样"。
"""
import json
import struct

import numpy as np
import pytest

from sweep_tasks.freqsel import FrequencyComb, _comb_kernel
from sweep_tasks.preproc.segy_coeff import (
    BoxGrid, SegyLayout, TopMute, decode_ibm32, extract_coeff_from_segy,
    filter_by_manifest, source_bbox_manifest)

NT, DT = 64, 0.004
LAYOUT = SegyLayout()


def _ibm32(v: np.ndarray) -> np.ndarray:
    """IEEE float -> IBM 360 float32 bytes, 供解码判据用。"""
    v = np.asarray(v, np.float64)
    out = np.zeros(v.shape + (4,), np.uint8)
    sign = (v < 0).astype(np.int64)
    a = np.abs(v)
    exp = np.zeros(a.shape, np.int64)
    nz = a > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        exp[nz] = np.ceil(np.log(a[nz]) / np.log(16.0)).astype(np.int64)
    frac = np.zeros(a.shape, np.float64)
    frac[nz] = a[nz] / np.power(16.0, exp[nz].astype(np.float64))
    # frac 必须落在 [1/16, 1)
    fix = nz & (frac >= 1.0)
    frac[fix] /= 16.0; exp[fix] += 1
    m = np.zeros(a.shape, np.int64)
    m[nz] = np.minimum((frac[nz] * 2.0 ** 24).astype(np.int64), 2 ** 24 - 1)
    u = (sign << 31) | (((exp + 64) & 0x7F) << 24) | m
    u = np.where(nz, u, 0)
    for i, sh in enumerate((24, 16, 8, 0)):
        out[..., i] = (u >> sh) & 0xFF
    return out


def _write_segy(path, traces, sxy, gxy):
    """traces (n, NT) float, sxy/gxy (n, 2) 米。坐标标量按 layout 的 /100 写。"""
    n = len(traces)
    tb = LAYOUT.trace_bytes(NT)
    buf = bytearray(LAYOUT.text_header_bytes + n * tb)
    samples = _ibm32(traces)                       # (n, NT, 4)
    for i in range(n):
        o = LAYOUT.text_header_bytes + i * tb
        for off, val in ((LAYOUT.sx, sxy[i, 0]), (LAYOUT.sy, sxy[i, 1]),
                         (LAYOUT.gx, gxy[i, 0]), (LAYOUT.gy, gxy[i, 1])):
            buf[o + off:o + off + 4] = struct.pack(">i", int(round(val * 100)))
        buf[o + LAYOUT.trace_header_bytes:o + tb] = samples[i].tobytes()
    open(path, "wb").write(bytes(buf))
    return path


def test_ibm_roundtrip():
    v = np.array([0.0, 1.0, -1.0, 0.5, -12345.678, 1e-8, 3.25], np.float64)
    back = decode_ibm32(_ibm32(v))
    assert np.allclose(back, v, rtol=1e-6, atol=1e-12), (back, v)


def _comb():
    return FrequencyComb(n_p=256, dt=DT, ks=np.arange(4, 12))


def _case(tmp_path, *, n_files=3, per_file=6, dup_cell=False, out_of_box=1):
    rng = np.random.default_rng(0)
    box = BoxGrid(origin_x=0.0, origin_y=0.0, dh=50.0, nx=8, ny=6)
    paths, all_tr, all_s, all_g = [], [], [], []
    for f in range(n_files):
        tr = rng.standard_normal((per_file, NT))
        sx = np.full(per_file, 100.0) + (0 if dup_cell else np.arange(per_file) * 50.0)
        sy = np.full(per_file, 100.0)
        sx[:out_of_box] = -5000.0                       # 盒外
        gx = np.full(per_file, 200.0 + 300.0 * f)       # 每个文件一个节点
        gy = np.full(per_file, 150.0)
        p = _write_segy(tmp_path / f"l{f}.sgy", tr,
                        np.stack([sx, sy], 1), np.stack([gx, gy], 1))
        paths.append(str(p)); all_tr.append(tr)
        all_s.append(np.stack([sx, sy], 1)); all_g.append(np.stack([gx, gy], 1))
    return box, paths, np.concatenate(all_tr), np.concatenate(all_s), np.concatenate(all_g)


def test_extract_matches_direct_dtft(tmp_path):
    box, paths, tr, s, g = _case(tmp_path)
    comb = _comb()
    out = extract_coeff_from_segy(str(tmp_path / "shard.npz"), paths,
                                  layout=LAYOUT, box=box, comb=comb, nt=NT,
                                  dt_record=DT, nproc=2)
    z = np.load(out)
    cx, cy = box.cells(s[:, 0], s[:, 1])
    keep = box.inside(cx, cy)
    assert z["D"].shape == (int(keep.sum()), comb.n_bins)
    assert int(z["fold"].sum()) == int(keep.sum())
    # 系数必须等于对同一批道直接做 DTFT (解码后的道, 不是原始 float)
    E = np.asarray(_comb_kernel(comb, NT, DT))
    ref = (decode_ibm32(_ibm32(tr[keep])).astype(np.float64) @ E.T).astype(np.complex64)
    got = z["D"]
    # 行序按 (node, cell) 排, 用 (node,cell) 键把参考也排一遍
    node = np.repeat(np.arange(len(paths)), keep.reshape(len(paths), -1).sum(1))
    cell = (cx[keep] * box.ny + cy[keep])
    o = np.lexsort((cell, node))
    assert np.allclose(got, ref[o], rtol=1e-5, atol=1e-6)


def test_out_of_box_sources_are_dropped(tmp_path):
    box, paths, tr, s, g = _case(tmp_path, out_of_box=2)
    z = np.load(extract_coeff_from_segy(str(tmp_path / "s.npz"), paths,
                                        layout=LAYOUT, box=box, comb=_comb(),
                                        nt=NT, dt_record=DT, nproc=2))
    cx, cy = box.cells(s[:, 0], s[:, 1])
    assert len(z["D"]) == int(box.inside(cx, cy).sum()) < len(s)


def test_fold_averages_shared_cells(tmp_path):
    box, paths, tr, s, g = _case(tmp_path, dup_cell=True, out_of_box=0)
    z = np.load(extract_coeff_from_segy(str(tmp_path / "s.npz"), paths,
                                        layout=LAYOUT, box=box, comb=_comb(),
                                        nt=NT, dt_record=DT, nproc=2))
    # 每个文件 6 道全落同一个 cell -> 每个节点 1 行, fold=6
    assert len(z["D"]) == len(paths)
    assert np.array_equal(z["fold"], np.full(len(paths), 6, np.int32))
    E = np.asarray(_comb_kernel(_comb(), NT, DT))
    ref0 = (decode_ibm32(_ibm32(tr[:6])).astype(np.float64) @ E.T).mean(0)
    assert np.allclose(z["D"][0], ref0.astype(np.complex64), rtol=1e-5, atol=1e-6)


def test_npart_partitions_nodes(tmp_path):
    box, paths, tr, s, g = _case(tmp_path, out_of_box=0)
    rows = []
    for part in (0, 1):
        z = np.load(extract_coeff_from_segy(str(tmp_path / f"p{part}.npz"), paths,
                                            layout=LAYOUT, box=box, comb=_comb(),
                                            nt=NT, dt_record=DT, nproc=2,
                                            part=part, npart=2))
        rows.append(len(z["D"]))
    whole = np.load(extract_coeff_from_segy(str(tmp_path / "w.npz"), paths,
                                            layout=LAYOUT, box=box, comb=_comb(),
                                            nt=NT, dt_record=DT, nproc=2))
    assert sum(rows) == len(whole["D"]), (rows, len(whole["D"]))
    assert all(r > 0 for r in rows), (
        f"分片退化, 某个 part 是空的: {rows} —— 节点 uid 打包后低位是同一个坐标, "
        "对原始 key 取模会按该坐标的奇偶分, 必须先散列")


def test_node_part_is_balanced_on_a_grid():
    """规则网格上的分片必须均衡 —— 这正是 uid % npart 退化的场景。"""
    from sweep_tasks.preproc.segy_coeff import _node_uid, node_part
    gx, gy = np.meshgrid(np.arange(40) * 250.0, np.arange(25) * 250.0)
    uid = _node_uid(gx.ravel(), gy.ravel(), 0.1)
    for npart in (2, 3, 4):
        c = np.bincount(node_part(uid, npart), minlength=npart)
        assert c.min() > 0.8 * c.max(), (npart, c)
    raw = np.bincount((np.abs(uid) % 2).astype(int), minlength=2)
    assert raw.min() == 0, "旧的取模方式在这个网格上竟然没退化, 判据失效"


def test_mute_zeroes_before_the_cut(tmp_path):
    box, paths, tr, s, g = _case(tmp_path, n_files=1, per_file=4, out_of_box=0)
    comb = _comb()
    mute = TopMute(t0=0.10, v=1e9, a=0.0, guard=0.0, taper=0.02)
    a = np.load(extract_coeff_from_segy(str(tmp_path / "a.npz"), paths,
                                        layout=LAYOUT, box=box, comb=comb,
                                        nt=NT, dt_record=DT, nproc=1))
    b = np.load(extract_coeff_from_segy(str(tmp_path / "b.npz"), paths,
                                        layout=LAYOUT, box=box, comb=comb,
                                        nt=NT, dt_record=DT, nproc=1, mute=mute))
    assert not np.allclose(a["D"], b["D"]), "mute 没生效, 判据恒真"
    E = np.asarray(_comb_kernel(comb, NT, DT))
    t = np.arange(NT) * DT
    w = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - 0.10) / 0.02, 0, 1))
    ref = ((decode_ibm32(_ibm32(tr)) * w).astype(np.float64) @ E.T)
    assert np.allclose(np.sort_complex(b["D"].ravel())[:5],
                       np.sort_complex(ref.astype(np.complex64).ravel())[:5],
                       rtol=1e-4, atol=1e-5)


def test_manifest_skip_is_bit_identical(tmp_path):
    box, paths, tr, s, g = _case(tmp_path, n_files=4, per_file=5, out_of_box=0)
    # 再加一个整文件都在盒外的
    far = _write_segy(tmp_path / "far.sgy", np.zeros((3, NT)),
                      np.full((3, 2), -9000.0), np.full((3, 2), -9000.0))
    allp = paths + [str(far)]
    comb = _comb()
    a = extract_coeff_from_segy(str(tmp_path / "a.npz"), allp, layout=LAYOUT,
                                box=box, comb=comb, nt=NT, dt_record=DT, nproc=2)
    man = source_bbox_manifest(allp, LAYOUT, NT, nproc=2)
    kept = filter_by_manifest(allp, man, box)
    assert len(kept) == len(paths), "盒外文件没被跳掉"
    b = extract_coeff_from_segy(str(tmp_path / "b.npz"), allp, layout=LAYOUT,
                                box=box, comb=comb, nt=NT, dt_record=DT,
                                nproc=2, manifest=man)
    za, zb = np.load(a), np.load(b)
    for k in za.files:
        assert np.array_equal(np.ascontiguousarray(za[k]).view(np.uint8),
                              np.ascontiguousarray(zb[k]).view(np.uint8)), k


def test_manifest_keeps_unknown_files(tmp_path):
    box, paths, *_ = _case(tmp_path, n_files=2, per_file=4, out_of_box=0)
    assert filter_by_manifest(paths, {}, box) == paths


def test_cli_extract_coeff_segy(tmp_path):
    """子命令端到端: 与直接调 API 的输出逐位相同, 且 manifest 会被建出来。"""
    from sweep_tasks.cli import main

    box, paths, tr, s, g = _case(tmp_path, n_files=3, per_file=5, out_of_box=1)
    comb = _comb()
    api = extract_coeff_from_segy(str(tmp_path / "api.npz"), paths,
                                  layout=LAYOUT, box=box, comb=comb, nt=NT,
                                  dt_record=DT, nproc=2)
    man = tmp_path / "man.json"
    rc = main(["extract-coeff-segy",
               "--segy", str(tmp_path / "l*.sgy"),
               "-o", str(tmp_path / "cli.npz"),
               "--nt", str(NT), "--dt-record", str(DT),
               "--n-p", str(comb.n_p), "--k-lo", "4", "--k-hi", "11",
               "--dt-solver", str(DT),
               "--origin", f"{box.origin_x},{box.origin_y}",
               "--dh", str(box.dh), "--nx", str(box.nx), "--ny", str(box.ny),
               "--manifest", str(man), "--nproc", "2"])
    assert rc == 0
    assert man.exists() and len(json.load(open(man))) == len(paths)
    za, zb = np.load(api), np.load(tmp_path / "cli.npz")
    assert sorted(za.files) == sorted(zb.files)
    for k in za.files:
        assert np.array_equal(np.ascontiguousarray(za[k]).view(np.uint8),
                              np.ascontiguousarray(zb[k]).view(np.uint8)), k


def test_cli_rejects_empty_glob(tmp_path):
    from sweep_tasks.cli import main
    assert main(["extract-coeff-segy", "--segy", str(tmp_path / "none*.sgy"),
                 "-o", str(tmp_path / "x.npz"), "--nt", "8", "--n-p", "64",
                 "--k-lo", "2", "--k-hi", "5", "--dt-solver", "0.004",
                 "--dh", "50", "--nx", "4", "--ny", "4"]) == 2
