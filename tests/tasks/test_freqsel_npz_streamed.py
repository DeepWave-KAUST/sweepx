"""write_npz_streamed: 与 np.savez 逐位一致, 且峰值不含第二份满尺寸副本。

extract_shard_gathers 的 docstring 承诺 "streaming by design ... still extracts
in one pass", 但它最后 np.savez(..., D=np.concatenate(D_parts, 0)) —— parts 和
它们的拷贝同时在手, 峰值是系数表的 2 倍。这组判据钉住修复。
"""
import json
import tracemalloc

import numpy as np
import pytest

from sweep_tasks.freqsel import FrequencyComb, extract_shard_gathers, write_npz_streamed


def _members(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _bitsame(a, b):
    a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
    return (a.shape == b.shape and a.dtype == b.dtype
            and np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def test_streamed_matches_savez_bitwise(tmp_path):
    rng = np.random.default_rng(0)
    parts = [(rng.standard_normal((n, 7)) + 1j * rng.standard_normal((n, 7))
              ).astype(np.complex64) for n in (5, 1, 9, 3)]
    parts[0][0, 0] = np.complex64(complex(np.nan, np.inf))   # 非有限值也要一致
    parts[2][1, 3] = np.complex64(complex(-0.0, 0.0))        # 负零
    whole = np.concatenate(parts, 0)
    small = dict(node_ptr=np.array([0, 5, 6, 15, 18], np.int64),
                 fold=np.ones(18, np.int32),
                 freqs=np.linspace(2.0, 4.0, 7),
                 meta=json.dumps(dict(a=1)))

    ref = tmp_path / "ref.npz"
    np.savez(ref, D=whole, **small)
    got = tmp_path / "got.npz"
    write_npz_streamed(got, stream_name="D", stream_parts=list(parts),
                       stream_shape=whole.shape, stream_dtype=np.complex64, **small)

    a, b = _members(ref), _members(got)
    assert sorted(a) == sorted(b)
    for k in a:
        assert _bitsame(a[k], b[k]), f"member {k} differs"


def test_streamed_consumes_parts(tmp_path):
    """parts 必须被就地释放, 否则峰值降不下来。"""
    parts = [np.zeros((4, 3), np.complex64) for _ in range(3)]
    write_npz_streamed(tmp_path / "o.npz", stream_name="D", stream_parts=parts,
                       stream_shape=(12, 3), stream_dtype=np.complex64,
                       fold=np.ones(12, np.int32))
    assert parts == [None, None, None]


def test_streamed_row_count_is_checked(tmp_path):
    parts = [np.zeros((4, 3), np.complex64)]
    with pytest.raises(ValueError, match="rows"):
        write_npz_streamed(tmp_path / "o.npz", stream_name="D",
                           stream_parts=parts, stream_shape=(9, 3),
                           stream_dtype=np.complex64)


def test_streamed_peak_is_half_of_savez(tmp_path):
    """配对对照: 同一批 parts, 新旧两条写入路径的峰值。

    上一版判据测的是整个 extract_shard_gathers 的峰值, 里面含记录、DTFT 中间量、
    trace_parts 等合法开销 —— 区分不出"有没有第二份副本"。这里只围住写入本身,
    旧路径的 np.concatenate 必然多拿一整张表, 新路径不该多拿。
    """
    rows, bins = 4096, 48
    nparts = 16
    parts = [np.zeros((rows // nparts, bins), np.complex64) for _ in range(nparts)]
    table = rows * bins * 8
    small = dict(fold=np.ones(rows, np.int32), freqs=np.linspace(2, 4, bins))

    old = list(parts)
    tracemalloc.start()
    np.savez(tmp_path / "old.npz", D=np.concatenate(old, 0), **small)
    _, peak_old = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del old

    new = list(parts)
    tracemalloc.start()
    write_npz_streamed(tmp_path / "new.npz", stream_name="D", stream_parts=new,
                       stream_shape=(rows, bins), stream_dtype=np.complex64, **small)
    _, peak_new = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 判据先自证可达: 旧路径确实要多拿一张表, 否则这条测试是恒真的
    assert peak_old >= table, (
        f"peak_old {peak_old} < one table {table}: tracemalloc 没抓到 numpy 分配, "
        "判据无效")
    assert peak_new < table / 2, f"peak_new {peak_new} >= half a table {table/2}"
    assert _bitsame(np.load(tmp_path / "old.npz")["D"],
                    np.load(tmp_path / "new.npz")["D"])
    print(f"\n  peak_old {peak_old/1024:.0f} KiB  peak_new {peak_new/1024:.0f} KiB  "
          f"one table {table/1024:.0f} KiB")
