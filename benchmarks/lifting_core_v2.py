#!/usr/bin/env python3
"""Zero-copy NumPy + ctypes Python binding & verification suite for liblifting_core_v2.dylib."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Q14_ONE = 16384
Q14_NOT = -32768
LOG_ZERO = -32768  # INT16_MIN (⊥)

# ============================================================================
# 1. C99 構造体と 1 バイトも違わない NumPy 構造化 dtype 定義
# ============================================================================
C16_DTYPE = np.dtype([("re", np.int16), ("im", np.int16)], align=True)
LOGC16_DTYPE = np.dtype([("rho", np.int16), ("theta", np.uint16)], align=True)

BLOCK8_DTYPE = np.dtype(
    [
        ("s_re",        np.int16, (8,)), ("s_im",        np.int16, (8,)),
        ("g_re",        np.int16, (8,)), ("g_im",        np.int16, (8,)),
        ("v_re",        np.int16, (8,)), ("v_im",        np.int16, (8,)),
        ("w_re",        np.int16, (8,)), ("w_im",        np.int16, (8,)),
        ("s_eff_re",    np.int16, (8,)), ("s_eff_im",    np.int16, (8,)),
        ("p_re",        np.int16, (8,)), ("p_im",        np.int16, (8,)),
        ("r_re",        np.int16, (8,)), ("r_im",        np.int16, (8,)),
        ("restored_re", np.int16, (8,)), ("restored_im", np.int16, (8,)),
        ("adj_in",      np.int32, (8,)), ("adj_s",       np.int32, (8,)),
        ("adj_g",       np.int32, (8,)), ("gate_mask",   np.int32, (8,)),
    ],
    align=True,
)

LOG_BLOCK8_DTYPE = np.dtype(
    [
        ("s_rho",    np.int16,  (8,)), ("s_theta",   np.uint16, (8,)),
        ("g_rho",    np.int16,  (8,)), ("g_theta",   np.uint16, (8,)),
        ("w_rho",    np.int16,  (8,)), ("w_theta",   np.uint16, (8,)),
        ("v_re",     np.int16,  (8,)), ("v_im",      np.int16,  (8,)),
        ("se_rho",   np.int16,  (8,)), ("se_theta",  np.uint16, (8,)),
        ("p_rho",    np.int16,  (8,)), ("p_theta",   np.uint16, (8,)),
        ("p_re",     np.int16,  (8,)), ("p_im",      np.int16,  (8,)),
        ("r_re",     np.int16,  (8,)), ("r_im",      np.int16,  (8,)),
        ("res_re",   np.int16,  (8,)), ("res_im",    np.int16,  (8,)),
        ("unb_rho",  np.int16,  (8,)), ("unb_theta", np.uint16, (8,)),
    ],
    align=True,
)

CHART_BLOCK8_DTYPE = np.dtype(
    [
        ("re",    np.int16,  (8,)), ("im",    np.int16,  (8,)),
        ("rho",   np.int16,  (8,)), ("theta", np.uint16, (8,)),
        ("rt_re", np.int16,  (8,)), ("rt_im", np.int16,  (8,)),
        ("pad",   np.int16,  (16,)),
    ],
    align=True,
)

HOLO_BLOCK8_DTYPE = np.dtype(
    [
        ("z1_rho",     np.int16, (8,)), ("z1_theta",   np.uint16, (8,)),
        ("z2_rho",     np.int16, (8,)), ("z2_theta",   np.uint16, (8,)),
        ("tw_rho",     np.int16, (8,)), ("tw_theta",   np.uint16, (8,)),
        ("out_drho",   np.int16, (8,)), ("out_dtheta", np.int16,  (8,)),
    ],
    align=True,
)

assert C16_DTYPE.itemsize == 4 and LOGC16_DTYPE.itemsize == 4
assert BLOCK8_DTYPE.itemsize == 384 and LOG_BLOCK8_DTYPE.itemsize == 320
assert CHART_BLOCK8_DTYPE.itemsize == 128 and HOLO_BLOCK8_DTYPE.itemsize == 128


def alloc_aligned_ndarray(count: int, dtype: np.dtype, align: int = 64) -> np.ndarray:
    """Allocate a zero-copy 1D C-contiguous NumPy structured array aligned to `align` bytes."""
    nbytes = count * dtype.itemsize
    raw = np.zeros(nbytes + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) & (align - 1)
    arr = raw[offset : offset + nbytes].view(dtype)
    assert arr.shape == (count,)
    assert (arr.ctypes.data & (align - 1)) == 0
    assert arr.flags["C_CONTIGUOUS"]
    return arr


def _check_ndarray(arr: np.ndarray, expected_dtype: np.dtype, require_64b: bool = True) -> tuple[ctypes.c_void_p, int]:
    if not isinstance(arr, np.ndarray) or arr.dtype != expected_dtype:
        raise TypeError(f"Expected np.ndarray with dtype={expected_dtype}, got {type(arr).__name__}({getattr(arr, 'dtype', None)})")
    if arr.ndim != 1 or not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Array must be 1D and C-contiguous")
    if require_64b and (arr.ctypes.data & 63) != 0:
        raise ValueError(f"Buffer at 0x{arr.ctypes.data:x} is not 64-byte aligned; use alloc_aligned_ndarray()")
    return ctypes.c_void_p(arr.ctypes.data), int(arr.shape[0])


@dataclass
class DiamondResult:
    ok: bool
    spawned_3morph: bool
    steps: int
    static_l1: int
    twist_rho: int
    twist_theta: int


@dataclass
class AssociatorResult:
    ok: bool
    rho: int
    theta: int
    alpha_drho: int
    alpha_dtheta: int
    alpha_has_bot: bool
    reassociated: bool


class LiftingCoreV2:
    def __init__(self, lib_path: str | Path | None = None):
        if lib_path is None:
            lib_path = Path(__file__).resolve().parent / "liblifting_core_v2.dylib"
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        if self.lib.lc_v2_ffi_init() != 0:
            raise RuntimeError("lc_v2_ffi_init() failed")
        assert self.lib.lc_v2_ffi_sizeof_cell() == 64
        assert self.lib.lc_v2_ffi_sizeof_cell_log() == 64
        assert self.lib.lc_v2_ffi_sizeof_block8() == 384
        assert self.lib.lc_v2_ffi_sizeof_log_block8() == 320
        assert self.lib.lc_v2_ffi_sizeof_chart_block8() == 128
        assert self.lib.lc_v2_ffi_sizeof_holo_block8() == 128

    def __enter__(self) -> LiftingCoreV2:
        self.mt_init(self.cpu_count)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.mt_shutdown()

    def _bind(self) -> None:
        L = self.lib
        L.lc_v2_ffi_init.restype = ctypes.c_int
        L.lc_v2_ffi_sizeof_cell.restype = ctypes.c_size_t
        L.lc_v2_ffi_sizeof_cell_log.restype = ctypes.c_size_t
        L.lc_v2_ffi_sizeof_block8.restype = ctypes.c_size_t
        L.lc_v2_ffi_sizeof_log_block8.restype = ctypes.c_size_t
        L.lc_v2_ffi_sizeof_chart_block8.restype = ctypes.c_size_t
        L.lc_v2_ffi_sizeof_holo_block8.restype = ctypes.c_size_t
        L.lc_v2_ffi_has_neon.restype = ctypes.c_int
        L.lc_v2_ffi_cpu_count.restype = ctypes.c_int

        L.lc_v2_ffi_mt_init.argtypes = [ctypes.c_int]
        L.lc_v2_ffi_mt_init.restype = ctypes.c_int
        L.lc_v2_ffi_mt_shutdown.argtypes = []

        L.lc_v2_ffi_dual_ladder_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int
        ]
        L.lc_v2_ffi_dual_ladder_inverse.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int
        ]
        L.lc_v2_ffi_chart_cart_to_log_mt.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        L.lc_v2_ffi_chart_cart_to_log_mt.restype = ctypes.c_int

        L.lc_v2_ffi_diamond_solve_h1.argtypes = [
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint16),
        ]
        L.lc_v2_ffi_diamond_solve_h1.restype = ctypes.c_int

        L.lc_v2_ffi_assoc_resolve.argtypes = [
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.c_int16, ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        L.lc_v2_ffi_assoc_resolve.restype = ctypes.c_int

        L.lc_v2_ffi_holo_eval_mt.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
        ]
        L.lc_v2_ffi_holo_eval_mt.restype = ctypes.c_int

        L.lc_v2_ffi_log_bind_mt.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        L.lc_v2_ffi_log_bind_mt.restype = ctypes.c_int

        L.lc_v2_ffi_log_dual_mt.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        L.lc_v2_ffi_log_dual_mt.restype = ctypes.c_int

        L.lc_v2_ffi_fill_log_blocks.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]

    @property
    def has_neon(self) -> bool:
        return bool(self.lib.lc_v2_ffi_has_neon())

    @property
    def cpu_count(self) -> int:
        return int(self.lib.lc_v2_ffi_cpu_count())

    def mt_init(self, max_workers: int = 0) -> int:
        n = self.lib.lc_v2_ffi_mt_init(max_workers)
        if n < 0:
            raise RuntimeError(f"lc_v2_ffi_mt_init failed ({n})")
        return n

    def mt_shutdown(self) -> None:
        self.lib.lc_v2_ffi_mt_shutdown()

    # --- Layer 1: Dual-Chart & 64-Layer Closed-Loop Ladder ---
    def dual_ladder_forward(self, s: np.ndarray, v: np.ndarray, weights: np.ndarray) -> None:
        ps, ns = _check_ndarray(s, C16_DTYPE, require_64b=False)
        pv, nv = _check_ndarray(v, C16_DTYPE, require_64b=False)
        pw, nw = _check_ndarray(weights, LOGC16_DTYPE, require_64b=False)
        assert ns == nv
        self.lib.lc_v2_ffi_dual_ladder_forward(ps, pv, ns, pw, nw)

    def dual_ladder_inverse(self, s: np.ndarray, v: np.ndarray, weights: np.ndarray) -> None:
        ps, ns = _check_ndarray(s, C16_DTYPE, require_64b=False)
        pv, nv = _check_ndarray(v, C16_DTYPE, require_64b=False)
        pw, nw = _check_ndarray(weights, LOGC16_DTYPE, require_64b=False)
        assert ns == nv
        self.lib.lc_v2_ffi_dual_ladder_inverse(ps, pv, ns, pw, nw)

    def cart_to_log(self, blocks: np.ndarray, iters: int = 1,
                    num_threads: int = 1, batch_chunks: int = 32) -> None:
        ptr, n_blocks = _check_ndarray(blocks, CHART_BLOCK8_DTYPE, require_64b=True)
        rc = self.lib.lc_v2_ffi_chart_cart_to_log_mt(ptr, n_blocks, iters, num_threads, batch_chunks)
        if rc != 0:
            raise RuntimeError(f"cart_to_log failed ({rc})")

    # --- Layer 2: Higher-Category H1 3-Morphism Rewrite & Associator & Holonomy ---
    def solve_diamond_h1(self, base_a: tuple[int, int], w_f1: tuple[int, int],
                         w_f2: tuple[int, int], w_g2_init: tuple[int, int],
                         env_twist: tuple[int, int]) -> DiamondResult:
        sp, st, sl1 = ctypes.c_int(), ctypes.c_int(), ctypes.c_int32()
        tr, tt = ctypes.c_int16(), ctypes.c_uint16()
        ok = self.lib.lc_v2_ffi_diamond_solve_h1(
            base_a[0], base_a[1], w_f1[0], w_f1[1], w_f2[0], w_f2[1],
            w_g2_init[0], w_g2_init[1], env_twist[0], env_twist[1],
            ctypes.byref(sp), ctypes.byref(st), ctypes.byref(sl1),
            ctypes.byref(tr), ctypes.byref(tt),
        )
        return DiamondResult(bool(ok), bool(sp.value), st.value, sl1.value, tr.value, tt.value)

    def resolve_associator(self, a: tuple[int, int], b: tuple[int, int],
                           c: tuple[int, int]) -> AssociatorResult:
        vr, vt = ctypes.c_int16(), ctypes.c_uint16()
        adr, adt = ctypes.c_int32(), ctypes.c_int16()
        abot, reass = ctypes.c_int(), ctypes.c_int()
        ok = self.lib.lc_v2_ffi_assoc_resolve(
            a[0], a[1], b[0], b[1], c[0], c[1],
            ctypes.byref(vr), ctypes.byref(vt), ctypes.byref(adr), ctypes.byref(adt),
            ctypes.byref(abot), ctypes.byref(reass),
        )
        return AssociatorResult(bool(ok), vr.value, vt.value, adr.value, adt.value,
                                bool(abot.value), bool(reass.value))

    def eval_holonomy_2morph(self, blocks: np.ndarray, apply_twist: bool = False,
                             iters: int = 1, num_threads: int = 1, batch_chunks: int = 32) -> int:
        ptr, n_blocks = _check_ndarray(blocks, HOLO_BLOCK8_DTYPE, require_64b=True)
        out_l1 = ctypes.c_uint64()
        rc = self.lib.lc_v2_ffi_holo_eval_mt(
            ptr, n_blocks, iters, int(apply_twist), num_threads, batch_chunks, ctypes.byref(out_l1)
        )
        if rc != 0:
            raise RuntimeError(f"eval_holonomy_2morph failed ({rc})")
        return int(out_l1.value)

    # --- Layer 6: Log-Polar FHRR Bind & Dual-Chart ---
    def fill_log_blocks(self, blocks: np.ndarray, seed: int = 0x20261123) -> None:
        ptr, n_blocks = _check_ndarray(blocks, LOG_BLOCK8_DTYPE, require_64b=True)
        self.lib.lc_v2_ffi_fill_log_blocks(ptr, n_blocks, seed)

    def log_fhrr_bind(self, blocks: np.ndarray, iters: int = 1,
                      num_threads: int = 1, batch_chunks: int = 32) -> None:
        ptr, n_blocks = _check_ndarray(blocks, LOG_BLOCK8_DTYPE, require_64b=True)
        rc = self.lib.lc_v2_ffi_log_bind_mt(ptr, n_blocks, iters, num_threads, batch_chunks)
        if rc != 0:
            raise RuntimeError(f"log_fhrr_bind failed ({rc})")

    def log_dual_step(self, blocks: np.ndarray, iters: int = 1,
                      num_threads: int = 1, batch_chunks: int = 32) -> None:
        ptr, n_blocks = _check_ndarray(blocks, LOG_BLOCK8_DTYPE, require_64b=True)
        rc = self.lib.lc_v2_ffi_log_dual_mt(ptr, n_blocks, iters, num_threads, batch_chunks)
        if rc != 0:
            raise RuntimeError(f"log_dual_step failed ({rc})")


def main() -> None:
    with LiftingCoreV2() as core:
        max_cpus = core.cpu_count
        print("=== Python NumPy ゼロコピー <-> liblifting_core_v2.dylib 統合検証 ===")
        print(f"NEON={core.has_neon} | 論理CPU={max_cpus} | "
              f"dtypes=(C16:4B, LOGC16:4B, LOG8:320B, CHART8:128B, HOLO8:128B)\n")

        rng = np.random.default_rng(20261123)

        # ------------------------------------------------------------------
        # [Test 1] 純整数双対チャート Cart -> Log (65,536 cells ゼロコピー + MT)
        # ------------------------------------------------------------------
        n_ch_blk = 8192  # 65,536 cells = 1.0 MB
        ch_ref = alloc_aligned_ndarray(n_ch_blk, CHART_BLOCK8_DTYPE)
        ch_mt = alloc_aligned_ndarray(n_ch_blk, CHART_BLOCK8_DTYPE)

        ch_ref["re"] = rng.integers(-32768, 32768, size=(n_ch_blk, 8), dtype=np.int16)
        ch_ref["im"] = rng.integers(-32768, 32768, size=(n_ch_blk, 8), dtype=np.int16)
        # 軸・原点 ⊥・(-32768, -32768) の境界検証ケースを先頭ブロックへ注入
        ch_ref["re"][0, :4] = [0, 16384, -16384, -32768]
        ch_ref["im"][0, :4] = [0, 16384,      0, -32768]
        ch_mt[:] = ch_ref

        iters_ch = 200
        core.cart_to_log(ch_ref, iters=5, num_threads=1)
        t0 = time.perf_counter_ns()
        core.cart_to_log(ch_ref, iters=iters_ch, num_threads=1)
        dt_ch_1t = time.perf_counter_ns() - t0

        core.cart_to_log(ch_mt, iters=5, num_threads=max_cpus, batch_chunks=16)
        t1 = time.perf_counter_ns()
        core.cart_to_log(ch_mt, iters=iters_ch, num_threads=max_cpus, batch_chunks=16)
        dt_ch_mt = time.perf_counter_ns() - t1

        assert np.array_equal(ch_ref, ch_mt)
        assert ch_mt["rho"][0, 0] == LOG_ZERO and ch_mt["theta"][0, 0] == 0
        assert ch_mt["rho"][0, 1] == 128 and ch_mt["theta"][0, 1] == 8192
        assert ch_mt["rho"][0, 2] == 0 and ch_mt["theta"][0, 2] == 32768
        assert ch_mt["rho"][0, 3] == 384 and ch_mt["theta"][0, 3] == 40960

        g_ch_1t = (n_ch_blk * 8 * iters_ch) / dt_ch_1t
        g_ch_mt = (n_ch_blk * 8 * iters_ch) / dt_ch_mt
        print(f"  [Test 1] 純整数双対チャート Cart->Log (65,536 cells, 0-bit 1T=={max_cpus}T OK):")
        print(f"           1T={g_ch_1t:5.2f} G/s | {max_cpus}T={g_ch_mt:5.2f} G/s ({g_ch_mt / g_ch_1t:.2f}x) | 軸・⊥・極値 4/4 一致\n")

        # ------------------------------------------------------------------
        # [Test 2] 64 層双対チャート閉ループ・リフティング梯子 (NumPy C16 直接渡し)
        # ------------------------------------------------------------------
        n_pairs, depth = 4096, 64
        s0 = alloc_aligned_ndarray(n_pairs, C16_DTYPE)
        v0 = alloc_aligned_ndarray(n_pairs, C16_DTYPE)
        w_lad = alloc_aligned_ndarray(depth, LOGC16_DTYPE)

        s0["re"] = rng.integers(-32768, 32768, size=n_pairs, dtype=np.int16)
        s0["im"] = rng.integers(-32768, 32768, size=n_pairs, dtype=np.int16)
        v0["re"] = rng.integers(-32768, 32768, size=n_pairs, dtype=np.int16)
        v0["im"] = rng.integers(-32768, 32768, size=n_pairs, dtype=np.int16)
        s0["re"][0] = s0["im"][0] = 0
        v0["re"][1] = v0["im"][1] = -32768
        w_lad["rho"] = rng.integers(-512, 512, size=depth, dtype=np.int16)
        w_lad["theta"] = rng.integers(0, 65536, size=depth, dtype=np.uint16)

        s = s0.copy()
        v = v0.copy()
        t0 = time.perf_counter_ns()
        core.dual_ladder_forward(s, v, w_lad)
        scrambled = int(np.count_nonzero((s["re"] != s0["re"]) | (v["re"] != v0["re"])))
        core.dual_ladder_inverse(s, v, w_lad)
        dt_lad_ms = (time.perf_counter_ns() - t0) / 1e6

        assert scrambled == n_pairs
        assert np.array_equal(s, s0) and np.array_equal(v, v0)
        print(f"  [Test 2] 64 層双対チャート閉ループ梯子 ({n_pairs} 複素ペア, 往復 {dt_lad_ms:.2f} ms):")
        print(f"           攪拌率={scrambled}/{n_pairs} -> μ^64∘Δ^64=id 0-bit 完全復元 OK\n")

        # ------------------------------------------------------------------
        # [Test 3] 高次圏 2-射 H1 -> 3-射生成・Associator 回復・65,536d NEON 検出＆1 パス補償
        # ------------------------------------------------------------------
        dm_res = core.solve_diamond_h1(
            base_a=(256, 0x1000), w_f1=(128, 0x1111), w_f2=(64, 0x2222),
            w_g2_init=(-100, 0x5555), env_twist=(-384, 0x2B73),
        )
        assert dm_res.ok and dm_res.spawned_3morph
        assert dm_res.twist_rho == -384 and dm_res.twist_theta == 0x2B73

        as_res = core.resolve_associator((-20000, 100), (-20000, 200), (20000, 300))
        assert as_res.ok and as_res.reassociated and as_res.alpha_has_bot
        assert as_res.rho == -20000 and as_res.theta == 600

        n_hb = 8192  # 65,536 次元 = 1.0 MB
        hb_ref = alloc_aligned_ndarray(n_hb, HOLO_BLOCK8_DTYPE)
        hb_mt = alloc_aligned_ndarray(n_hb, HOLO_BLOCK8_DTYPE)
        z2_r = rng.integers(-8192, 8192, size=(n_hb, 8), dtype=np.int16)
        z2_t = rng.integers(0, 65536, size=(n_hb, 8), dtype=np.uint16)
        hid_r = rng.integers(-4096, 4096, size=(n_hb, 8), dtype=np.int16)
        hid_t = rng.integers(0, 65536, size=(n_hb, 8), dtype=np.uint16)
        hid_t[:, 0] = 0x8000  # 対蹠点 π 強制注入

        hb_ref["z2_rho"] = z2_r
        hb_ref["z2_theta"] = z2_t
        hb_ref["z1_rho"] = (z2_r.astype(np.int32) + hid_r.astype(np.int32)).astype(np.int16)
        hb_ref["z1_theta"] = ((z2_t.astype(np.uint32) + hid_t.astype(np.uint32)) & 0xFFFF).astype(np.uint16)
        hb_mt[:] = hb_ref

        l1_pre_1t = core.eval_holonomy_2morph(hb_ref, apply_twist=False, num_threads=1)
        l1_pre_mt = core.eval_holonomy_2morph(hb_mt, apply_twist=False, num_threads=max_cpus, batch_chunks=16)
        assert l1_pre_1t == l1_pre_mt and l1_pre_mt > 0 and np.array_equal(hb_ref, hb_mt)

        # NumPy ゼロコピー 1 パス補償 1-射合成 -> 再評価で全 65,536 次元 0-bit 平坦化
        hb_mt["tw_rho"][:] = hb_mt["out_drho"]
        hb_mt["tw_theta"][:] = hb_mt["out_dtheta"].view(np.uint16)
        iters_h = 300
        t0 = time.perf_counter_ns()
        l1_post_mt = core.eval_holonomy_2morph(
            hb_mt, apply_twist=True, iters=iters_h, num_threads=max_cpus, batch_chunks=16
        )
        dt_h_mt = time.perf_counter_ns() - t0
        assert l1_post_mt == 0
        assert np.all(hb_mt["out_drho"] == 0) and np.all(hb_mt["out_dtheta"] == 0)
        g_h_mt = (n_hb * 8 * iters_h) / dt_h_mt

        print(f"  [Test 3] 高次圏 2-射/3-射 & Associator:")
        print(f"           H1(L1={dm_res.static_l1}) -> 3-射生成 {dm_res.steps} steps, w_twist=({dm_res.twist_rho}, 0x{dm_res.twist_theta:04X}) OK")
        print(f"           Associator 偽⊥検出 -> 組換回復 rho={as_res.rho}, theta={as_res.theta} OK")
        print(f"           65,536d 2-射 NEON({max_cpus}T): 前L1={l1_pre_mt:,} -> 補償後L1={l1_post_mt} ({g_h_mt:5.2f} G dims/s, 0-bit OK)\n")

        # ------------------------------------------------------------------
        # [Test 4] Log-Polar FHRR Bind & Dual-Chart Step (1,048,576 cells = 40.0 MB, batch=32)
        # ------------------------------------------------------------------
        n_lb = 131072  # 1,048,576 cells = 40.0 MB
        n_cells = n_lb * 8
        iters_lb = 60

        lb_ref = alloc_aligned_ndarray(n_lb, LOG_BLOCK8_DTYPE)
        lb_mt = alloc_aligned_ndarray(n_lb, LOG_BLOCK8_DTYPE)

        print(f"  [Test 4] Log-Polar FHRR AoSoA-8 ({n_cells:,} cells = 40.0 MB, {iters_lb} iters, batch=32):")
        for label, fn in [("Pure FHRR Bind (0-bit)", core.log_fhrr_bind),
                          ("Dual-Chart Bind+LUT+Δ/μ", core.log_dual_step)]:
            core.fill_log_blocks(lb_ref, 0x20261123)
            fn(lb_ref, iters=5, num_threads=1)
            core.fill_log_blocks(lb_ref, 0x20261123)
            t0 = time.perf_counter_ns()
            fn(lb_ref, iters=iters_lb, num_threads=1)
            dt_1t = time.perf_counter_ns() - t0

            core.fill_log_blocks(lb_mt, 0x20261123)
            fn(lb_mt, iters=5, num_threads=max_cpus, batch_chunks=32)
            core.fill_log_blocks(lb_mt, 0x20261123)
            t1 = time.perf_counter_ns()
            fn(lb_mt, iters=iters_lb, num_threads=max_cpus, batch_chunks=32)
            dt_mt = time.perf_counter_ns() - t1

            assert np.array_equal(lb_ref, lb_mt)
            if fn is core.log_dual_step:
                assert np.array_equal(lb_mt["res_re"], lb_mt["v_re"])
                assert np.array_equal(lb_mt["res_im"], lb_mt["v_im"])
                assert np.array_equal(lb_mt["unb_theta"], lb_mt["se_theta"])

            g_1t = (n_cells * iters_lb) / dt_1t
            g_mt = (n_cells * iters_lb) / dt_mt
            print(f"           {label:24s} | 1T={g_1t:5.2f} G/s | {max_cpus}T={g_mt:5.2f} G/s ({g_mt / g_1t:.2f}x, 0-bit OK)")

    print("\n>>> ALL V2 PYTHON NUMPY ZERO-COPY VERIFICATION TESTS PASSED <<<")


if __name__ == "__main__":
    main()