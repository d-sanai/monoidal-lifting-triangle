#!/usr/bin/env python3
"""256-D x 64-Layer Backprop-Free O(1)-Memory Local 2-Morphism Learner via liblifting_core_v2.dylib."""

from __future__ import annotations

import ctypes
import time
import numpy as np

from lifting_core_v2 import (
    LiftingCoreV2,
    C16_DTYPE,
    LOGC16_DTYPE,
    LOG_ZERO,
)


class Ladder256Learner:
    """Zero-copy NumPy interface to 256-D x 64-layer reversible FHRR O(1) learner."""

    def __init__(self, core: LiftingCoreV2, dim: int = 256, depth: int = 64):
        assert dim % 8 == 0 and dim <= 512
        self.core = core
        self.lib = core.lib
        self.dim = dim
        self.depth = depth
        self._bind()

        self.w = np.zeros((depth, dim), dtype=LOGC16_DTYPE)
        self.w["rho"] = LOG_ZERO
        self.b = np.zeros((depth, dim), dtype=C16_DTYPE)
        self.err_fb_re = np.zeros((depth, dim), dtype=np.int32)
        self.err_fb_im = np.zeros((depth, dim), dtype=np.int32)
        self.modes = np.zeros(depth, dtype=np.uint8)
        self.shifts = np.array([1 << (L & 7) for L in range(depth)], dtype=np.int32)

        self._work_n = 0
        self._work_up_s = None
        self._work_up_v = None
        self._work_dn_s = None
        self._work_dn_v = None

    def _bind(self) -> None:
        L = self.lib
        L.lc_v2_l256_mt_init.argtypes = [ctypes.c_int]
        L.lc_v2_l256_mt_init.restype = ctypes.c_int
        L.lc_v2_l256_mt_shutdown.argtypes = []

        L.lc_v2_l256_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ]
        L.lc_v2_l256_forward.restype = ctypes.c_int

        L.lc_v2_l256_inverse.argtypes = L.lc_v2_l256_forward.argtypes
        L.lc_v2_l256_inverse.restype = ctypes.c_int

        L.lc_v2_l256_train_epoch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint8, ctypes.c_int16, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        L.lc_v2_l256_train_epoch.restype = ctypes.c_int

    def mt_init(self, max_workers: int = 0) -> int:
        n = self.lib.lc_v2_l256_mt_init(max_workers)
        if n < 0:
            raise RuntimeError(f"lc_v2_l256_mt_init failed ({n})")
        return n

    def mt_shutdown(self) -> None:
        self.lib.lc_v2_l256_mt_shutdown()

    def _ensure_work(self, n_samples: int) -> None:
        if self._work_n != n_samples:
            self._work_n = n_samples
            self._work_up_s = np.empty((n_samples, self.dim), dtype=C16_DTYPE)
            self._work_up_v = np.empty((n_samples, self.dim), dtype=C16_DTYPE)
            self._work_dn_s = np.empty((n_samples, self.dim), dtype=C16_DTYPE)
            self._work_dn_v = np.empty((n_samples, self.dim), dtype=C16_DTYPE)

    def forward(self, s: np.ndarray, v: np.ndarray, num_threads: int = 1) -> None:
        assert s.dtype == C16_DTYPE and v.dtype == C16_DTYPE
        assert s.shape == v.shape and s.shape[1] == self.dim
        assert s.flags["C_CONTIGUOUS"] and v.flags["C_CONTIGUOUS"]
        rc = self.lib.lc_v2_l256_forward(
            s.ctypes.data, v.ctypes.data, s.shape[0], self.dim, self.depth,
            self.w.ctypes.data, self.b.ctypes.data,
            self.modes.ctypes.data, self.shifts.ctypes.data, num_threads,
        )
        if rc != 0:
            raise RuntimeError(f"lc_v2_l256_forward failed ({rc})")

    def inverse(self, s: np.ndarray, v: np.ndarray, num_threads: int = 1) -> None:
        assert s.dtype == C16_DTYPE and v.dtype == C16_DTYPE
        assert s.shape == v.shape and s.shape[1] == self.dim
        assert s.flags["C_CONTIGUOUS"] and v.flags["C_CONTIGUOUS"]
        rc = self.lib.lc_v2_l256_inverse(
            s.ctypes.data, v.ctypes.data, s.shape[0], self.dim, self.depth,
            self.w.ctypes.data, self.b.ctypes.data,
            self.modes.ctypes.data, self.shifts.ctypes.data, num_threads,
        )
        if rc != 0:
            raise RuntimeError(f"lc_v2_l256_inverse failed ({rc})")

    def train_epoch(self, in_s: np.ndarray, in_v: np.ndarray,
                    tgt_s: np.ndarray, tgt_v: np.ndarray,
                    lr_shift: int = 6, max_rho: int = -768,
                    learn_bias: bool = False, num_threads: int = 1) -> float:
        n_samples = in_s.shape[0]
        self._ensure_work(n_samples)
        rmse = ctypes.c_double()
        rc = self.lib.lc_v2_l256_train_epoch(
            in_s.ctypes.data, in_v.ctypes.data, tgt_s.ctypes.data, tgt_v.ctypes.data,
            self._work_up_s.ctypes.data, self._work_up_v.ctypes.data,
            self._work_dn_s.ctypes.data, self._work_dn_v.ctypes.data,
            n_samples, self.dim, self.depth,
            self.w.ctypes.data, self.b.ctypes.data,
            self.err_fb_re.ctypes.data, self.err_fb_im.ctypes.data,
            self.modes.ctypes.data, self.shifts.ctypes.data,
            lr_shift, max_rho, int(learn_bias), num_threads, ctypes.byref(rmse),
        )
        if rc != 0:
            raise RuntimeError(f"lc_v2_l256_train_epoch failed ({rc})")
        return float(rmse.value)


def make_c16_array(rng: np.random.Generator, shape: tuple[int, int], lo: int, hi: int) -> np.ndarray:
    arr = np.empty(shape, dtype=C16_DTYPE)
    arr["re"] = rng.integers(lo, hi, size=shape, dtype=np.int16)
    arr["im"] = rng.integers(lo, hi, size=shape, dtype=np.int16)
    return arr


def calc_rmse(ps: np.ndarray, pv: np.ndarray, ts: np.ndarray, tv: np.ndarray) -> float:
    e0 = ps["re"].astype(np.float64) - ts["re"].astype(np.float64)
    e1 = ps["im"].astype(np.float64) - ts["im"].astype(np.float64)
    e2 = pv["re"].astype(np.float64) - tv["re"].astype(np.float64)
    e3 = pv["im"].astype(np.float64) - tv["im"].astype(np.float64)
    return float(np.sqrt(np.mean(e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3) / 4.0))


def main() -> None:
    core = LiftingCoreV2()
    max_cpus = core.cpu_count
    dim, depth = 256, 64

    learner = Ladder256Learner(core, dim=dim, depth=depth)
    learner.mt_init(max_cpus)

    print("=== 256 次元 FHRR x 64 層: 逆伝播ゼロ・O(1) メモリ並列学習 (liblifting_core_v2.dylib) ===")
    print(f"モデル構成 = {dim}D x {depth} 層 ({dim * depth:,} 複素対数重み, バタフライ循環結合) | "
          f"NEON={core.has_neon} | 論理CPU={max_cpus}\n")

    rng = np.random.default_rng(20261123)

    try:
        # ------------------------------------------------------------------
        # [Test 1] 256-D x 64 層バタフライ結合往復の 0-bit 可逆性 & 1T vs MT スループット
        # ------------------------------------------------------------------
        n_bench = 1024
        learner.modes[:] = np.arange(depth, dtype=np.uint8) & 3
        learner.w["rho"] = rng.integers(-1280, -768, size=(depth, dim), dtype=np.int16)
        learner.w["theta"] = rng.integers(0, 65536, size=(depth, dim), dtype=np.uint16)
        learner.b["re"] = rng.integers(-256, 256, size=(depth, dim), dtype=np.int16)
        learner.b["im"] = rng.integers(-256, 256, size=(depth, dim), dtype=np.int16)

        s0 = make_c16_array(rng, (n_bench, dim), -32768, 32768)
        v0 = make_c16_array(rng, (n_bench, dim), -32768, 32768)
        s0["re"][0, 0] = s0["im"][0, 0] = 0
        v0["re"][1, 1] = v0["im"][1, 1] = -32768

        s_1t, v_1t = s0.copy(), v0.copy()
        t0 = time.perf_counter_ns()
        learner.forward(s_1t, v_1t, num_threads=1)
        mid_1t_s, mid_1t_v = s_1t.copy(), v_1t.copy()
        learner.inverse(s_1t, v_1t, num_threads=1)
        dt_1t_ms = (time.perf_counter_ns() - t0) / 1e6

        s_mt, v_mt = s0.copy(), v0.copy()
        t1 = time.perf_counter_ns()
        learner.forward(s_mt, v_mt, num_threads=max_cpus)
        mid_mt_s, mid_mt_v = s_mt.copy(), v_mt.copy()
        learner.inverse(s_mt, v_mt, num_threads=max_cpus)
        dt_mt_ms = (time.perf_counter_ns() - t1) / 1e6

        tot_cell_layers = n_bench * dim * depth * 2
        print(f"  [Test 1] 256D x 64 層バタフライ閉ループ往復 ({n_bench} samples = {tot_cell_layers / 1e6:.2f}M cell-layers):")
        print(f"           1T = {dt_1t_ms:6.2f} ms ({tot_cell_layers / (dt_1t_ms * 1e3):6.1f} M/s) | "
              f"{max_cpus}T = {dt_mt_ms:5.2f} ms ({tot_cell_layers / (dt_mt_ms * 1e3):6.1f} M/s, {dt_1t_ms / dt_mt_ms:.2f}x)")
        assert np.array_equal(mid_1t_s, mid_mt_s) and np.array_equal(mid_1t_v, mid_mt_v)
        assert np.array_equal(s_mt, s0) and np.array_equal(v_mt, v0)
        print(f"           1T=={max_cpus}T 中間状態 0-bit 一致 & μ^64∘Δ^64=id 完全復元: OK\n")

        # ------------------------------------------------------------------
        # [Test 2] 全 16,384 重み同時・逆伝播ゼロ O(1) 学習 & 1T == 10T 0-bit 決定論検証
        # ------------------------------------------------------------------
        teacher = Ladder256Learner(core, dim=dim, depth=depth)
        teacher.modes[:] = 0
        teacher.w["rho"] = rng.integers(-1664, -1280, size=(depth, dim), dtype=np.int16)
        teacher.w["theta"] = rng.integers(0, 65536, size=(depth, dim), dtype=np.uint16)

        n_tr, n_te, epochs = 512, 128, 80
        tr_s = make_c16_array(rng, (n_tr, dim), -4096, 4097)
        tr_v = make_c16_array(rng, (n_tr, dim), -4096, 4097)
        te_s = make_c16_array(rng, (n_te, dim), -4096, 4097)
        te_v = make_c16_array(rng, (n_te, dim), -4096, 4097)

        tr_ys, tr_yv = tr_s.copy(), tr_v.copy()
        te_ys, te_yv = te_s.copy(), te_v.copy()
        teacher.forward(tr_ys, tr_yv, num_threads=max_cpus)
        teacher.forward(te_ys, te_yv, num_threads=max_cpus)

        # 微小な決定論的対称性破り初期重み (|w| ≈ 0.002) を 1T / MT 共通で使用
        init_w = np.empty((depth, dim), dtype=LOGC16_DTYPE)
        init_w["rho"] = -2304
        init_w["theta"] = rng.integers(0, 65536, size=(depth, dim), dtype=np.uint16)

        stu_1t = Ladder256Learner(core, dim=dim, depth=depth)
        stu_mt = Ladder256Learner(core, dim=dim, depth=depth)
        stu_1t.w[:] = init_w
        stu_mt.w[:] = init_w

        t0 = time.perf_counter_ns()
        for _ in range(5):
            stu_1t.train_epoch(tr_s, tr_v, tr_ys, tr_yv, lr_shift=6, max_rho=-768, num_threads=1)
        dt_tr_1t = (time.perf_counter_ns() - t0) / 1e6

        t1 = time.perf_counter_ns()
        for _ in range(5):
            stu_mt.train_epoch(tr_s, tr_v, tr_ys, tr_yv, lr_shift=6, max_rho=-768, num_threads=max_cpus)
        dt_tr_mt = (time.perf_counter_ns() - t1) / 1e6

        assert np.array_equal(stu_1t.w, stu_mt.w)
        assert np.array_equal(stu_1t.err_fb_re, stu_mt.err_fb_re)
        assert np.array_equal(stu_1t.err_fb_im, stu_mt.err_fb_im)

        stu_full = Ladder256Learner(core, dim=dim, depth=depth)
        stu_full.w[:] = init_w
        t_all0 = time.perf_counter_ns()
        rmse_0 = 0.0
        rmse_end = 0.0
        for ep in range(epochs):
            r = stu_full.train_epoch(tr_s, tr_v, tr_ys, tr_yv, lr_shift=6, max_rho=-768, num_threads=max_cpus)
            if ep == 0:
                rmse_0 = r
            rmse_end = r
        dt_full_ms = (time.perf_counter_ns() - t_all0) / 1e6

        ps, pv = te_s.copy(), te_v.copy()
        stu_full.forward(ps, pv, num_threads=max_cpus)
        test_rmse = calc_rmse(ps, pv, te_ys, te_yv)
        stu_full.inverse(ps, pv, num_threads=max_cpus)
        rev_ok = np.array_equal(ps, te_s) and np.array_equal(pv, te_v)

        print(f"  [Test 2] 256D x 64 層 Teacher-Student 全 16,384 重み同時・逆伝播ゼロ O(1) 学習:")
        print(f"           1T vs {max_cpus}T (5 ep, N={n_tr}): 1T={dt_tr_1t:6.1f} ms | {max_cpus}T={dt_tr_mt:5.1f} ms "
              f"({dt_tr_1t / dt_tr_mt:.2f}x) -> 全 16,384 重み 0-bit 完全一致 OK")
        print(f"           {epochs} epochs ({dt_full_ms:.1f} ms, {dt_full_ms / epochs:.2f} ms/ep): "
              f"RMSE {rmse_0:6.2f} -> {rmse_end:5.2f} (-{100.0 * (1.0 - rmse_end / rmse_0):.1f}%) | "
              f"未知テスト RMSE={test_rmse:5.2f} (-{100.0 * (1.0 - test_rmse / rmse_0):.1f}%) | "
              f"0-bit可逆={'OK' if rev_ok else 'FAIL'}\n")
        assert rev_ok
        assert test_rmse < rmse_0 * 0.15

        # ------------------------------------------------------------------
        # [Test 3] 256-D FHRR 重畳ロール XOR (線形分離不能 50% タスク) の非線形逆伝播ゼロ分類
        # ------------------------------------------------------------------
        ang1 = rng.uniform(0.0, 2.0 * np.pi, size=dim)
        ang2 = rng.uniform(0.0, 2.0 * np.pi, size=dim)
        c1_re, c1_im = 3000.0 * np.cos(ang1), 3000.0 * np.sin(ang1)
        c2_re, c2_im = 3000.0 * np.cos(ang2), 3000.0 * np.sin(ang2)

        def gen_fhrr_xor_dataset(n_samp: int):
            z1 = rng.choice([-1.0, 1.0], size=(n_samp, 1))
            z2 = rng.choice([-1.0, 1.0], size=(n_samp, 1))
            y = (z1 * z2).reshape(-1)
            re = z1 * c1_re + z2 * c2_re + rng.normal(0.0, 600.0, size=(n_samp, dim))
            im = z1 * c1_im + z2 * c2_im + rng.normal(0.0, 600.0, size=(n_samp, dim))
            xs = np.empty((n_samp, dim), dtype=C16_DTYPE)
            xs["re"] = np.clip(np.rint(re), -12000, 12000).astype(np.int16)
            xs["im"] = np.clip(np.rint(im), -12000, 12000).astype(np.int16)
            xv = xs.copy()
            ys = np.zeros((n_samp, dim), dtype=C16_DTYPE)
            ys["re"] = (y[:, None] * 3072.0).astype(np.int16)
            yv = ys.copy()
            return xs, xv, ys, yv, (y > 0)

        xtr_s, xtr_v, ytr_s, ytr_v, tr_lbl = gen_fhrr_xor_dataset(256)
        xte_s, xte_v, _, _, te_lbl = gen_fhrr_xor_dataset(256)

        Xtr_lin = np.hstack([xtr_s["re"].astype(np.float64), xtr_s["im"].astype(np.float64)])
        Xte_lin = np.hstack([xte_s["re"].astype(np.float64), xte_s["im"].astype(np.float64)])
        y_pm = np.where(tr_lbl, 1.0, -1.0)
        w_lin = np.linalg.solve(Xtr_lin.T @ Xtr_lin + 1e6 * np.eye(2 * dim), Xtr_lin.T @ y_pm)
        acc_linear = 100.0 * float(np.mean((Xte_lin @ w_lin > 0.0) == te_lbl))

        clf = Ladder256Learner(core, dim=dim, depth=depth)
        clf.modes[:] = np.arange(depth, dtype=np.uint8) & 3

        def eval_acc(model: Ladder256Learner, xs: np.ndarray, xv: np.ndarray, lbl: np.ndarray) -> tuple[float, float]:
            ps, pv = xs.copy(), xv.copy()
            model.forward(ps, pv, num_threads=max_cpus)
            bundled_pred = np.mean(ps["re"].astype(np.float64), axis=1) > 0.0
            single_pred = ps["re"][:, 0] > 0
            return 100.0 * float(np.mean(bundled_pred == lbl)), 100.0 * float(np.mean(single_pred == lbl))

        acc_init, _ = eval_acc(clf, xte_s, xte_v, te_lbl)
        t_c0 = time.perf_counter_ns()
        for _ in range(40):
            clf.train_epoch(xtr_s, xtr_v, ytr_s, ytr_v, lr_shift=6, max_rho=-768,
                            learn_bias=True, num_threads=max_cpus)
        dt_clf_ms = (time.perf_counter_ns() - t_c0) / 1e6
        acc_test_256d, acc_test_1d = eval_acc(clf, xte_s, xte_v, te_lbl)

        ps, pv = xte_s.copy(), xte_v.copy()
        clf.forward(ps, pv, num_threads=max_cpus)
        clf.inverse(ps, pv, num_threads=max_cpus)
        clf_rev_ok = np.array_equal(ps, xte_s) and np.array_equal(pv, xte_v)

        print(f"  [Test 3] 256-D FHRR 重畳ロール XOR 分類 (X = z1*C1 + z2*C2 + noise, 40 epochs, {dt_clf_ms:.1f} ms):")
        print(f"           線形リッジ回帰 (512次元) : 未知テスト正答率 = {acc_linear:5.1f}% (線形分離不能)")
        print(f"           64 層 O(1) 局所 2-射学習 : {acc_init:5.1f}% -> {acc_test_256d:5.1f}% (256D集約) / "
              f"{acc_test_1d:5.1f}% (単一チャネル) | 0-bit可逆={'OK' if clf_rev_ok else 'FAIL'}")
        assert clf_rev_ok
        assert acc_test_256d >= 98.0
    finally:
        learner.mt_shutdown()

    print("\n>>> ALL 256-D x 64-LAYER BACKPROP-FREE O(1) MULTI-CORE TESTS PASSED <<<")


if __name__ == "__main__":
    main()