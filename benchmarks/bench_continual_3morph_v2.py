#!/usr/bin/env python3
"""Continual Learning with Zero Catastrophic Forgetting via 2-Morphism H1 Detection
& 2-Tier 3-Morphism Autonomous Rewriting on 256-D x 64-Layer Reversible Ladder."""

from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np

from lifting_core_v2 import (
    LiftingCoreV2,
    C16_DTYPE,
    CHART_BLOCK8_DTYPE,
    alloc_aligned_ndarray,
)
from bench_ladder256_v2 import Ladder256Learner


# ============================================================================
# 1. 0-bit 完全可逆 3-Shear リフティング複素回転セル (Tier-1 TwistedSection)
# ============================================================================
def _mul_q14_sym(coef: int, x: np.ndarray) -> np.ndarray:
    prod = np.int64(coef) * x.astype(np.int64)
    q = (np.abs(prod) + 8192) >> 14
    return np.where(prod < 0, -q, q).astype(np.int32)


def _mod16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int32) + b) & 0xFFFF).astype(np.uint16).view(np.int16)


def _mod16_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int32) - b) & 0xFFFF).astype(np.uint16).view(np.int16)


def _mod16_neg(a: np.ndarray) -> np.ndarray:
    return ((-a.astype(np.int32)) & 0xFFFF).astype(np.uint16).view(np.int16)


class ThreeShearGaugeCell:
    """0-bit exact bijective complex rotation on (Z/2^16Z)^2 for arbitrary theta_u16."""

    def __init__(self, theta_u16: int):
        self.theta_u16 = int(theta_u16) & 0xFFFF
        t_signed = self.theta_u16 - 65536 if self.theta_u16 >= 32768 else self.theta_u16
        if abs(t_signed) > 16384:
            self.do_flip = True
            phi_u16 = (self.theta_u16 + 32768) & 0xFFFF
            phi_s = phi_u16 - 65536 if phi_u16 >= 32768 else phi_u16
        else:
            self.do_flip = False
            phi_s = t_signed

        phi_rad = phi_s * (2.0 * np.pi / 65536.0)
        self.a_q14 = int(np.rint(16384.0 * np.tan(0.5 * phi_rad)))
        self.b_q14 = int(np.rint(16384.0 * np.sin(phi_rad)))

    def _rot_fwd(self, arr: np.ndarray) -> None:
        re = arr["re"]
        im = arr["im"]
        if self.do_flip:
            re = _mod16_neg(re)
            im = _mod16_neg(im)
        if self.a_q14 != 0 or self.b_q14 != 0:
            re = _mod16_sub(re, _mul_q14_sym(self.a_q14, im))
            im = _mod16_add(im, _mul_q14_sym(self.b_q14, re))
            re = _mod16_sub(re, _mul_q14_sym(self.a_q14, im))
        arr["re"] = re
        arr["im"] = im

    def _rot_inv(self, arr: np.ndarray) -> None:
        re = arr["re"]
        im = arr["im"]
        if self.a_q14 != 0 or self.b_q14 != 0:
            re = _mod16_add(re, _mul_q14_sym(self.a_q14, im))
            im = _mod16_sub(im, _mul_q14_sym(self.b_q14, re))
            re = _mod16_add(re, _mul_q14_sym(self.a_q14, im))
        if self.do_flip:
            re = _mod16_neg(re)
            im = _mod16_neg(im)
        arr["re"] = re
        arr["im"] = im

    def forward(self, s: np.ndarray, v: np.ndarray) -> None:
        self._rot_fwd(s)
        self._rot_fwd(v)

    def inverse(self, s: np.ndarray, v: np.ndarray) -> None:
        self._rot_inv(s)
        self._rot_inv(v)


# ============================================================================
# 2. 継続学習タスク定義 & 4 タスク FHRR 重畳ロール XOR ベンチマーク生成
# ============================================================================
@dataclass
class ContinualTask:
    ctx_id: int
    name: str
    target_angle_deg: float
    tr_s: np.ndarray
    tr_v: np.ndarray
    tr_ys: np.ndarray
    tr_yv: np.ndarray
    tr_lbl: np.ndarray
    te_s: np.ndarray
    te_v: np.ndarray
    te_lbl: np.ndarray

    @property
    def axis(self) -> complex:
        rad = np.deg2rad(self.target_angle_deg)
        return complex(np.cos(rad), np.sin(rad))


def build_4task_stream(rng: np.random.Generator, dim: int = 256,
                       n_tr: int = 256, n_te: int = 256) -> list[ContinualTask]:
    def make_carriers():
        a1 = rng.uniform(0.0, 2.0 * np.pi, size=dim)
        a2 = rng.uniform(0.0, 2.0 * np.pi, size=dim)
        return (
            3000.0 * np.cos(a1), 3000.0 * np.sin(a1),
            3000.0 * np.cos(a2), 3000.0 * np.sin(a2),
        )

    c12 = make_carriers()
    c34 = make_carriers()  # タスク 0..2 と完全直交する新概念多様体

    def gen_split(carriers, n_samp: int, parity_sign: float, target_deg: float):
        c1r, c1i, c2r, c2i = carriers
        z1 = rng.choice([-1.0, 1.0], size=(n_samp, 1))
        z2 = rng.choice([-1.0, 1.0], size=(n_samp, 1))
        y_pm = (parity_sign * z1 * z2).reshape(-1)
        re = z1 * c1r + z2 * c2r + rng.normal(0.0, 600.0, size=(n_samp, dim))
        im = z1 * c1i + z2 * c2i + rng.normal(0.0, 600.0, size=(n_samp, dim))

        xs = np.empty((n_samp, dim), dtype=C16_DTYPE)
        xs["re"] = np.clip(np.rint(re), -12000, 12000).astype(np.int16)
        xs["im"] = np.clip(np.rint(im), -12000, 12000).astype(np.int16)
        xv = xs.copy()

        rad = np.deg2rad(target_deg)
        ys = np.empty((n_samp, dim), dtype=C16_DTYPE)
        ys["re"] = np.rint(y_pm[:, None] * 3072.0 * np.cos(rad)).astype(np.int16)
        ys["im"] = np.rint(y_pm[:, None] * 3072.0 * np.sin(rad)).astype(np.int16)
        yv = ys.copy()
        return xs, xv, ys, yv, (y_pm > 0.0)

    specs = [
        (0, "T0: 基底多様体 (C1,C2) XOR  0°", c12, +1.0,   0.0),
        (1, "T1: メビウスπ反転 (C1,C2) 180°", c12, -1.0,   0.0),
        (2, "T2: 任意位相ゲージ (C1,C2) 105°", c12, +1.0, 105.0),
        (3, "T3: 直交新多様体 (C3,C4) XOR  0°", c34, +1.0,   0.0),
    ]
    tasks = []
    for cid, name, carr, psign, deg in specs:
        tr_s, tr_v, tr_ys, tr_yv, tr_lbl = gen_split(carr, n_tr, psign, deg)
        te_s, te_v, _, _, te_lbl = gen_split(carr, n_te, psign, deg)
        tasks.append(ContinualTask(cid, name, deg, tr_s, tr_v, tr_ys, tr_yv, tr_lbl, te_s, te_v, te_lbl))
    return tasks


def eval_task_accuracy(ps: np.ndarray, task: ContinualTask) -> float:
    s_bundle = np.mean(ps["re"].astype(np.float64), axis=1) + 1j * np.mean(ps["im"].astype(np.float64), axis=1)
    proj = np.real(np.conj(task.axis) * s_bundle)
    return 100.0 * float(np.mean((proj > 0.0) == task.te_lbl))


# ============================================================================
# 3. 高次圏 3-射自律書き換え継続学習エンジン (ContinualLadder256)
# ============================================================================
@dataclass
class ContextSection:
    tier: str  # "BASE", "TIER1_GAUGE", "TIER2_SUBLADDER"
    coherence: float
    theta_h1_u16: int
    rel_res: float = 0.0
    gauge: ThreeShearGaugeCell | None = None
    subladder: Ladder256Learner | None = None
    adapt_ms: float = 0.0
    extra_bytes: int = 0


class ContinualLadder256:
    def __init__(self, core: LiftingCoreV2, dim: int = 256, base_depth: int = 64, sub_depth: int = 32):
        self.core = core
        self.dim = dim
        self.base_depth = base_depth
        self.sub_depth = sub_depth
        self.base = Ladder256Learner(core, dim=dim, depth=base_depth)
        self.base.modes[:] = np.arange(base_depth, dtype=np.uint8) & 3
        self.base_locked = False
        self.sections: dict[int, ContextSection] = {}
        self._ch_probe = alloc_aligned_ndarray(1, CHART_BLOCK8_DTYPE)

    def _probe_h1_holonomy(self, base_s: np.ndarray, tgt_s: np.ndarray) -> tuple[float, int, float]:
        """X-Spider 帯域集約 -> liblifting_core_v2.dylib 純整数 NEON Cart->Log で H1 位相角抽出"""
        s_b = np.mean(base_s["re"].astype(np.float64), axis=1) + 1j * np.mean(base_s["im"].astype(np.float64), axis=1)
        y_b = np.mean(tgt_s["re"].astype(np.float64), axis=1) + 1j * np.mean(tgt_s["im"].astype(np.float64), axis=1)

        z_h1 = complex(np.sum(y_b * np.conj(s_b)))
        denom = float(np.sum(np.abs(y_b) * np.abs(s_b))) + 1e-12
        coherence = float(abs(z_h1) / denom)

        scale = max(abs(z_h1.real), abs(z_h1.imag), 1e-12)
        self._ch_probe["re"][:] = 0
        self._ch_probe["im"][:] = 0
        self._ch_probe["re"][0, 0] = int(np.clip(np.rint(16384.0 * z_h1.real / scale), -32767, 32767))
        self._ch_probe["im"][0, 0] = int(np.clip(np.rint(16384.0 * z_h1.imag / scale), -32767, 32767))
        self.core.cart_to_log(self._ch_probe, iters=1, num_threads=1)
        theta_h1 = int(self._ch_probe["theta"][0, 0])
        rot = np.exp(1j * (2.0 * np.pi * theta_h1 / 65536.0))
        rel_res = float(np.linalg.norm(y_b - rot * s_b) / (np.linalg.norm(y_b) + 1e-12))
        return coherence, theta_h1, rel_res

    def learn_task(self, task: ContinualTask, num_threads: int) -> ContextSection:
        t0 = time.perf_counter_ns()

        if not self.base_locked:
            for _ in range(40):
                self.base.train_epoch(
                    task.tr_s, task.tr_v, task.tr_ys, task.tr_yv,
                    lr_shift=6, max_rho=-768, learn_bias=True, num_threads=num_threads,
                )
            self.base_locked = True
            sec = ContextSection(
                tier="BASE",
                coherence=1.0,
                theta_h1_u16=0,
                adapt_ms=(time.perf_counter_ns() - t0) / 1e6,
                extra_bytes=0,
            )
            self.sections[task.ctx_id] = sec
            return sec

        # ロック済み基底 F_base へ読み取り専用で順伝播し、2-射ホロノミー障害を自律診断
        bs = task.tr_s.copy()
        bv = task.tr_v.copy()
        self.base.forward(bs, bv, num_threads=num_threads)
        coherence, theta_h1, rel_res = self._probe_h1_holonomy(bs, task.tr_ys)
        t_signed = theta_h1 - 65536 if theta_h1 >= 32768 else theta_h1

        if coherence > 0.85 and rel_res < 0.35 and abs(t_signed) < 1024:
            # H1 = 0 (平坦): 書き換え不要
            sec = ContextSection(
                tier="BASE",
                coherence=coherence,
                theta_h1_u16=theta_h1,
                rel_res=rel_res,
                adapt_ms=(time.perf_counter_ns() - t0) / 1e6,
                extra_bytes=0,
            )
        elif coherence > 0.85 and rel_res < 0.35:
            # Tier-1 3-射自律書き換え: 可換 H1 ゲージ障害 -> 4-byte 0-bit 可逆 3-Shear セル即応生成
            sec = ContextSection(
                tier="TIER1_GAUGE",
                coherence=coherence,
                theta_h1_u16=theta_h1,
                rel_res=rel_res,
                gauge=ThreeShearGaugeCell(theta_h1),
                adapt_ms=(time.perf_counter_ns() - t0) / 1e6,
                extra_bytes=4,
            )
        else:
            # Tier-2 3-射自律書き換え: 非可換新多様体障害 -> 32 層可逆サブ梯子を直列スポーン・局所学習
            sub = Ladder256Learner(self.core, dim=self.dim, depth=self.sub_depth)
            sub.modes[:] = np.arange(self.sub_depth, dtype=np.uint8) & 3
            for _ in range(40):
                sub.train_epoch(
                    bs, bv, task.tr_ys, task.tr_yv,
                    lr_shift=5, max_rho=-640, learn_bias=True, num_threads=num_threads,
                )
            sec = ContextSection(
                tier="TIER2_SUBLADDER",
                coherence=coherence,
                theta_h1_u16=theta_h1,
                rel_res=rel_res,
                subladder=sub,
                adapt_ms=(time.perf_counter_ns() - t0) / 1e6,
                extra_bytes=sub.w.nbytes + sub.b.nbytes,
            )

        self.sections[task.ctx_id] = sec
        return sec

    def forward_ctx(self, ctx_id: int, s: np.ndarray, v: np.ndarray, num_threads: int) -> None:
        self.base.forward(s, v, num_threads=num_threads)
        sec = self.sections[ctx_id]
        if sec.gauge is not None:
            sec.gauge.forward(s, v)
        if sec.subladder is not None:
            sec.subladder.forward(s, v, num_threads=num_threads)

    def inverse_ctx(self, ctx_id: int, s: np.ndarray, v: np.ndarray, num_threads: int) -> None:
        sec = self.sections[ctx_id]
        if sec.subladder is not None:
            sec.subladder.inverse(s, v, num_threads=num_threads)
        if sec.gauge is not None:
            sec.gauge.inverse(s, v)
        self.base.inverse(s, v, num_threads=num_threads)

    def total_depth_ctx(self, ctx_id: int) -> int:
        sec = self.sections[ctx_id]
        return self.base_depth + (self.sub_depth if sec.subladder is not None else 0)


# ============================================================================
# 4. 対照検証メイン (Naive Fine-Tuning vs 3-Morphism Autonomous Rewrite)
# ============================================================================
def main() -> None:
    core = LiftingCoreV2()
    max_cpus = core.cpu_count
    dim = 256

    # ワーカースレッドプール初期化
    pool_holder = Ladder256Learner(core, dim=dim, depth=64)
    pool_holder.mt_init(max_cpus)

    print("=== 256D x 64 層梯子: 3-射自律書き換えによる破滅的忘却ゼロ継続学習実証 ===")
    print(f"モデル = 256D x 64 層基底 + 自律 3-射 (Tier-1 3-Shear 4B / Tier-2 32層サブ梯子) | 論理CPU = {max_cpus}\n")

    rng = np.random.default_rng(20261123)
    tasks = build_4task_stream(rng, dim=dim, n_tr=256, n_te=256)

    try:
        # ------------------------------------------------------------------
        # [Baseline] 3-射書き換えなし: 静的 64 層梯子の逐次ファインチューニング
        # ------------------------------------------------------------------
        naive = Ladder256Learner(core, dim=dim, depth=64)
        naive.modes[:] = np.arange(64, dtype=np.uint8) & 3
        naive_mat = np.zeros((4, 4), dtype=np.float64)

        for stage, tk in enumerate(tasks):
            for _ in range(40):
                naive.train_epoch(
                    tk.tr_s, tk.tr_v, tk.tr_ys, tk.tr_yv,
                    lr_shift=6, max_rho=-768, learn_bias=True, num_threads=max_cpus,
                )
            for ev_i in range(stage + 1):
                ps, pv = tasks[ev_i].te_s.copy(), tasks[ev_i].te_v.copy()
                naive.forward(ps, pv, num_threads=max_cpus)
                naive_mat[stage, ev_i] = eval_task_accuracy(ps, tasks[ev_i])

        # ------------------------------------------------------------------
        # [Proposed] 2-射ホロノミー自律検出 + 2 段階 3-射書き換え継続学習
        # ------------------------------------------------------------------
        cl = ContinualLadder256(core, dim=dim, base_depth=64, sub_depth=32)
        cl_mat = np.zeros((4, 4), dtype=np.float64)
        saved_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        print("[Phase 1] 各タスク流入時の 2-射ホロノミー自律診断 & 3-射トポロジー書き換えログ:")
        for stage, tk in enumerate(tasks):
            sec = cl.learn_task(tk, num_threads=max_cpus)
            deg_est = 360.0 * sec.theta_h1_u16 / 65536.0
            print(f"  {tk.name:31s} | κ={sec.coherence:.3f}, r={sec.rel_res:.2f}, θ_H1={deg_est:6.2f}° (0x{sec.theta_h1_u16:04X}) "
                  f"-> {sec.tier:15s} (+{sec.extra_bytes:6d} B, {sec.adapt_ms:6.1f} ms)")

            for ev_i in range(stage + 1):
                ps, pv = tasks[ev_i].te_s.copy(), tasks[ev_i].te_v.copy()
                cl.forward_ctx(ev_i, ps, pv, num_threads=max_cpus)
                cl_mat[stage, ev_i] = eval_task_accuracy(ps, tasks[ev_i])
                if ev_i == stage:
                    saved_outputs[ev_i] = (ps.copy(), pv.copy())
                else:
                    # 過去タスクの出力が 1 ビットも変化していないこと (0-bit 不変性) を検証
                    assert np.array_equal(ps, saved_outputs[ev_i][0])
                    assert np.array_equal(pv, saved_outputs[ev_i][1])
        print()

        # ------------------------------------------------------------------
        # [Phase 2] 対照結果マトリクス & 忘却率比較
        # ------------------------------------------------------------------
        print("[Phase 2] 4 タスク逐次学習後の未知テスト正答率マトリクス (T0 -> T1 -> T2 -> T3):")
        print("  [Naive 静的64層 (3-射なし)]                |  [3-射自律書き換え (本提案)]")
        print("  段階  |   T0     T1     T2     T3   | 平均  |  段階  |   T0     T1     T2     T3   | 平均")
        print("  ------+-----------------------------+-------+--------+-----------------------------+------")
        for st in range(4):
            n_row = " ".join(f"{naive_mat[st, j]:5.1f}%" if j <= st else "   -  " for j in range(4))
            c_row = " ".join(f"{cl_mat[st, j]:5.1f}%" if j <= st else "   -  " for j in range(4))
            n_avg = float(np.mean(naive_mat[st, : st + 1]))
            c_avg = float(np.mean(cl_mat[st, : st + 1]))
            print(f"  後 T{st} | {n_row} | {n_avg:5.1f}% |  後 T{st} | {c_row} | {c_avg:5.1f}%")

        naive_forgetting = float(np.mean([naive_mat[j, j] - naive_mat[3, j] for j in range(3)]))
        cl_forgetting = float(np.mean([cl_mat[j, j] - cl_mat[3, j] for j in range(3)]))
        print(f"\n  >> 過去 3 タスク平均忘却量: Naive = {naive_forgetting:+5.1f}%  vs  3-射自律書き換え = {cl_forgetting:+5.1f}% (0-bit 不変)")

        # ------------------------------------------------------------------
        # [Phase 3] 全 4 タスク合成パイプライン (最大 96 層) の 0-bit 厳密可逆性検証
        # ------------------------------------------------------------------
        print("\n[Phase 3] 継続学習完了後の全タスク 0-bit 厳密可逆性検証 (μ ∘ Δ = id):")
        for tk in tasks:
            ps, pv = tk.te_s.copy(), tk.te_v.copy()
            cl.forward_ctx(tk.ctx_id, ps, pv, num_threads=max_cpus)
            cl.inverse_ctx(tk.ctx_id, ps, pv, num_threads=max_cpus)
            rev_ok = np.array_equal(ps, tk.te_s) and np.array_equal(pv, tk.te_v)
            assert rev_ok, f"Reversibility failed on Task {tk.ctx_id}"
            sec = cl.sections[tk.ctx_id]
            print(f"  Task {tk.ctx_id} ({sec.tier:15s}, 総深さ={cl.total_depth_ctx(tk.ctx_id):2d}層): "
                  f"正答率={cl_mat[3, tk.ctx_id]:5.1f}% | 256サンプル×256D 0-bit 完全復元 OK")

        assert cl.sections[1].tier == "TIER1_GAUGE"
        assert cl.sections[2].tier == "TIER1_GAUGE"
        assert cl.sections[3].tier == "TIER2_SUBLADDER"
        assert cl_forgetting == 0.0
        assert float(np.min(cl_mat[3, :])) >= 98.0
        assert float(np.mean(cl_mat[3, :])) > float(np.mean(naive_mat[3, :])) + 20.0
    finally:
        pool_holder.mt_shutdown()

    print("\n>>> ALL ASSERTIONS PASSED: 3-射自律書き換えによる破滅的忘却ゼロ & 96 層 0-bit 可逆性を実証 <<<")


if __name__ == "__main__":
    main()