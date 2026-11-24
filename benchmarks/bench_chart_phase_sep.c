/* bench_chart_phase_sep.c — Breaking Phase Gauge Equivalence (θ1, θ2) via Dual-Chart
 * Round-Trip (Log <-> Cart), Orthogonal Bias & 2nd-Harmonic Interference:
 *   [Test 1] Symmetry Breaking U(1) -> C4 (90° Lattice Anisotropy vs Exact 90° Degeneracy)
 *   [Test 2] 50 Random Log->Cart->Log Circuits: Grid-Extinction Z4 Probe -> θ1 mod 90° ID
 *   [Test 3] 50 Random Orthogonal-Bias Circuits: Z1 Interference Probe -> Full 360° (θ1, θ2) ID
 *   [Test 4] 50 Random 2nd-Harmonic (z ⊕ z²) Circuits: Bias-Free Z1 Probe -> Full 360° ID */
#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define LIFTING_CORE_V2_IMPLEMENTATION
#include "lifting_core_v2.h"

static inline uint32_t next_rng(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static inline int32_t abs_phase_err_360(uint16_t a, uint16_t b) {
    int16_t d = (int16_t)((uint16_t)(a - b));
    return (d == -32768) ? 32768 : lc_iabs32((int32_t)d);
}

static inline int32_t abs_phase_err_90(uint16_t a, uint16_t b) {
    /* mod 90° (0x4000) 上の最短角距離: 4 倍角にして mod 360° 距離を取り 4 で割る */
    int16_t d4 = (int16_t)((uint16_t)((uint16_t)(a - b) << 2));
    int32_t ad = (d4 == -32768) ? 32768 : lc_iabs32((int32_t)d4);
    return (ad + 2) >> 2;
}

/* int64 の複素モーメント (re, im) を int16 範囲へ正規化して偏角 uint16_t を抽出 */
static uint16_t moment_to_angle_u16(int64_t re, int64_t im) {
    int64_t ar = (re < 0) ? -re : re;
    int64_t ai = (im < 0) ? -im : im;
    int64_t mx = (ar > ai) ? ar : ai;
    if (mx == 0) return 0u;
    int32_t sx = (int32_t)((re * 16384LL) / mx);
    int32_t sy = (int32_t)((im * 16384LL) / mx);
    return lc_cart_to_log((int16_t)sx, (int16_t)sy).theta;
}

/* ============================================================================
 * 1. ブラックボックス直列 DUT 定義 (3 種の中間構造)
 * ============================================================================ */
typedef struct {
    lc_logc16_t w1;
    lc_logc16_t w2;
    lc_c16_t    bias_cart; /* Test 3 用: 直交実軸バイアス */
} lc_chart_dut_t;

/* (A) 中間 Log -> Cart -> Log 往復のみ */
static inline lc_logc16_t dut_roundtrip_eval(const lc_chart_dut_t *d, lc_logc16_t x) {
    lc_logc16_t z1 = lc_fhrr_bind(d->w1, x);
    lc_c16_t    c  = lc_log_to_cart(z1.rho, z1.theta);
    lc_logc16_t z2 = lc_cart_to_log(c.re, c.im);
    return lc_fhrr_bind(d->w2, z2);
}

/* (B) 中間直交バイアス加算: Log -> Cart -> (⊕ bias) -> Log */
static inline lc_logc16_t dut_bias_eval(const lc_chart_dut_t *d, lc_logc16_t x) {
    lc_logc16_t z1 = lc_fhrr_bind(d->w1, x);
    lc_c16_t    c  = lc_c16_add_mod(lc_log_to_cart(z1.rho, z1.theta), d->bias_cart);
    lc_logc16_t z2 = lc_cart_to_log(c.re, c.im);
    return lc_fhrr_bind(d->w2, z2);
}

/* (C) 中間 2 次調和干渉 (外部バイアスなし): Log -> Cart(z) ⊕ Cart(z²) -> Log */
static inline lc_logc16_t dut_harm2_eval(const lc_chart_dut_t *d, lc_logc16_t x) {
    lc_logc16_t z1  = lc_fhrr_bind(d->w1, x);
    lc_logc16_t z1q = lc_fhrr_bind(z1, z1); /* 2次調和: (2ρ, 2θ) */
    lc_c16_t    c   = lc_c16_add_mod(lc_log_to_cart(z1.rho, z1.theta),
                                     lc_log_to_cart(z1q.rho, z1q.theta));
    lc_logc16_t z2  = lc_cart_to_log(c.re, c.im);
    return lc_fhrr_bind(d->w2, z2);
}

/* 64 角度スイープから総位相 θ_tot = θ1 + θ2 を円周平均でブラインド抽出 */
typedef lc_logc16_t (*dut_fn_t)(const lc_chart_dut_t *, lc_logc16_t);

static uint16_t probe_total_phase(const lc_chart_dut_t *d, dut_fn_t fn, int16_t rho_probe) {
    int64_t sum_c = 0, sum_s = 0;
    for (int k = 0; k < 64; k++) {
        uint16_t th_in = (uint16_t)(k * 1024);
        lc_logc16_t y  = fn(d, (lc_logc16_t){ rho_probe, th_in });
        if (y.rho == LC_LOG_ZERO) continue;
        uint16_t dth = (uint16_t)(y.theta - th_in);
        sum_c += (int64_t)lc_trig_cos_q14(dth);
        sum_s += (int64_t)lc_trig_sin_q14(dth);
    }
    return moment_to_angle_u16(sum_c, sum_s);
}

/* ============================================================================
 * 2. [Test 1] 対称性の段階的破れ: U(1) -> C4 (90° 周期の格子異方性と 4 重縮退)
 * ============================================================================ */
static bool run_test1_u1_to_c4_symmetry_breaking(void) {
    const lc_chart_dut_t base    = { { -512, 0x1234u }, { -512, 0x4321u }, { 0, 0 } };
    const lc_chart_dut_t rot90   = { { -512, (uint16_t)(0x1234u + 0x4000u) },
                                     { -512, (uint16_t)(0x4321u - 0x4000u) }, { 0, 0 } };
    const lc_chart_dut_t rot37   = { { -512, (uint16_t)(0x1234u + 0x1A50u) },
                                     { -512, (uint16_t)(0x4321u - 0x1A50u) }, { 0, 0 } };

    int diff_pure = 0, diff_c4_90 = 0, diff_non90 = 0;
    const int16_t probe_rhos[3] = { -1792, -1280, -256 };

    for (int ri = 0; ri < 3; ri++) {
        int16_t rp = probe_rhos[ri];
        for (int k = 0; k < 256; k++) {
            lc_logc16_t x = { rp, (uint16_t)(k * 256) };
            /* 純粋 Log 直列: 37° シフトでも 0-bit 完全一致 */
            lc_logc16_t p0 = lc_fhrr_bind(base.w2,  lc_fhrr_bind(base.w1,  x));
            lc_logc16_t p3 = lc_fhrr_bind(rot37.w2, lc_fhrr_bind(rot37.w1, x));
            if (p0.rho != p3.rho || p0.theta != p3.theta) diff_pure++;

            /* Log->Cart->Log 往復: 90° シフトは C4 対称で 0-bit 一致、37° シフトは不一致 */
            lc_logc16_t r0 = dut_roundtrip_eval(&base,  x);
            lc_logc16_t r9 = dut_roundtrip_eval(&rot90, x);
            lc_logc16_t r3 = dut_roundtrip_eval(&rot37, x);
            if (r0.rho != r9.rho || r0.theta != r9.theta) diff_c4_90++;
            if (r0.rho != r3.rho || r0.theta != r3.theta) diff_non90++;
        }
    }

    if (diff_pure != 0 || diff_c4_90 != 0 || diff_non90 == 0) {
        fprintf(stderr, "FAIL: Test 1 対称性破れ異常 (pure=%d, c4=%d, non90=%d)\n",
                diff_pure, diff_c4_90, diff_non90);
        return false;
    }

    printf("[Test 1] 位相ゲージ対称性の段階的破れ (U(1) -> C4, 768 プローブ点比較):\n");
    printf("  純粋 Log 直列 (δ=37°)       : 差異 = %3d/768 (U(1) 連続ゲージ対称 -> 観測不能)\n", diff_pure);
    printf("  Log↔Cart 往復 (δ=37°)       : 差異 = %3d/768 (直交格子異方性で U(1) 破れ -> 分離可能!)\n", diff_non90);
    printf("  Log↔Cart 往復 (δ=90°=0x4000): 差異 = %3d/768 (C4 4重縮退が厳密な残存ゲージ同値類)\n\n", diff_c4_90);
    return true;
}

/* ============================================================================
 * 3. [Test 2] 最小非ゼロ格子生存窓 (Grid Extinction) の Z4 モーメントによる θ1 mod 90° 同定
 * ============================================================================ */
static bool probe_roundtrip_theta_mod90(const lc_chart_dut_t *d,
                                        uint16_t *out_th1_m90, uint16_t *out_th2_m90,
                                        int *out_alive_cnt) {
    /* Step 1: 全 int16 域 [-32767, +8192] を 18 歩二分探索し、部分消滅窓 (alive≈128) を 1 LSB 精度で特定 */
    int32_t lo = -32767;
    int32_t hi =   8192;
    int32_t best_rho   = lo;
    int     best_alive = 0;
    int     best_dist  = 1000;

    for (int iter = 0; iter < 18; iter++) {
        int32_t mid = lo + (hi - lo) / 2;
        int alive = 0;
        for (int k = 0; k < 256; k++) {
            lc_logc16_t y = dut_roundtrip_eval(d, (lc_logc16_t){ (int16_t)mid, (uint16_t)(k * 256) });
            if (y.rho != LC_LOG_ZERO) alive++;
        }
        int dist = (alive > 128) ? (alive - 128) : (128 - alive);
        if (alive > 0 && alive < 256 && dist < best_dist) {
            best_dist  = dist;
            best_rho   = mid;
            best_alive = alive;
        }
        if (alive < 128) lo = mid;
        else             hi = mid;
    }

    int64_t z4_re = 0, z4_im = 0;
    if (best_alive > 0) {
        /* 部分消滅窓 (軸方向のみ生存・対角方向は ⊥) の 4 倍角モーメント Z4 */
        for (int k = 0; k < 256; k++) {
            uint16_t th_in = (uint16_t)(k * 256);
            lc_logc16_t y  = dut_roundtrip_eval(d, (lc_logc16_t){ (int16_t)best_rho, th_in });
            if (y.rho != LC_LOG_ZERO) {
                uint16_t ang4 = (uint16_t)(th_in << 2);
                z4_re += (int64_t)lc_trig_cos_q14(ang4);
                z4_im += (int64_t)lc_trig_sin_q14(ang4);
            }
        }
    } else {
        /* 1 LSB で 0->256 へ遷移する場合は最小全生存振幅 hi の出力振幅変調から Z4 を抽出 */
        int32_t vals[256];
        int64_t sum_v = 0;
        for (int k = 0; k < 256; k++) {
            uint16_t th_in = (uint16_t)(k * 256);
            lc_logc16_t y  = dut_roundtrip_eval(d, (lc_logc16_t){ (int16_t)hi, th_in });
            vals[k] = (int32_t)y.rho;
            sum_v  += vals[k];
        }
        int32_t mean_v = (int32_t)(sum_v / 256);
        for (int k = 0; k < 256; k++) {
            uint16_t ang4 = (uint16_t)((uint16_t)(k * 256) << 2);
            int64_t  dv   = (int64_t)(vals[k] - mean_v);
            /* 最小全生存格子では (±1,±1) (norm=√2, 45° mod 90°) が極大となるため符号反転 */
            z4_re -= dv * (int64_t)lc_trig_cos_q14(ang4);
            z4_im -= dv * (int64_t)lc_trig_sin_q14(ang4);
        }
        best_alive = 256;
    }

    if (z4_re == 0 && z4_im == 0) return false;
    uint16_t arg_z4  = moment_to_angle_u16(z4_re, z4_im);
    uint16_t est_th1 = (uint16_t)((uint16_t)(0u - arg_z4) >> 2);
    uint16_t th_tot  = probe_total_phase(d, dut_roundtrip_eval, -256);
    uint16_t est_th2 = (uint16_t)((uint16_t)(th_tot - est_th1) & 0x3FFFu);

    *out_th1_m90   = (uint16_t)(est_th1 & 0x3FFFu);
    *out_th2_m90   = est_th2;
    *out_alive_cnt = best_alive;
    return true;
}

static bool run_test2_roundtrip_50_circuits_mod90(void) {
    uint32_t rng = 0x20261124u;
    int32_t max_e1 = 0, max_e2 = 0;
    int64_t sum_e1 = 0;

    for (int i = 0; i < 50; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng);
        lc_chart_dut_t d;
        d.w1 = (lc_logc16_t){ (int16_t)(-768 + (int32_t)(r0 & 0x3FFu)), (uint16_t)(r0 >> 16) };
        d.w2 = (lc_logc16_t){ (int16_t)(-768 + (int32_t)(r1 & 0x3FFu)), (uint16_t)(r1 >> 16) };
        d.bias_cart = (lc_c16_t){ 0, 0 };

        uint16_t e1 = 0, e2 = 0;
        int alive = 0;
        if (!probe_roundtrip_theta_mod90(&d, &e1, &e2, &alive)) {
            fprintf(stderr, "FAIL: Test 2 回路 %d 臨界格子窓探索失敗\n", i);
            return false;
        }
        int32_t err1 = abs_phase_err_90(e1, d.w1.theta);
        int32_t err2 = abs_phase_err_90(e2, d.w2.theta);
        if (err1 > max_e1) max_e1 = err1;
        if (err2 > max_e2) max_e2 = err2;
        sum_e1 += err1;
    }

    double mean_deg = 360.0 * ((double)sum_e1 / 50.0) / 65536.0;
    double max_deg1 = 360.0 * (double)max_e1 / 65536.0;
    double max_deg2 = 360.0 * (double)max_e2 / 65536.0;

    printf("[Test 2] ランダム 50 回路 (Log↔Cart 往復のみ, 未知 ρ1・ρ2・θ1・θ2) の θ1/θ2 mod 90° 同定:\n");
    printf("  最小非ゼロ格子生存窓 Z4 抽出: 平均誤差 = %.3f deg | 最大誤差 θ1 = %.3f deg, θ2 = %.3f deg OK\n\n",
           mean_deg, max_deg1, max_deg2);
    return (max_deg1 < 1.5 && max_deg2 < 1.5);
}

/* ============================================================================
 * 4. [Test 3 & 4] 残存 C4 対称性の完全破壊 -> 全円 360° での (θ1, θ2) 一意分離
 * ============================================================================ */
static void probe_full360_via_z1(const lc_chart_dut_t *d, dut_fn_t fn,
                                 int16_t rho_z1, int16_t rho_tot,
                                 uint16_t *out_th1, uint16_t *out_th2) {
    int32_t vals[64];
    int64_t sum_v = 0;
    for (int k = 0; k < 64; k++) {
        uint16_t th_in = (uint16_t)(k * 1024);
        lc_logc16_t y  = fn(d, (lc_logc16_t){ rho_z1, th_in });
        vals[k] = (y.rho == LC_LOG_ZERO) ? -4096 : (int32_t)y.rho;
        sum_v += vals[k];
    }
    int32_t mean_v = (int32_t)(sum_v / 64);

    /* 1 倍角モーメント Z1 = Σ (ρ_out(θ_k) - mean) * exp(+i * θ_k) ∝ exp(-i * θ1) */
    int64_t z1_re = 0, z1_im = 0;
    for (int k = 0; k < 64; k++) {
        uint16_t th_in = (uint16_t)(k * 1024);
        int64_t  dv    = (int64_t)(vals[k] - mean_v);
        z1_re += dv * (int64_t)lc_trig_cos_q14(th_in);
        z1_im += dv * (int64_t)lc_trig_sin_q14(th_in);
    }

    uint16_t arg_z1 = moment_to_angle_u16(z1_re, z1_im);
    uint16_t th1    = (uint16_t)(0u - arg_z1);
    uint16_t th_tot = probe_total_phase(d, fn, rho_tot);
    *out_th1 = th1;
    *out_th2 = (uint16_t)(th_tot - th1);
}

static bool run_test3_and_4_full360_separation(void) {
    uint32_t rng = 0x20261125u;

    /* --- Test 3: 中間直交バイアス B = (+1024, 0) による C4 完全破壊 --- */
    int32_t max_b1 = 0, max_b2 = 0;
    int64_t sum_b1 = 0;
    for (int i = 0; i < 50; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng);
        lc_chart_dut_t d;
        d.w1 = (lc_logc16_t){ -384, (uint16_t)(r0 >> 16) };
        d.w2 = (lc_logc16_t){ -384, (uint16_t)(r1 >> 16) };
        d.bias_cart = (lc_c16_t){ 1024, 0 };

        uint16_t e1 = 0, e2 = 0;
        probe_full360_via_z1(&d, dut_bias_eval, -256, -64, &e1, &e2);
        int32_t err1 = abs_phase_err_360(e1, d.w1.theta);
        int32_t err2 = abs_phase_err_360(e2, d.w2.theta);
        if (err1 > max_b1) max_b1 = err1;
        if (err2 > max_b2) max_b2 = err2;
        sum_b1 += err1;
    }

    double b_mean = 360.0 * ((double)sum_b1 / 50.0) / 65536.0;
    double b_max1 = 360.0 * (double)max_b1 / 65536.0;
    double b_max2 = 360.0 * (double)max_b2 / 65536.0;

    printf("[Test 3] 中間直交バイアス (+1024, 0) による C4 完全破壊 (50 ランダム回路, 全円 360° 分離):\n");
    printf("  1 倍角干渉 Z1 抽出: 平均誤差 = %.3f deg | 最大誤差 θ1 = %.3f deg, θ2 = %.3f deg OK\n\n",
           b_mean, b_max1, b_max2);

    /* --- Test 4: 外部バイアス不要・中間 2 次調和干渉 (z ⊕ z²) による C4 完全破壊 --- */
    int32_t max_h1 = 0, max_h2 = 0;
    int64_t sum_h1 = 0;
    for (int i = 0; i < 50; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng);
        lc_chart_dut_t d;
        d.w1 = (lc_logc16_t){ -256, (uint16_t)(r0 >> 16) };
        d.w2 = (lc_logc16_t){ -256, (uint16_t)(r1 >> 16) };
        d.bias_cart = (lc_c16_t){ 0, 0 };

        uint16_t e1 = 0, e2 = 0;
        probe_full360_via_z1(&d, dut_harm2_eval, -256, -256, &e1, &e2);
        int32_t err1 = abs_phase_err_360(e1, d.w1.theta);
        int32_t err2 = abs_phase_err_360(e2, d.w2.theta);
        if (err1 > max_h1) max_h1 = err1;
        if (err2 > max_h2) max_h2 = err2;
        sum_h1 += err1;
    }

    double h_mean = 360.0 * ((double)sum_h1 / 50.0) / 65536.0;
    double h_max1 = 360.0 * (double)max_h1 / 65536.0;
    double h_max2 = 360.0 * (double)max_h2 / 65536.0;

    printf("[Test 4] 中間 2 次調和干渉 (z ⊕ z², 外部バイアスなし) による全円 360° (θ1, θ2) 分離:\n");
    printf("  1 倍角干渉 Z1 抽出: 平均誤差 = %.3f deg | 最大誤差 θ1 = %.3f deg, θ2 = %.3f deg OK\n\n",
           h_mean, h_max1, h_max2);

    return (b_max1 < 1.0 && b_max2 < 1.0 && h_max1 < 1.0 && h_max2 < 1.0);
}

int main(void) {
    if (!lc_v2_init()) { fprintf(stderr, "FATAL: lc_v2_init 失敗\n"); return 1; }

    printf("=== 直交チャート往復 (Log↔Cart)・バイアス・2 次調和干渉による位相配分 (θ1, θ2) 分離実証 ===\n\n");

    if (!run_test1_u1_to_c4_symmetry_breaking())  return 1;
    if (!run_test2_roundtrip_50_circuits_mod90()) return 1;
    if (!run_test3_and_4_full360_separation())    return 1;

    printf(">>> ALL 4 DUAL-CHART PHASE SEPARATION TESTS PASSED (U(1) -> C4 -> {e}) <<<\n");
    return 0;
}