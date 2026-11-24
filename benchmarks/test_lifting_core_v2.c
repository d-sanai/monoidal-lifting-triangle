/* test_lifting_core_v2.c — 8-Module Integrated Verification Suite for lifting_core_v2.h */
#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LIFTING_CORE_V2_IMPLEMENTATION
#include "lifting_core_v2.h"

static bool fn_xor3(bool a, bool b, bool c)  { return (a ^ b ^ c); }
static bool fn_and(bool a, bool b, bool c)   { (void)c; return (a && b); }
static bool fn_carry(bool a, bool b, bool c) { return (a && b) || (b && c) || (c && a); }
static bool fn_or3(bool a, bool b, bool c)   { return (a || b || c); }

static inline uint32_t next_rng(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* 4-bit 可逆アレイ乗算器ビルダー (Test 7 用) */
static void build_mul4(lc_cdcl_graph_t *g, lc_currying_net_t *and_n,
                       lc_currying_net_t *sum_n, lc_currying_net_t *car_n,
                       int16_t obs_p, int vx[4], int vy[4]) {
    lc_cdcl_init(g);
    int v0 = lc_cdcl_add_var(g, 0);
    for (int i = 0; i < 4; i++) {
        vx[i] = lc_cdcl_add_var(g, -1);
        vy[i] = lc_cdcl_add_var(g, -1);
    }
    int8_t bp[8];
    for (int i = 0; i < 8; i++) bp[i] = (int8_t)((obs_p >> i) & 1);

    int m[4][4];
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            m[i][j] = lc_cdcl_add_var(g, (i == 0 && j == 0) ? bp[0] : -1);
            lc_cdcl_add_gate(g, and_n, vx[i], vy[j], v0, m[i][j]);
        }
    }
    int c1[4], s1[4], c2[4], s2[4], c3[3];
    int vp1 = lc_cdcl_add_var(g, bp[1]); c1[0] = lc_cdcl_add_var(g, -1);
    lc_cdcl_add_gate(g, sum_n, m[1][0], m[0][1], v0, vp1);
    lc_cdcl_add_gate(g, car_n, m[1][0], m[0][1], v0, c1[0]);
    for (int i = 1; i <= 3; i++) {
        int a = (i < 3) ? m[i + 1][0] : v0;
        s1[i] = lc_cdcl_add_var(g, -1); c1[i] = lc_cdcl_add_var(g, -1);
        lc_cdcl_add_gate(g, sum_n, a, m[i][1], c1[i - 1], s1[i]);
        lc_cdcl_add_gate(g, car_n, a, m[i][1], c1[i - 1], c1[i]);
    }
    int vp2 = lc_cdcl_add_var(g, bp[2]); c2[0] = lc_cdcl_add_var(g, -1);
    lc_cdcl_add_gate(g, sum_n, s1[1], m[0][2], v0, vp2);
    lc_cdcl_add_gate(g, car_n, s1[1], m[0][2], v0, c2[0]);
    for (int i = 1; i <= 3; i++) {
        int a = (i < 3) ? s1[i + 1] : c1[3];
        s2[i] = lc_cdcl_add_var(g, -1); c2[i] = lc_cdcl_add_var(g, -1);
        lc_cdcl_add_gate(g, sum_n, a, m[i][2], c2[i - 1], s2[i]);
        lc_cdcl_add_gate(g, car_n, a, m[i][2], c2[i - 1], c2[i]);
    }
    int vp3 = lc_cdcl_add_var(g, bp[3]); c3[0] = lc_cdcl_add_var(g, -1);
    lc_cdcl_add_gate(g, sum_n, s2[1], m[0][3], v0, vp3);
    lc_cdcl_add_gate(g, car_n, s2[1], m[0][3], v0, c3[0]);
    for (int i = 1; i <= 3; i++) {
        int a  = (i < 3) ? s2[i + 1] : c2[3];
        int pv = lc_cdcl_add_var(g, bp[i + 3]);
        int co = (i < 3) ? lc_cdcl_add_var(g, -1) : lc_cdcl_add_var(g, bp[7]);
        if (i < 3) c3[i] = co;
        lc_cdcl_add_gate(g, sum_n, a, m[i][3], c3[i - 1], pv);
        lc_cdcl_add_gate(g, car_n, a, m[i][3], c3[i - 1], co);
    }
}

int main(void) {
    if (!lc_v2_init()) {
        fprintf(stderr, "FATAL: lc_v2_init 失敗\n");
        return 1;
    }

    printf("=== lifting_core_v2.h 単一ヘッダ 8 モジュール統合検証スイート ===\n");
    printf("LiftingCell=%zu B | LiftingCellLog=%zu B | AoSoA8=(%zu/%zu/%zu/%zu B) | NEON=%s\n\n",
           sizeof(LiftingCell), sizeof(LiftingCellLog),
           sizeof(lc_block8_t), sizeof(lc_log_block8_t),
           sizeof(lc_chart_block8_t), sizeof(lc_holo_block8_t),
           LC_HAS_ARM_NEON ? "ON (ARM64)" : "OFF (Scalar)");

    uint32_t rng = 0x20261123u;

    /* [Test 1] 純整数双対チャート (Cart <-> Log) 往復精度 & 16,384 セル NEON 0-bit 一致 */
    int max_dr = 0, max_dt = 0;
    for (int r = -1024; r <= 128; r += 64) {
        for (int t = 0; t < 65536; t += 211) {
            lc_c16_t    c  = lc_log_to_cart((int16_t)r, (uint16_t)t);
            lc_logc16_t z2 = lc_cart_to_log(c.re, c.im);
            int dr = lc_iabs32((int32_t)z2.rho - r);
            int dt = lc_2morph_phase_norm(lc_eval_2morph(z2, (lc_logc16_t){(int16_t)r, (uint16_t)t}));
            if (dr > max_dr) max_dr = dr;
            if (dt > max_dt) max_dt = dt;
        }
    }
    static lc_chart_block8_t LC_ALIGN64 ch_sc[2048], ch_nm[2048];
    for (int bi = 0; bi < 2048; bi++) {
        for (int k = 0; k < 8; k++) {
            uint32_t r0 = next_rng(&rng);
            ch_sc[bi].re[k] = (bi == 0 && k == 0) ? 0 : ((bi == 0 && k == 1) ? -32768 : (int16_t)(r0 & 0xFFFFu));
            ch_sc[bi].im[k] = (bi == 0 && k == 0) ? 0 : ((bi == 0 && k == 1) ? -32768 : (int16_t)(r0 >> 16));
        }
    }
    memcpy(ch_nm, ch_sc, sizeof(ch_nm));
    lc_chart_aosoa8_cart_to_log_scalar(ch_sc, 2048);
    lc_chart_aosoa8_cart_to_log(ch_nm, 2048);
    if (memcmp(ch_sc, ch_nm, sizeof(ch_sc)) != 0 || max_dr > 2 || max_dt > 64) {
        fprintf(stderr, "FAIL: Test 1 Dual-Chart Cart<->Log\n");
        return 1;
    }
    double t0 = now_ns();
    for (int it = 0; it < 2000; it++) lc_chart_aosoa8_cart_to_log(ch_nm, 2048);
    double dt_c2l = now_ns() - t0;
    printf("[Test 1] 純整数双対チャート Cart<->Log: |Δrho|<=%d, |Δtheta|<=%d (%.3f deg), NEON %.0f M/s (0-bit OK)\n",
           max_dr, max_dt, 360.0 * (double)max_dt / 65536.0, (16384.0 * 2000.0 * 1e3) / dt_c2l);

    /* [Test 2] 64 層双対チャート閉ループ・リフティング梯子 (μ^64 ∘ Δ^64 = id) */
    lc_logc16_t lad_w[64];
    lc_c16_t    ls0[256], lv0[256], ls[256], lv[256];
    for (int L = 0; L < 64; L++) {
        uint32_t r0 = next_rng(&rng);
        lad_w[L] = (lc_logc16_t){ (int16_t)((int32_t)(r0 & 0x3FFu) - 512), (uint16_t)(r0 >> 16) };
    }
    for (int i = 0; i < 256; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng);
        ls0[i] = (i == 0) ? (lc_c16_t){0, 0} : (lc_c16_t){(int16_t)(r0 & 0xFFFFu), (int16_t)(r0 >> 16)};
        lv0[i] = (i == 1) ? (lc_c16_t){-32768, -32768} : (lc_c16_t){(int16_t)(r1 & 0xFFFFu), (int16_t)(r1 >> 16)};
        ls[i] = ls0[i]; lv[i] = lv0[i];
    }
    lc_dual_ladder_forward(ls, lv, 256, lad_w, 64);
    lc_dual_ladder_inverse(ls, lv, 256, lad_w, 64);
    if (memcmp(ls, ls0, sizeof(ls)) != 0 || memcmp(lv, lv0, sizeof(lv)) != 0) {
        fprintf(stderr, "FAIL: Test 2 64 層閉ループ梯子可逆性違反\n");
        return 1;
    }
    printf("[Test 2] 64 層双対チャート閉ループ梯子 (μ^64 ∘ Δ^64 = id): 256 複素ペア 0-bit 完全復元 (OK)\n");

    /* [Test 3] Log-Polar FHRR Bind & Dual-Chart NEON AoSoA-8 (0-bit 一致 + L1D 速度) */
    static lc_log_block8_t LC_ALIGN64 lb_sc[64], lb_nm[64];
    for (int bi = 0; bi < 64; bi++) {
        for (int k = 0; k < 8; k++) {
            uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng), r2 = next_rng(&rng);
            lb_sc[bi].s_rho[k]   = (k == 0 && bi == 0) ? LC_LOG_ZERO : (int16_t)((int32_t)(r0 & 0x7FFu) - 1024);
            lb_sc[bi].g_rho[k]   = (int16_t)((int32_t)(r1 & 0x3FFu) - 512);
            lb_sc[bi].w_rho[k]   = (int16_t)((int32_t)(r2 & 0x3FFu) - 512);
            lb_sc[bi].s_theta[k] = (uint16_t)(r0 >> 16);
            lb_sc[bi].g_theta[k] = (uint16_t)(r1 >> 16);
            lb_sc[bi].w_theta[k] = (uint16_t)(r2 >> 16);
            lb_sc[bi].v_re[k]    = (int16_t)(bi * 31 + k);
            lb_sc[bi].v_im[k]    = (int16_t)(-(bi * 17 + k));
        }
    }
    memcpy(lb_nm, lb_sc, sizeof(lb_nm));
    lc_log_aosoa8_dual_step_scalar(lb_sc, 64);
    lc_log_aosoa8_dual_step(lb_nm, 64);
    if (memcmp(lb_sc, lb_nm, sizeof(lb_sc)) != 0) {
        fprintf(stderr, "FAIL: Test 3 Log-Polar NEON 0-bit 不一致\n");
        return 1;
    }
    t0 = now_ns();
    for (int it = 0; it < 40000; it++) lc_log_aosoa8_bind_only(lb_nm, 64);
    double dt_bnd = now_ns() - t0;
    printf("[Test 3] Log-Polar FHRR AoSoA-8 (512 cells L1D): Bind %.2f ns (%.0f M/s), Dual-Step 0-bit 一致 (OK)\n",
           dt_bnd / (512.0 * 40000.0), (512.0 * 40000.0 * 1e3) / dt_bnd);

    /* [Test 4] 2-射 H1 ホロノミー障害 -> 3-射自律生成 + Associator 回復 + 512d NEON 検出 */
    lc_diamond_t dm;
    lc_diamond_init(&dm, (lc_logc16_t){256, 0x1000u},
                    (lc_logc16_t){128, 0x1111u}, (lc_logc16_t){64, 0x2222u},
                    (lc_logc16_t){-100, 0x5555u});
    bool spawned_3m = false;
    int  steps_3m   = 0;
    int32_t stat_l1 = 0;
    lc_logc16_t env_tw = { -384, 0x2B73u };
    if (!lc_diamond_solve_h1(&dm, env_tw, &spawned_3m, &steps_3m, &stat_l1) ||
        !spawned_3m || dm.p2_tail->w.rho != env_tw.rho || dm.p2_tail->w.theta != env_tw.theta) {
        fprintf(stderr, "FAIL: Test 4-A 3-射 TwistedSection 生成失敗\n");
        return 1;
    }
    lc_logc16_t assoc_v;
    lc_2morph_t assoc_a;
    bool reassoc = false;
    if (!lc_assoc_resolve((lc_logc16_t){-20000, 100}, (lc_logc16_t){-20000, 200},
                          (lc_logc16_t){20000, 300}, &assoc_v, &assoc_a, &reassoc) ||
        !reassoc || !assoc_a.has_bot || assoc_v.rho != -20000 || assoc_v.theta != 600u) {
        fprintf(stderr, "FAIL: Test 4-B Associator 偽⊥回復失敗\n");
        return 1;
    }
    static lc_holo_block8_t LC_ALIGN64 hb_sc[64], hb_nm[64];
    for (int bi = 0; bi < 64; bi++) {
        for (int k = 0; k < 8; k++) {
            uint32_t r0 = next_rng(&rng);
            hb_sc[bi].z2_rho[k] = 100; hb_sc[bi].z2_theta[k] = (uint16_t)(r0 & 0xFFFFu);
            hb_sc[bi].z1_rho[k] = 228; hb_sc[bi].z1_theta[k] = (uint16_t)(hb_sc[bi].z2_theta[k] + (k == 0 ? 0x8000u : 0x1234u));
            hb_sc[bi].tw_rho[k] = 128; hb_sc[bi].tw_theta[k] = (k == 0) ? 0x8000u : 0x1234u;
        }
    }
    memcpy(hb_nm, hb_sc, sizeof(hb_nm));
    uint64_t h_pre_sc = lc_holo_aosoa8_eval_2morph_scalar(hb_sc, 64, false);
    uint64_t h_pre_nm = lc_holo_aosoa8_eval_2morph(hb_nm, 64, false);
    uint64_t h_aft_nm = lc_holo_aosoa8_eval_2morph(hb_nm, 64, true);
    if (h_pre_sc != h_pre_nm || h_pre_nm == 0 || h_aft_nm != 0) {
        fprintf(stderr, "FAIL: Test 4-C 512d NEON ホロノミー検証失敗\n");
        return 1;
    }
    printf("[Test 4] 高次圏 2-射/3-射 & Associator: H1(L1=%d)->3射 %d steps (0-bit), 偽⊥組換 OK, 512d NEON 0-bit OK\n",
           stat_l1, steps_3m);

    /* [Test 5] Frobenius ZX-Spider 公理 + 256d 閉ループホロノミー自己修復 (int64 集約) */
    const uint16_t z2[2] = { 0x0000u, 0x8000u };
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            lc_logc16_t a0, a1, b0, b1, m0, m1;
            lc_zx_z_copy((lc_logc16_t){0, z2[a]}, (lc_logc16_t){0, 0}, &a0, &a1);
            lc_zx_z_copy((lc_logc16_t){0, z2[b]}, (lc_logc16_t){0, 0}, &b0, &b1);
            lc_zx_z_copy((lc_logc16_t){0, lc_zx_x_parity(z2[a], z2[b], 0)}, (lc_logc16_t){0, 0}, &m0, &m1);
            if (lc_zx_x_parity(a0.theta, b0.theta, 0) != m0.theta ||
                lc_zx_x_parity(a0.theta, a1.theta, 0) != 0u) {
                fprintf(stderr, "FAIL: Test 5-A ZX 公理違反\n");
                return 1;
            }
        }
    }
    enum { D = 256, NB = 4, BW = 64, NI = 8 };
    static uint16_t fk[NI][D], fv[NI][D];
    static lc_c16_t fb[D];
    const uint16_t tw_b[NB] = { 0x8000u, 0x4000u, 0x2B73u, 0xC915u };
    for (int m = 0; m < NI; m++)
        for (int d = 0; d < D; d++) { fk[m][d] = (uint16_t)next_rng(&rng); fv[m][d] = (uint16_t)next_rng(&rng); }
    for (int d = 0; d < D; d++) {
        int32_t ar = 0, ai = 0;
        for (int m = 0; m < NI; m++) {
            lc_logc16_t kv = lc_fhrr_bind((lc_logc16_t){0, fk[m][d]}, (lc_logc16_t){0, fv[m][d]});
            lc_c16_t    c  = lc_log_to_cart(kv.rho, kv.theta);
            ar += c.re; ai += c.im;
        }
        fb[d] = (lc_c16_t){ (int16_t)(ar >> 3), (int16_t)(ai >> 3) };
    }
    uint16_t zcomp[NB] = {0};
    for (int b = 0; b < NB; b++) {
        int64_t sr = 0, si = 0;
        for (int i = 0; i < BW; i++) {
            int d = b * BW + i;
            uint16_t rot = (uint16_t)(fk[0][d] + tw_b[b] + fv[0][d]);
            int32_t cr = lc_trig_cos_q14(rot), cs = lc_trig_sin_q14(rot);
            sr += (int64_t)fb[d].re * cr + (int64_t)fb[d].im * cs;
            si += (int64_t)fb[d].im * cr - (int64_t)fb[d].re * cs;
        }
        zcomp[b] = lc_cart_to_log((int16_t)(sr >> 18), (int16_t)(si >> 18)).theta;
    }
    int zx_hits = 0;
    for (int q = 0; q < NI; q++) {
        int best = -1; int64_t bdot = INT64_MIN;
        for (int c = 0; c < NI; c++) {
            int64_t dot = 0;
            for (int d = 0; d < D; d++) {
                uint16_t rot = (uint16_t)(fk[q][d] + tw_b[d / BW] + zcomp[d / BW] + fv[c][d]);
                dot += (int64_t)fb[d].re * lc_trig_cos_q14(rot) + (int64_t)fb[d].im * lc_trig_sin_q14(rot);
            }
            if (dot > bdot) { bdot = dot; best = c; }
        }
        if (best == q) zx_hits++;
    }
    if (zx_hits != NI) { fprintf(stderr, "FAIL: Test 5-B ZX 修復 %d/8\n", zx_hits); return 1; }
    printf("[Test 5] Frobenius ZX-Spider 公理 & 256d 閉ループホロノミー修復: 4公理 0-bit, 想起 %d/8 回復 (OK)\n", zx_hits);

    /* [Test 6] 動的カリー化自己合成 & 共有 DAG 参照カウント代謝 */
    lc_currying_net_t xor3_n, and_n, car_n, or3_n;
    lc_net_init(&xor3_n, 3); lc_net_init(&and_n, 3);
    lc_net_init(&car_n,  3); lc_net_init(&or3_n, 3);
    if (!lc_net_synthesize(&xor3_n, fn_xor3, 300) || !lc_net_synthesize(&and_n, fn_and, 300) ||
        !lc_net_synthesize(&car_n,  fn_carry, 400) || !lc_net_synthesize(&or3_n, fn_or3, 500)) {
        fprintf(stderr, "FAIL: Test 6 動的カリー化合成\n");
        return 1;
    }
    static lc_dag_pool_t dp;
    lc_dag_init(&dp);
    const lc_lit_t t1[3] = {{1, 1}, {2, 1}, {3, 1}}, t2[3] = {{1, 1}, {2, 1}, {4, 0}};
    LiftingCell *dc1 = lc_dag_add_term(&dp, t1, 3);
    (void)lc_dag_add_term(&dp, t2, 3);
    dc1->idle_counter = 200;
    (void)lc_dag_prune_leaf(&dp, true);
    (void)lc_dag_add_term(&dp, t1, 3);
    if (dp.shared_hits < 2 || dp.pruned_cells < 1 || dp.reused_slots < 1) {
        fprintf(stderr, "FAIL: Test 6 共有 DAG 代謝\n");
        return 1;
    }
    printf("[Test 6] 動的カリー化合成 (XOR3=%d,AND=%d,Carry=%d,OR3=%d) & 共有DAG代謝 (shared=%d,reused=%d): OK\n",
           xor3_n.num_cells, and_n.num_cells, car_n.num_cells, or3_n.num_cells, dp.shared_hits, dp.reused_slots);

    /* [Test 7] 最小十分原因 + 1-UIP CDCL アブダクション (143 = 11*13 & 199 = UNSAT) */
    static lc_cdcl_graph_t cg;
    bool sol[LC_CDCL_MAX_VARS];
    lc_cdcl_stats_t cst;
    int vx[4], vy[4];
    build_mul4(&cg, &and_n, &xor3_n, &car_n, 143, vx, vy);
    if (!lc_cdcl_solve(&cg, 1000, sol, &cst)) { fprintf(stderr, "FAIL: Test 7-A 143\n"); return 1; }
    uint8_t x = 0, y = 0;
    for (int i = 0; i < 4; i++) {
        if (sol[vx[i]]) x |= (uint8_t)(1u << i);
        if (sol[vy[i]]) y |= (uint8_t)(1u << i);
    }
    build_mul4(&cg, &and_n, &xor3_n, &car_n, 199, vx, vy);
    if ((uint16_t)x * (uint16_t)y != 143u || lc_cdcl_solve(&cg, 1000, sol, &cst)) {
        fprintf(stderr, "FAIL: Test 7-B CDCL\n");
        return 1;
    }
    printf("[Test 7] 1-UIP CDCL アブダクション: 143 = %u * %u, 素数 199 = UNSAT 証明 (OK)\n", x, y);

    /* [Test 8] 連続位相 ECC (ハミング 7,4) & L1 ブロック分割 Q1.14 3-pass NEON AoSoA-8 */
    static lc_ecc_graph_t ham;
    lc_ecc_init(&ham, &xor3_n, 7);
    const int r0[4] = {0, 2, 4, 6}, r1[4] = {1, 2, 5, 6}, r2[4] = {3, 4, 5, 6};
    lc_ecc_add_check(&ham, r0, 4); lc_ecc_add_check(&ham, r1, 4); lc_ecc_add_check(&ham, r2, 4);
    int ok_1b = 0, ok_2s = 0;
    const int dpos[4] = {2, 4, 5, 6};
    for (int msg = 0; msg < 16; msg++) {
        int8_t obs[7] = {-1,-1,-1,-1,-1,-1,-1};
        for (int k = 0; k < 4; k++) obs[dpos[k]] = (int8_t)((msg >> k) & 1);
        uint8_t cw[7], dec[7];
        if (lc_ecc_decode_erasure(&ham, obs, cw) != 0) return 1;
        for (int fl = 0; fl < 7; fl++) {
            int32_t llr[7];
            for (int i = 0; i < 7; i++) llr[i] = ((cw[i] ^ (i == fl ? 1u : 0u)) == 0u) ? 512 : -512;
            if (lc_ecc_decode_soft(&ham, llr, 20, false, dec) >= 0 && memcmp(cw, dec, 7) == 0) ok_1b++;
        }
        int32_t sllr[7];
        for (int i = 0; i < 7; i++) {
            int32_t s = (cw[i] == 0u) ? 1 : -1;
            sllr[i] = (i == 0 || i == 3) ? (-s * 64) : (s * 768);
        }
        if (lc_ecc_decode_soft(&ham, sllr, 20, true, dec) >= 0 && memcmp(cw, dec, 7) == 0) ok_2s++;
    }
    static lc_block8_t LC_ALIGN64 qblk[2048];
    for (int bi = 0; bi < 2048; bi++) {
        for (int k = 0; k < 8; k++) {
            uint32_t rv = next_rng(&rng);
            qblk[bi].s_re[k] = (int16_t)(rv & 0xFFFFu); qblk[bi].s_im[k] = (int16_t)(rv >> 16);
            qblk[bi].g_re[k] = (k & 1) ? 8192 : LC_Q14_ONE; qblk[bi].g_im[k] = 0;
            qblk[bi].v_re[k] = (int16_t)(bi * 17 + k); qblk[bi].v_im[k] = (int16_t)(-(bi * 13 + k));
            qblk[bi].w_re[k] = (int16_t)(k * 1024 - 4096); qblk[bi].w_im[k] = LC_Q14_NOT;
            qblk[bi].adj_in[k] = (bi & 1) ? 4096 : -4096; qblk[bi].gate_mask[k] = (k & 1) ? -1 : 0;
        }
    }
    t0 = now_ns();
    for (int it = 0; it < 500; it++) lc_aosoa8_step_l1_blocked(qblk, 2048, 2);
    double dt_q14 = now_ns() - t0;
    for (int bi = 0; bi < 2048; bi++)
        for (int k = 0; k < 8; k++)
            if (qblk[bi].restored_re[k] != qblk[bi].v_re[k] || qblk[bi].restored_im[k] != qblk[bi].v_im[k]) return 1;
    if (ok_1b != 112 || ok_2s != 16) return 1;
    printf("[Test 8] 連続位相 ECC (%d/112, 軟2b=%d/16) & Q1.14 3-pass NEON (%.0f M/s, 0-bit 可逆): OK\n\n",
           ok_1b, ok_2s, (16384.0 * 500.0 * 1e3) / dt_q14);

    printf(">>> ALL 8 INTEGRATED MODULES PASSED IN lifting_core_v2.h <<<\n");
    return 0;
}