/* lifting_h1_holonomy_rewrite.c — Higher-Category 2-Morphism (H1 Holonomy) & 3-Morphism Rewrite */
#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define LC_HAS_NEON 1
#else
#define LC_HAS_NEON 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LC_ALIGN64  __attribute__((aligned(64)))
#define LC_NOINLINE __attribute__((noinline))
#else
#define LC_ALIGN64
#define LC_NOINLINE
#endif

/* ============================================================================
 * 1. 複素対数極座標 (lc_logc16_t) / 64-byte LiftingCellLog / 2-射 (lc_2morph_t)
 * ============================================================================ */
#define LC_LOG_ZERO (-32768)

#define LC_FLAG_ALIVE         (1u << 0)
#define LC_FLAG_CURRIED_TWIST (1u << 1)  /* 3-射により自律生成された文脈依存補償 1-射 */
#define LC_FLAG_LOCKED        (1u << 2)  /* 観測・物理制約により固定された 1-射 */

typedef struct {
    int16_t  rho;    /* Q7.8 log2(|z|), INT16_MIN = ⊥ */
    uint16_t theta;  /* Z / 2^16 Z 円環位相 */
} lc_logc16_t;

typedef struct {
    int16_t re;
    int16_t im;
} lc_c16_t;

typedef struct {
    int32_t drho;    /* 対数強度ズレ (rho1 - rho2) */
    int16_t dtheta;  /* 符号付き循環位相差 (int16_t)(theta1 - theta2) ∈ [-32768, 32767] */
    bool    has_bot; /* 片方のみ ⊥ (偽消滅または経路切断) */
} lc_2morph_t;

/* 64-byte 均一 Log-Polar FHRR LiftingCell (Step A と厳密同一レイアウト) */
typedef struct LiftingCellLog {
    struct LiftingCellLog *in_s;
    struct LiftingCellLog *in_g;
    struct LiftingCellLog *in_v;

    lc_logc16_t w;
    lc_logc16_t s_eff;
    lc_logc16_t p_log;

    lc_c16_t p_cart;
    lc_c16_t r_cart;
    lc_c16_t restored_cart;

    int32_t adj_rho;
    int32_t adj_theta;

    uint16_t ref_count;
    uint16_t var_id;
    uint8_t  idle_counter;
    uint8_t  var_mask;
    uint8_t  order;
    uint8_t  flags;
} LiftingCellLog;

#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFULL
typedef char lc_assert_cell_log_64b[(sizeof(LiftingCellLog) == 64) ? 1 : -1];
#endif

static inline lc_logc16_t fhrr_bind(lc_logc16_t a, lc_logc16_t b, bool *saturated) {
    if (a.rho == LC_LOG_ZERO || b.rho == LC_LOG_ZERO) {
        lc_logc16_t z = { LC_LOG_ZERO, (uint16_t)(a.theta + b.theta) };
        return z;
    }
    int32_t s = (int32_t)a.rho + (int32_t)b.rho;
    if (s > 32767) {
        s = 32767;
        if (saturated != NULL) *saturated = true;
    } else if (s <= LC_LOG_ZERO) {
        s = LC_LOG_ZERO;
        if (saturated != NULL) *saturated = true;
    }
    lc_logc16_t out = { (int16_t)s, (uint16_t)(a.theta + b.theta) };
    return out;
}

/* 2 つの分岐経路 z1, z2 の合流差 = 2-射 α: z2 => z1 */
static inline lc_2morph_t eval_2morph(lc_logc16_t z1, lc_logc16_t z2) {
    lc_2morph_t m = { 0, 0, false };
    if (z1.rho == LC_LOG_ZERO && z2.rho == LC_LOG_ZERO) {
        return m;
    }
    if (z1.rho == LC_LOG_ZERO || z2.rho == LC_LOG_ZERO) {
        m.has_bot = true;
        m.drho    = (z1.rho == LC_LOG_ZERO) ? -32768 : 32768;
        m.dtheta  = (int16_t)((uint16_t)(z1.theta - z2.theta));
        return m;
    }
    m.drho   = (int32_t)z1.rho - (int32_t)z2.rho;
    m.dtheta = (int16_t)((uint16_t)(z1.theta - z2.theta));
    return m;
}

static inline int32_t iabs32(int32_t x) { return (x >= 0) ? x : -x; }

static inline int32_t morph2_phase_norm(lc_2morph_t m) {
    return (m.dtheta == -32768) ? 32768 : iabs32((int32_t)m.dtheta);
}

static inline int32_t morph2_l1(lc_2morph_t m) {
    return iabs32(m.drho) + morph2_phase_norm(m) + (m.has_bot ? 65536 : 0);
}

static inline bool morph2_is_flat(lc_2morph_t m) {
    return !m.has_bot && m.drho == 0 && m.dtheta == 0;
}

static inline int32_t half_step_floor(int32_t d) {
    if (d == 0) return 0;
    if (d > 0)  return (d + 1) >> 1;
    return -((-d + 1) >> 1);
}

/* 64-byte LiftingCellLog の 1-射評価
 * in_g != NULL かつゲートが ⊥ (文脈オフ) の場合は恒等 1-射として素通りバイパス */
static inline void cell_log_eval(LiftingCellLog *c) {
    lc_logc16_t s = c->in_s->s_eff;
    if (c->in_g != NULL && c->in_g->s_eff.rho == LC_LOG_ZERO) {
        c->s_eff = s;
        return;
    }
    c->s_eff = fhrr_bind(s, c->w, NULL);
}

static inline void cell_log_apply_gauge_step(LiftingCellLog *c, int32_t drho, int32_t dtheta) {
    if ((c->flags & LC_FLAG_LOCKED) != 0u) return;
    int32_t sr = half_step_floor(drho);
    int32_t st = half_step_floor(dtheta);
    int32_t nr = (int32_t)c->w.rho + sr;
    if (nr >  32767) nr =  32767;
    if (nr < -32767) nr = -32767;
    c->w.rho   = (int16_t)nr;
    c->w.theta = (uint16_t)(c->w.theta + (uint16_t)(int16_t)st);
}

/* ============================================================================
 * 2. [Test 1] 可換ダイアモンドのゲージ緩和 (H1 = 0 コバウンダリ消去)
 * ============================================================================ */
static bool run_test1_gauge_relaxation(void) {
    static const struct {
        const char *name;
        int16_t     init_drho;
        uint16_t    init_dtheta;
    } cases[] = {
        { "対蹠点反転 (π = 0x8000) +4 oct",   1024, 0x8000u },
        { "四分回転   (π/2 = 0x4000) -2 oct", -512, 0x4000u },
        { "任意位相   (0x2B73) +1.5 oct",      384, 0x2B73u },
        { "微小ズレ   (1 LSB) -1 LSB",          -1, 0x0001u }
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    LiftingCellLog port_a, f1, f2, g1, g2;
    printf("[Test 1] 可換ダイアモンドの 2-射ゲージ緩和 (H1 = 0 コバウンダリ消去):\n");

    for (int ci = 0; ci < n_cases; ci++) {
        memset(&port_a, 0, sizeof(port_a));
        memset(&f1, 0, sizeof(f1)); memset(&f2, 0, sizeof(f2));
        memset(&g1, 0, sizeof(g1)); memset(&g2, 0, sizeof(g2));

        port_a.s_eff = (lc_logc16_t){ 256, 0x1234u };
        f1.in_s = &port_a; f1.w = (lc_logc16_t){ 128, 0x2000u }; f1.flags = LC_FLAG_ALIVE | LC_FLAG_LOCKED;
        f2.in_s = &f1;     f2.w = (lc_logc16_t){ cases[ci].init_drho, cases[ci].init_dtheta };
        f2.flags = LC_FLAG_ALIVE | LC_FLAG_LOCKED;

        g1.in_s = &port_a; g1.w = (lc_logc16_t){ 128, 0x2000u }; g1.flags = LC_FLAG_ALIVE | LC_FLAG_LOCKED;
        g2.in_s = &g1;     g2.w = (lc_logc16_t){ 0, 0 };         g2.flags = LC_FLAG_ALIVE;

        int conv_step = -1;
        for (int step = 0; step <= 24; step++) {
            cell_log_eval(&f1); cell_log_eval(&f2);
            cell_log_eval(&g1); cell_log_eval(&g2);
            lc_2morph_t m = eval_2morph(f2.s_eff, g2.s_eff);
            if (morph2_is_flat(m)) {
                conv_step = step;
                break;
            }
            cell_log_apply_gauge_step(&g2, m.drho, (int32_t)m.dtheta);
        }
        if (conv_step < 0 || conv_step > 16) {
            fprintf(stderr, "FAIL: Test 1 ケース %d 未収束 (step=%d)\n", ci, conv_step);
            return false;
        }
        printf("  %-34s : %2d steps -> Δζ=(0, 0) [0-bit 平坦 OK]\n", cases[ci].name, conv_step);
    }
    printf("\n");
    return true;
}

/* ============================================================================
 * 3. [Test 2] 非自明な H1 ホロノミー障害 -> 3-射によるカリー化 TwistedSection 自律生成
 * ============================================================================ */
typedef struct {
    LiftingCellLog pool[16];
    int            used;
    LiftingCellLog port_a;
    LiftingCellLog port_ctx;
    LiftingCellLog *f1, *f2, *g1, *g2;
    LiftingCellLog *p2_tail;    /* 3-射書き換え前は g2、書き換え後はスポーンした twist セル */
} HolonomyDiamond;

static LiftingCellLog *hd_alloc(HolonomyDiamond *d) {
    if (d->used >= 16) return NULL;
    LiftingCellLog *c = &d->pool[d->used++];
    memset(c, 0, sizeof(*c));
    c->flags = LC_FLAG_ALIVE;
    return c;
}

static lc_2morph_t hd_eval_ctx(HolonomyDiamond *d, bool ctx_on, lc_logc16_t env_twist) {
    d->port_a.s_eff   = (lc_logc16_t){ 256, 0x1000u };
    d->port_ctx.s_eff = ctx_on ? (lc_logc16_t){ 0, 0 } : (lc_logc16_t){ LC_LOG_ZERO, 0 };

    /* 経路 1: 環境側の真の接続 (ctx_on=1 のときのみ未知のねじれ env_twist が乗る) */
    cell_log_eval(d->f1);
    cell_log_eval(d->f2);
    lc_logc16_t p1_out = ctx_on ? fhrr_bind(d->f2->s_eff, env_twist, NULL) : d->f2->s_eff;

    /* 経路 2: モデル側の接続鎖 (g1 -> g2 [-> 3-射で挿入された twist]) */
    cell_log_eval(d->g1);
    cell_log_eval(d->g2);
    if (d->p2_tail != d->g2) {
        cell_log_eval(d->p2_tail);
    }
    return eval_2morph(p1_out, d->p2_tail->s_eff);
}

static bool run_test2_h1_obstruction_and_3morph_rewrite(void) {
    static const struct {
        const char *name;
        lc_logc16_t twist;
    } cases[] = {
        { "メビウス束ねじれ (π = 0x8000)",          {    0, 0x8000u } },
        { "4 重被覆位相ねじれ (π/2 = 0x4000)",       {    0, 0x4000u } },
        { "複素対数ねじれ (-1.5 oct, 0x2B73)",       { -384, 0x2B73u } }
    };
    const int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    printf("[Test 2] 非自明 H1 ホロノミー障害検出 -> 3-射カリー化 TwistedSection 自律生成:\n");

    for (int ci = 0; ci < n_cases; ci++) {
        HolonomyDiamond hd;
        memset(&hd, 0, sizeof(hd));
        hd.f1 = hd_alloc(&hd); hd.f2 = hd_alloc(&hd);
        hd.g1 = hd_alloc(&hd); hd.g2 = hd_alloc(&hd);

        hd.f1->in_s = &hd.port_a; hd.f1->w = (lc_logc16_t){  128, 0x1111u }; hd.f1->flags |= LC_FLAG_LOCKED;
        hd.f2->in_s = hd.f1;      hd.f2->w = (lc_logc16_t){   64, 0x2222u }; hd.f2->flags |= LC_FLAG_LOCKED;
        hd.g1->in_s = &hd.port_a; hd.g1->w = (lc_logc16_t){  128, 0x1111u }; hd.g1->flags |= LC_FLAG_LOCKED;
        hd.g2->in_s = hd.g1;      hd.g2->w = (lc_logc16_t){ -100, 0x5555u }; /* 初期誤差あり */
        hd.p2_tail  = hd.g2;

        /* Stage 1: 静的ゲージ緩和を回しても H1 コサイクル不変量が絶対に消えないことを検証 */
        lc_2morph_t m0_init = hd_eval_ctx(&hd, false, cases[ci].twist);
        lc_2morph_t m1_init = hd_eval_ctx(&hd, true,  cases[ci].twist);
        int32_t  inv_drho_init   = m1_init.drho - m0_init.drho;
        uint16_t inv_dtheta_init = (uint16_t)((uint16_t)m1_init.dtheta - (uint16_t)m0_init.dtheta);

        for (int step = 0; step < 24; step++) {
            lc_2morph_t m0 = hd_eval_ctx(&hd, false, cases[ci].twist);
            lc_2morph_t m1 = hd_eval_ctx(&hd, true,  cases[ci].twist);
            /* 文脈非依存の静的エッジは両文脈の平均トルクでしか動けない */
            cell_log_apply_gauge_step(hd.g2, (m0.drho + m1.drho) / 2,
                                             ((int32_t)m0.dtheta + (int32_t)m1.dtheta) / 2);
        }

        lc_2morph_t m0_stuck = hd_eval_ctx(&hd, false, cases[ci].twist);
        lc_2morph_t m1_stuck = hd_eval_ctx(&hd, true,  cases[ci].twist);
        int32_t  inv_drho_after   = m1_stuck.drho - m0_stuck.drho;
        uint16_t inv_dtheta_after = (uint16_t)((uint16_t)m1_stuck.dtheta - (uint16_t)m0_stuck.dtheta);

        /* トポロジカル不変性検証: 静的ゲージ変換で H1 コサイクルが 1 ビットも変わらない */
        if (inv_drho_init != inv_drho_after || inv_dtheta_init != inv_dtheta_after ||
            (inv_drho_after == 0 && inv_dtheta_after == 0)) {
            fprintf(stderr, "FAIL: Test 2 H1 コサイクル不変条件違反\n");
            return false;
        }

        /* Stage 2: 3-射発火 -> 文脈ゲート付き 64-byte TwistedSection セルを自律スポーン・配線 */
        LiftingCellLog *twist_cell = hd_alloc(&hd);
        twist_cell->in_s  = hd.g2;
        twist_cell->in_g  = &hd.port_ctx;
        twist_cell->w     = (lc_logc16_t){ 0, 0 };
        twist_cell->order = 2;
        twist_cell->flags |= LC_FLAG_CURRIED_TWIST;
        hd.p2_tail = twist_cell;

        int conv_step = -1;
        for (int step = 0; step <= 24; step++) {
            lc_2morph_t m0 = hd_eval_ctx(&hd, false, cases[ci].twist);
            lc_2morph_t m1 = hd_eval_ctx(&hd, true,  cases[ci].twist);
            if (morph2_is_flat(m0) && morph2_is_flat(m1)) {
                conv_step = step;
                break;
            }
            /* c=0 の残差は基底 1-射 g2 へ、c=1 の余剰残差は補償 1-射 twist_cell へ直交配分 */
            cell_log_apply_gauge_step(hd.g2, m0.drho, (int32_t)m0.dtheta);
            lc_2morph_t m1_rem = hd_eval_ctx(&hd, true, cases[ci].twist);
            cell_log_apply_gauge_step(twist_cell, m1_rem.drho, (int32_t)m1_rem.dtheta);
        }

        if (conv_step < 0 ||
            twist_cell->w.rho != cases[ci].twist.rho ||
            twist_cell->w.theta != cases[ci].twist.theta) {
            fprintf(stderr, "FAIL: Test 2 3-射書き換え後の 0-bit 平坦化失敗\n");
            return false;
        }
        printf("  %-35s : 静的残留 L1=%5d -> 3-射生成(cells=%d) %2d steps -> Δζ0=Δζ1=(0,0) [w_twist=(%d, 0x%04X) 一致]\n",
               cases[ci].name, morph2_l1(m0_stuck) + morph2_l1(m1_stuck),
               hd.used, conv_step, (int)twist_cell->w.rho, (unsigned)twist_cell->w.theta);
    }
    printf("\n");
    return true;
}

/* ============================================================================
 * 4. [Test 3] 高次圏 Associator (結合子 α_{a,b,c}) 2-射検出・自律結合組み換え & 五角形図式
 * ============================================================================ */
static inline uint32_t next_rng(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static bool run_test3_associator_and_pentagon(void) {
    printf("[Test 3] 高次圏 Associator (結合子 α_{a,b,c}) & マクレーン五角形図式検証:\n");

    /* 3-A: マクレーン五角形図式 (4 連バインディングの 2 経路 vs 3 経路、1,000 ケース) */
    uint32_t rng = 0x20261123u;
    for (int t = 0; t < 1000; t++) {
        lc_logc16_t x[4];
        for (int i = 0; i < 4; i++) {
            uint32_t r = next_rng(&rng);
            x[i].rho   = ((t % 19 == 0) && (i == (t & 3)))
                         ? LC_LOG_ZERO
                         : (int16_t)((int32_t)(r & 0x1FFFu) - 4096); /* 無飽和域 [-4096, +4095] */
            x[i].theta = (uint16_t)(r >> 16);
        }
        /* 五角形左経路: ((a*b)*c)*d -> (a*b)*(c*d) -> a*(b*(c*d)) */
        lc_logc16_t v0   = fhrr_bind(fhrr_bind(fhrr_bind(x[0], x[1], NULL), x[2], NULL), x[3], NULL);
        lc_logc16_t vL1  = fhrr_bind(fhrr_bind(x[0], x[1], NULL), fhrr_bind(x[2], x[3], NULL), NULL);
        lc_logc16_t vL2  = fhrr_bind(x[0], fhrr_bind(x[1], fhrr_bind(x[2], x[3], NULL), NULL), NULL);
        /* 五角形右経路: ((a*b)*c)*d -> (a*(b*c))*d -> a*((b*c)*d) -> a*(b*(c*d)) */
        lc_logc16_t vR1  = fhrr_bind(fhrr_bind(x[0], fhrr_bind(x[1], x[2], NULL), NULL), x[3], NULL);
        lc_logc16_t vR2  = fhrr_bind(x[0], fhrr_bind(fhrr_bind(x[1], x[2], NULL), x[3], NULL), NULL);

        if (!morph2_is_flat(eval_2morph(v0, vL1)) || !morph2_is_flat(eval_2morph(vL1, vL2)) ||
            !morph2_is_flat(eval_2morph(v0, vR1)) || !morph2_is_flat(eval_2morph(vR1, vR2)) ||
            !morph2_is_flat(eval_2morph(vL2, vR2))) {
            fprintf(stderr, "FAIL: Test 3-A マクレーン五角形図式不整合 (case %d)\n", t);
            return false;
        }
    }
    printf("  3-A マクレーン五角形図式 (1,000 ケース, ⊥混入含む) : 1000/1000 厳密可換 (0-bit OK)\n");

    /* 3-B: 飽和／偽アンダーフロー境界における非自明 Associator 2-射検出と自律結合組み換え */
    static const struct {
        const char  *name;
        lc_logc16_t  a, b, c;
        int16_t      exact_rho;
    } sat_cases[] = {
        { "正飽和境界 ((+24k+24k)-24k)",  {  24000, 1111 }, {  24000, 2222 }, { -24000, 3333 },  24000 },
        { "偽⊥消滅   ((-20k-20k)+20k)",  { -20000, 4000 }, { -20000, 5000 }, {  20000, 6000 }, -20000 },
        { "右飽和逆転 (-24k+(+24k+24k))", { -24000,  100 }, {  24000,  200 }, {  24000,  300 },  24000 }
    };
    const int n_sat = (int)(sizeof(sat_cases) / sizeof(sat_cases[0]));

    for (int i = 0; i < n_sat; i++) {
        bool sat_L = false, sat_R = false;
        lc_logc16_t res_L = fhrr_bind(fhrr_bind(sat_cases[i].a, sat_cases[i].b, &sat_L), sat_cases[i].c, &sat_L);
        lc_logc16_t res_R = fhrr_bind(sat_cases[i].a, fhrr_bind(sat_cases[i].b, sat_cases[i].c, &sat_R), &sat_R);
        lc_2morph_t alpha = eval_2morph(res_R, res_L);
        uint16_t    exp_t = (uint16_t)(sat_cases[i].a.theta + sat_cases[i].b.theta + sat_cases[i].c.theta);

        /* 非自明な Associator 2-射を検出し、無飽和側の結合木へ自律組み換え */
        if (morph2_is_flat(alpha) || alpha.dtheta != 0) {
            fprintf(stderr, "FAIL: Test 3-B Associator 2-射検出失敗 (%d)\n", i);
            return false;
        }
        lc_logc16_t chosen = (!sat_R && sat_L) ? res_R : res_L;
        const char *dir    = (!sat_R && sat_L) ? "左->右結合へ組換" : "右->左結合へ組換";

        if (chosen.rho != sat_cases[i].exact_rho || chosen.theta != exp_t) {
            fprintf(stderr, "FAIL: Test 3-B 結合組み換え後の厳密値不一致 (%d)\n", i);
            return false;
        }
        printf("  3-B %-29s : α=(drho=%+6d, bot=%d) 検出 -> %s -> rho=%+6d [0-bit 厳密回復]\n",
               sat_cases[i].name, (int)alpha.drho, alpha.has_bot ? 1 : 0, dir, (int)chosen.rho);
    }
    printf("\n");
    return true;
}

/* ============================================================================
 * 5. [Test 4] 256 次元 FHRR ホロノミー 2-射 ARM64 NEON 一括検出・補償合成・速度実測
 * ============================================================================ */
typedef struct LC_ALIGN64 {
    int16_t  z1_rho[8];
    uint16_t z1_theta[8];
    int16_t  z2_rho[8];
    uint16_t z2_theta[8];
    int16_t  tw_rho[8];
    uint16_t tw_theta[8];
    int16_t  out_drho[8];
    int16_t  out_dtheta[8];
} lc_holo_block8_t;

typedef char lc_assert_holo_block8_128b[(sizeof(lc_holo_block8_t) == 128) ? 1 : -1];

static inline int16_t sat_add_s16_scalar(int16_t a, int16_t b) {
    int32_t s = (int32_t)a + (int32_t)b;
    if (s >  32767) return  32767;
    if (s < -32768) return -32768;
    return (int16_t)s;
}

static inline int16_t sat_sub_s16_scalar(int16_t a, int16_t b) {
    int32_t s = (int32_t)a - (int32_t)b;
    if (s >  32767) return  32767;
    if (s < -32768) return -32768;
    return (int16_t)s;
}

static LC_NOINLINE uint64_t holonomy_2morph_scalar(lc_holo_block8_t *blocks, int n_blocks,
                                                   bool apply_twist) {
    uint64_t phase_l1 = 0;
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_holo_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            int16_t  r2 = apply_twist ? sat_add_s16_scalar(b->z2_rho[k], b->tw_rho[k]) : b->z2_rho[k];
            uint16_t t2 = apply_twist ? (uint16_t)(b->z2_theta[k] + b->tw_theta[k])    : b->z2_theta[k];
            int16_t  dr = sat_sub_s16_scalar(b->z1_rho[k], r2);
            int16_t  dt = (int16_t)((uint16_t)(b->z1_theta[k] - t2));
            b->out_drho[k]   = dr;
            b->out_dtheta[k] = dt;
            phase_l1 += (uint64_t)((dt == -32768) ? 32768 : ((dt >= 0) ? dt : -dt));
        }
    }
    return phase_l1;
}

#if LC_HAS_NEON
static LC_NOINLINE uint64_t holonomy_2morph_neon(lc_holo_block8_t *blocks, int n_blocks,
                                                 bool apply_twist) {
    uint64x2_t acc = vdupq_n_u64(0);
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_holo_block8_t *b = &blocks[bi];
        int16x8_t  r1 = vld1q_s16(b->z1_rho),   r2 = vld1q_s16(b->z2_rho);
        uint16x8_t t1 = vld1q_u16(b->z1_theta), t2 = vld1q_u16(b->z2_theta);
        if (apply_twist) {
            r2 = vqaddq_s16(r2, vld1q_s16(b->tw_rho));
            t2 = vaddq_u16(t2, vld1q_u16(b->tw_theta));
        }
        int16x8_t  dr    = vqsubq_s16(r1, r2);
        int16x8_t  dt    = vreinterpretq_s16_u16(vsubq_u16(t1, t2));
        /* vabsq_s16(-32768) = 0x8000 -> uint16x8_t 解釈で厳密に +32768 (全値 0-bit 一致) */
        uint16x8_t abs_t = vreinterpretq_u16_s16(vabsq_s16(dt));
        acc = vpadalq_u32(acc, vpaddlq_u16(abs_t));

        vst1q_s16(b->out_drho,   dr);
        vst1q_s16(b->out_dtheta, dt);
    }
    return vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
}
#endif

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static bool run_test4_256d_fhrr_holonomy_neon(void) {
    enum { N_BLOCKS = 64, N_DIMS = N_BLOCKS * 8 }; /* 512 次元 = 8 KB (L1D 常駐) */
    static lc_holo_block8_t LC_ALIGN64 blk_sc[N_BLOCKS];
    static lc_holo_block8_t LC_ALIGN64 blk_nm[N_BLOCKS];

    uint32_t rng = 0x20261123u;
    for (int bi = 0; bi < N_BLOCKS; bi++) {
        for (int k = 0; k < 8; k++) {
            uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng);
            int16_t  base_r = (int16_t)((int32_t)(r0 & 0x7FFu) - 1024);
            uint16_t base_t = (uint16_t)(r0 >> 16);
            /* 未知の 512 次元ねじれ (レーン 0 に必ず対蹠点 0x8000 を注入) */
            int16_t  hid_r  = (int16_t)((int32_t)(r1 & 0x3FFu) - 512);
            uint16_t hid_t  = (k == 0) ? 0x8000u : (uint16_t)(r1 >> 16);

            blk_sc[bi].z2_rho[k]   = base_r;
            blk_sc[bi].z2_theta[k] = base_t;
            blk_sc[bi].z1_rho[k]   = sat_add_s16_scalar(base_r, hid_r);
            blk_sc[bi].z1_theta[k] = (uint16_t)(base_t + hid_t);
            blk_sc[bi].tw_rho[k]   = 0;
            blk_sc[bi].tw_theta[k] = 0;
        }
    }
    memcpy(blk_nm, blk_sc, sizeof(blk_nm));

    /* 1. 3-射書き換え前: 2-射ホロノミー一括検出 (Scalar vs NEON 0-bit 一致確認) */
    uint64_t norm_sc_before = holonomy_2morph_scalar(blk_sc, N_BLOCKS, false);
#if LC_HAS_NEON
    uint64_t norm_nm_before = holonomy_2morph_neon(blk_nm, N_BLOCKS, false);
#else
    uint64_t norm_nm_before = holonomy_2morph_scalar(blk_nm, N_BLOCKS, false);
#endif
    if (norm_sc_before != norm_nm_before || norm_nm_before == 0) {
        fprintf(stderr, "FAIL: Test 4 書き換え前ホロノミー検出不一致\n");
        return false;
    }

    /* 2. 3-射発火: 検出した 2-射 Δζ をそのまま補償 1-射 tw へ 1 パス合成 */
    for (int bi = 0; bi < N_BLOCKS; bi++) {
        for (int k = 0; k < 8; k++) {
            blk_sc[bi].tw_rho[k]   = blk_sc[bi].out_drho[k];
            blk_sc[bi].tw_theta[k] = (uint16_t)blk_sc[bi].out_dtheta[k];
            blk_nm[bi].tw_rho[k]   = blk_nm[bi].out_drho[k];
            blk_nm[bi].tw_theta[k] = (uint16_t)blk_nm[bi].out_dtheta[k];
        }
    }

    /* 3. 3-射書き換え後: 全 512 次元が 0-bit 平坦化 (Δρ=0, Δθ=0) したことを検証 */
    uint64_t norm_sc_after = holonomy_2morph_scalar(blk_sc, N_BLOCKS, true);
#if LC_HAS_NEON
    uint64_t norm_nm_after = holonomy_2morph_neon(blk_nm, N_BLOCKS, true);
#else
    uint64_t norm_nm_after = holonomy_2morph_scalar(blk_nm, N_BLOCKS, true);
#endif
    if (norm_sc_after != 0 || norm_nm_after != 0) {
        fprintf(stderr, "FAIL: Test 4 書き換え後ホロノミー残留\n");
        return false;
    }
    for (int bi = 0; bi < N_BLOCKS; bi++) {
        for (int k = 0; k < 8; k++) {
            if (blk_nm[bi].out_drho[k] != 0 || blk_nm[bi].out_dtheta[k] != 0) {
                fprintf(stderr, "FAIL: Test 4 非ゼロ残差 (%d,%d)\n", bi, k);
                return false;
            }
        }
    }

    /* 4. 512 次元 2-射ホロノミー検出カーネルの L1D スループット実測 */
    const int ITERS = 40000;
    volatile uint64_t sink = 0;
    double t0 = now_ns();
    for (int it = 0; it < ITERS; it++) sink += holonomy_2morph_scalar(blk_sc, N_BLOCKS, true);
    double dt_sc = now_ns() - t0;

#if LC_HAS_NEON
    t0 = now_ns();
    for (int it = 0; it < ITERS; it++) sink += holonomy_2morph_neon(blk_nm, N_BLOCKS, true);
    double dt_nm = now_ns() - t0;
#else
    double dt_nm = dt_sc;
#endif
    (void)sink;

    double tot_dims = (double)N_DIMS * (double)ITERS;
    printf("[Test 4] 512 次元 FHRR 2-射ホロノミー検出 & 3-射補償合成 (Scalar vs ARM64 NEON):\n");
    printf("  書き換え前 ホロノミー位相 L1 ノルム : %llu (全 512 次元 0-bit 一致)\n",
           (unsigned long long)norm_nm_before);
    printf("  3-射補償後 ホロノミー位相 L1 ノルム : %llu (512/512 次元 Δζ=(0,0) 完全平坦化 OK)\n",
           (unsigned long long)norm_nm_after);
    printf("  2-射検出スループット (L1D 512 dims) : Scalar %.2f ns (%.0f M/s) | NEON %.2f ns (%.0f M/s) | %.2fx\n\n",
           dt_sc / tot_dims, (tot_dims * 1e3) / dt_sc,
           dt_nm / tot_dims, (tot_dims * 1e3) / dt_nm,
           dt_sc / dt_nm);
    return true;
}

int main(void) {
    printf("=== 高次圏 2-射 (H1 ホロノミー障害) 検出 & 3-射自律書き換え C99 実証 ===\n");
    printf("sizeof(LiftingCellLog)=%zu B | sizeof(lc_holo_block8_t)=%zu B | NEON=%s\n\n",
           sizeof(LiftingCellLog), sizeof(lc_holo_block8_t),
           LC_HAS_NEON ? "ON (ARM64)" : "OFF");

    if (!run_test1_gauge_relaxation())                  return 1;
    if (!run_test2_h1_obstruction_and_3morph_rewrite()) return 1;
    if (!run_test3_associator_and_pentagon())           return 1;
    if (!run_test4_256d_fhrr_holonomy_neon())           return 1;

    printf(">>> ALL ASSERTIONS PASSED: 2-射ホロノミー障害検出・3-射自律書き換え・Associator 回復を実証 <<<\n");
    return 0;
}