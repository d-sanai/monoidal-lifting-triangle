/* lifting_log_fhrr_neon.c — 64-byte Log-Polar FHRR LiftingCell + ARM64 NEON Prototype */
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
 * 1. 複素対数極座標 (lc_logc16_t) & 双対直交ガウス整数 (lc_c16_t)
 * ============================================================================ */
#define LC_Q14_ONE    16384
#define LC_Q14_HALF    8192
#define LC_LOG_SCALE    256          /* Q7.8: +256 = x2.0, -256 = x0.5, 0 = x1.0 */
#define LC_LOG_ZERO  (-32768)        /* INT16_MIN: 対数空間の吸収元 ⊥ (厳密ゼロ) */

#define LC_TRIG_BITS  10
#define LC_TRIG_SIZE  (1 << LC_TRIG_BITS) /* 1024 点 (cos/sin 合計 4 KB: L1D 完全常駐) */
#define LC_TRIG_SHIFT (16 - LC_TRIG_BITS) /* theta >> 6 */

typedef struct {
    int16_t  rho;    /* Q7.8 log2(|z|), INT16_MIN = ⊥ */
    uint16_t theta;  /* Z / 2^16 Z 円環位相 */
} lc_logc16_t;

typedef struct {
    int16_t re;
    int16_t im;
} lc_c16_t;

typedef char lc_assert_logc16_4b[(sizeof(lc_logc16_t) == 4) ? 1 : -1];

/* 64-byte 均一 Log-Polar FHRR LiftingCell (パディングゼロ・キャッシュライン整列) */
typedef struct LiftingCellLog {
    /* 動的 DAG 配線トポロジー (24 bytes) */
    struct LiftingCellLog *in_s;
    struct LiftingCellLog *in_g;
    struct LiftingCellLog *in_v;

    /* Log-Polar FHRR チャート: 0-bit Binding 状態 (12 bytes) */
    lc_logc16_t w;
    lc_logc16_t s_eff;
    lc_logc16_t p_log;

    /* 双対直交チャート: 0-bit 可逆 Bundling (μ∘Δ = id) 状態 (12 bytes) */
    lc_c16_t p_cart;
    lc_c16_t r_cart;
    lc_c16_t restored_cart;

    /* 逆流対数強度・位相トルク (8 bytes) */
    int32_t adj_rho;
    int32_t adj_theta;

    /* 参照カウント & メタデータ (8 bytes -> 64-bit 環境で厳密に 64 bytes) */
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

/* 320-byte AoSoA-8 ブロック (5 キャッシュライン / 8 セル = 40 B/cell, 非2べき乗ストライド) */
typedef struct LC_ALIGN64 {
    /* Log-Polar 入力・重み (96 bytes) */
    int16_t  s_rho[8];
    uint16_t s_theta[8];
    int16_t  g_rho[8];
    uint16_t g_theta[8];
    int16_t  w_rho[8];
    uint16_t w_theta[8];

    /* 直交チャート変分入力 v (32 bytes) */
    int16_t  v_re[8];
    int16_t  v_im[8];

    /* Log-Polar Binding 中間結果 (64 bytes) */
    int16_t  se_rho[8];
    uint16_t se_theta[8];
    int16_t  p_rho[8];
    uint16_t p_theta[8];

    /* 双対チャート変換・Δ/μ 出力 (96 bytes) */
    int16_t  p_re[8];
    int16_t  p_im[8];
    int16_t  r_re[8];
    int16_t  r_im[8];
    int16_t  res_re[8];
    int16_t  res_im[8];

    /* アンバインディング検証出力 (32 bytes -> 合計 320 bytes = 64B x 5) */
    int16_t  unb_rho[8];
    uint16_t unb_theta[8];
} lc_log_block8_t;

typedef char lc_assert_log_block8_320b[(sizeof(lc_log_block8_t) == 320) ? 1 : -1];

/* ============================================================================
 * 2. -lm 不要の L1D 常駐双対チャート LUT (cos/sin 4KB + 2^frac 512B)
 * ============================================================================ */
static int16_t  LC_ALIGN64 g_cos_lut[LC_TRIG_SIZE];
static int16_t  LC_ALIGN64 g_sin_lut[LC_TRIG_SIZE];
static uint16_t LC_ALIGN64 g_exp2_frac_lut[256];

static double taylor_sin(double x) {
    static const double PI = 3.14159265358979323846;
    while (x >  PI) x -= 2.0 * PI;
    while (x < -PI) x += 2.0 * PI;
    double x2 = x * x, term = x, sum = x;
    for (int k = 1; k <= 10; k++) {
        term *= -x2 / (double)((2 * k) * (2 * k + 1));
        sum  += term;
    }
    return sum;
}

static double taylor_cos(double x) {
    static const double PI = 3.14159265358979323846;
    while (x >  PI) x -= 2.0 * PI;
    while (x < -PI) x += 2.0 * PI;
    double x2 = x * x, term = 1.0, sum = 1.0;
    for (int k = 1; k <= 10; k++) {
        term *= -x2 / (double)((2 * k - 1) * (2 * k));
        sum  += term;
    }
    return sum;
}

static double taylor_exp2_frac(double f) {
    static const double LN2 = 0.693147180559945309417;
    double x = f * LN2, term = 1.0, sum = 1.0;
    for (int k = 1; k <= 12; k++) {
        term *= x / (double)k;
        sum  += term;
    }
    return sum;
}

static void init_dual_chart_luts(void) {
    static const double TWO_PI = 6.28318530717958647692;
    for (int i = 0; i < LC_TRIG_SIZE; i++) {
        double ang = TWO_PI * (double)i / (double)LC_TRIG_SIZE;
        double c   = taylor_cos(ang) * (double)LC_Q14_ONE;
        double s   = taylor_sin(ang) * (double)LC_Q14_ONE;
        g_cos_lut[i] = (int16_t)((c >= 0.0) ? (c + 0.5) : (c - 0.5));
        g_sin_lut[i] = (int16_t)((s >= 0.0) ? (s + 0.5) : (s - 0.5));
    }
    for (int f = 0; f < 256; f++) {
        double v = taylor_exp2_frac((double)f / 256.0) * (double)LC_Q14_ONE;
        g_exp2_frac_lut[f] = (uint16_t)(v + 0.5);
    }
}

/* ============================================================================
 * 3. スカラー厳密参照演算 (0-bit 仕様定義)
 * ============================================================================ */
static inline int16_t log_add_rho_scalar(int16_t a, int16_t b) {
    if (a == LC_LOG_ZERO || b == LC_LOG_ZERO) return LC_LOG_ZERO;
    int32_t s = (int32_t)a + (int32_t)b;
    if (s > 32767)       return 32767;
    if (s <= LC_LOG_ZERO) return LC_LOG_ZERO;
    return (int16_t)s;
}

static inline lc_logc16_t fhrr_bind_scalar(lc_logc16_t a, lc_logc16_t b) {
    lc_logc16_t out = {
        log_add_rho_scalar(a.rho, b.rho),
        (uint16_t)(a.theta + b.theta)
    };
    return out;
}

static inline lc_logc16_t fhrr_unbind_scalar(lc_logc16_t a, lc_logc16_t key) {
    /* 位相の逆元は (uint16_t)(0 - key.theta)、強度の逆元は -key.rho */
    int16_t inv_rho = (key.rho == LC_LOG_ZERO) ? LC_LOG_ZERO : (int16_t)(-key.rho);
    lc_logc16_t out = {
        log_add_rho_scalar(a.rho, inv_rho),
        (uint16_t)(a.theta - key.theta)
    };
    return out;
}

static inline int16_t quantize_q14_s32_scalar(int32_t acc) {
    int32_t mag = (acc >= 0) ? acc : -acc;
    int32_t q   = (mag + LC_Q14_HALF) >> 14;
    if (acc < 0) q = -q;
    if (q > 32767)  return 32767;
    if (q < -32768) return -32768;
    return (int16_t)q;
}

/* Log-Polar (rho, theta) -> 直交 (re, im) Q1.14 変換
 * rho == LC_LOG_ZERO (-32768) は e = -128 -> clamp(-16) により分岐なしで振幅 0 */
static inline lc_c16_t log_to_cart_scalar(int16_t rho, uint16_t theta) {
    int32_t e = (int32_t)(rho >> 8);
    if (e < -16) e = -16;
    if (e >   2) e =   2;

    int32_t base = (int32_t)g_exp2_frac_lut[(uint8_t)(rho & 0xFF)];
    int32_t mag  = (e >= 0) ? (base << e) : (base >> (-e));
    if (mag > 32767) mag = 32767;

    uint32_t idx = (uint32_t)(theta >> LC_TRIG_SHIFT);
    lc_c16_t out = {
        quantize_q14_s32_scalar(mag * (int32_t)g_cos_lut[idx]),
        quantize_q14_s32_scalar(mag * (int32_t)g_sin_lut[idx])
    };
    return out;
}

static LC_NOINLINE void aosoa8_bind_only_scalar(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            lc_logc16_t s  = { b->s_rho[k], b->s_theta[k] };
            lc_logc16_t g  = { b->g_rho[k], b->g_theta[k] };
            lc_logc16_t w  = { b->w_rho[k], b->w_theta[k] };
            lc_logc16_t se = fhrr_bind_scalar(g, s);
            lc_logc16_t pl = fhrr_bind_scalar(w, se);
            b->se_rho[k]   = se.rho; b->se_theta[k] = se.theta;
            b->p_rho[k]    = pl.rho; b->p_theta[k]  = pl.theta;
        }
    }
}

static LC_NOINLINE void aosoa8_dual_step_scalar(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            lc_logc16_t s   = { b->s_rho[k], b->s_theta[k] };
            lc_logc16_t g   = { b->g_rho[k], b->g_theta[k] };
            lc_logc16_t w   = { b->w_rho[k], b->w_theta[k] };
            lc_logc16_t se  = fhrr_bind_scalar(g, s);
            lc_logc16_t pl  = fhrr_bind_scalar(w, se);
            lc_logc16_t unb = fhrr_unbind_scalar(pl, w);
            lc_c16_t    pc  = log_to_cart_scalar(pl.rho, pl.theta);

            int16_t r_re   = (int16_t)((uint16_t)b->v_re[k] - (uint16_t)pc.re);
            int16_t r_im   = (int16_t)((uint16_t)b->v_im[k] - (uint16_t)pc.im);
            int16_t res_re = (int16_t)((uint16_t)r_re + (uint16_t)pc.re);
            int16_t res_im = (int16_t)((uint16_t)r_im + (uint16_t)pc.im);

            b->se_rho[k]    = se.rho;  b->se_theta[k]  = se.theta;
            b->p_rho[k]     = pl.rho;  b->p_theta[k]   = pl.theta;
            b->unb_rho[k]   = unb.rho; b->unb_theta[k] = unb.theta;
            b->p_re[k]      = pc.re;   b->p_im[k]      = pc.im;
            b->r_re[k]      = r_re;    b->r_im[k]      = r_im;
            b->res_re[k]    = res_re;  b->res_im[k]    = res_im;
        }
    }
}

/* ============================================================================
 * 4. ARM64 NEON 8 セル同時並列カーネル (0-bit スカラー完全一致)
 * ============================================================================ */
#if LC_HAS_NEON

static inline int16x8_t log_add_rho_x8_neon(int16x8_t a, int16x8_t b) {
    const int16x8_t z = vdupq_n_s16(LC_LOG_ZERO);
    uint16x8_t is_z = vorrq_u16(vceqq_s16(a, z), vceqq_s16(b, z));
    return vbslq_s16(is_z, z, vqaddq_s16(a, b));
}

static inline int16x8_t log_neg_rho_x8_neon(int16x8_t a) {
    /* -(-32768) は 2 の補数で -32768 (LC_LOG_ZERO) のまま保存される */
    return vnegq_s16(a);
}

static inline int32x4_t quantize_q14_s32_x4_neon(int32x4_t acc) {
    const int32x4_t z32    = vdupq_n_s32(0);
    const int32x4_t half32 = vdupq_n_s32(LC_Q14_HALF);
    uint32x4_t neg = vcltq_s32(acc, z32);
    int32x4_t  mag = vshrq_n_s32(vaddq_s32(vabsq_s32(acc), half32), 14);
    return vbslq_s32(neg, vnegq_s32(mag), mag);
}

static inline void log_to_cart_x8_neon(const int16_t rho[8], const uint16_t theta[8],
                                       int16x8_t *out_re, int16x8_t *out_im) {
    /* 1. L1D 常駐 LUT から 8 セル分の (exp2_frac, cos, sin) を一括ロード */
    int16_t tmp_base[8], tmp_cos[8], tmp_sin[8];
    for (int k = 0; k < 8; k++) {
        uint32_t idx = (uint32_t)(theta[k] >> LC_TRIG_SHIFT);
        tmp_base[k]  = (int16_t)g_exp2_frac_lut[(uint8_t)(rho[k] & 0xFF)];
        tmp_cos[k]   = g_cos_lut[idx];
        tmp_sin[k]   = g_sin_lut[idx];
    }

    int16x8_t  v_rho  = vld1q_s16(rho);
    int16x8_t  v_e16  = vshrq_n_s16(v_rho, 8);
    uint16x8_t v_base = vreinterpretq_u16_s16(vld1q_s16(tmp_base));
    int16x8_t  v_cos  = vld1q_s16(tmp_cos);
    int16x8_t  v_sin  = vld1q_s16(tmp_sin);

    const int32x4_t min_e  = vdupq_n_s32(-16);
    const int32x4_t max_e  = vdupq_n_s32(2);
    const int32x4_t max15  = vdupq_n_s32(32767);

    /* 2. 可変ビットシフト (vshlq_s32) で 8 セル同時に 2^e * 2^frac -> mag (0..32767) */
    int32x4_t e_lo = vmaxq_s32(vminq_s32(vmovl_s16(vget_low_s16(v_e16)), max_e), min_e);
    int32x4_t e_hi = vmaxq_s32(vminq_s32(vmovl_high_s16(v_e16), max_e), min_e);

    int32x4_t b_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_base)));
    int32x4_t b_hi = vreinterpretq_s32_u32(vmovl_high_u16(v_base));

    int32x4_t m_lo = vminq_s32(vshlq_s32(b_lo, e_lo), max15);
    int32x4_t m_hi = vminq_s32(vshlq_s32(b_hi, e_hi), max15);

    /* 3. mag * cos, mag * sin -> Q1.14 符号対称量子化 */
    int32x4_t c_lo = vmovl_s16(vget_low_s16(v_cos)), c_hi = vmovl_high_s16(v_cos);
    int32x4_t s_lo = vmovl_s16(vget_low_s16(v_sin)), s_hi = vmovl_high_s16(v_sin);

    int32x4_t re_lo = quantize_q14_s32_x4_neon(vmulq_s32(m_lo, c_lo));
    int32x4_t re_hi = quantize_q14_s32_x4_neon(vmulq_s32(m_hi, c_hi));
    int32x4_t im_lo = quantize_q14_s32_x4_neon(vmulq_s32(m_lo, s_lo));
    int32x4_t im_hi = quantize_q14_s32_x4_neon(vmulq_s32(m_hi, s_hi));

    *out_re = vqmovn_high_s32(vqmovn_s32(re_lo), re_hi);
    *out_im = vqmovn_high_s32(vqmovn_s32(im_lo), im_hi);
}

/* [Kernel A] 純粋 Log-Polar FHRR Binding のみ (0-bit 加算・8 セル同時) */
static LC_NOINLINE void aosoa8_bind_only_neon(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
        int16x8_t  sr = vld1q_s16(b->s_rho),   gr = vld1q_s16(b->g_rho),   wr = vld1q_s16(b->w_rho);
        uint16x8_t st = vld1q_u16(b->s_theta), gt = vld1q_u16(b->g_theta), wt = vld1q_u16(b->w_theta);

        int16x8_t  ser = log_add_rho_x8_neon(gr, sr);
        uint16x8_t set = vaddq_u16(gt, st);
        int16x8_t  pr  = log_add_rho_x8_neon(wr, ser);
        uint16x8_t pt  = vaddq_u16(wt, set);

        vst1q_s16(b->se_rho, ser); vst1q_u16(b->se_theta, set);
        vst1q_s16(b->p_rho,  pr);  vst1q_u16(b->p_theta,  pt);
    }
}

/* [Kernel B] 双対チャート統合ステップ (FHRR Bind + Unbind + LUT 変換 + Δ/μ 可逆 Bundling) */
static LC_NOINLINE void aosoa8_dual_step_neon(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
        int16x8_t  sr = vld1q_s16(b->s_rho),   gr = vld1q_s16(b->g_rho),   wr = vld1q_s16(b->w_rho);
        uint16x8_t st = vld1q_u16(b->s_theta), gt = vld1q_u16(b->g_theta), wt = vld1q_u16(b->w_theta);

        int16x8_t  ser = log_add_rho_x8_neon(gr, sr);
        uint16x8_t set = vaddq_u16(gt, st);
        int16x8_t  pr  = log_add_rho_x8_neon(wr, ser);
        uint16x8_t pt  = vaddq_u16(wt, set);

        int16x8_t  unbr = log_add_rho_x8_neon(pr, log_neg_rho_x8_neon(wr));
        uint16x8_t unbt = vsubq_u16(pt, wt);

        vst1q_s16(b->se_rho,  ser);  vst1q_u16(b->se_theta,  set);
        vst1q_s16(b->p_rho,   pr);   vst1q_u16(b->p_theta,   pt);
        vst1q_s16(b->unb_rho, unbr); vst1q_u16(b->unb_theta, unbt);

        int16x8_t p_re, p_im;
        log_to_cart_x8_neon(b->p_rho, b->p_theta, &p_re, &p_im);

        int16x8_t v_re = vld1q_s16(b->v_re), v_im = vld1q_s16(b->v_im);
        int16x8_t r_re = vreinterpretq_s16_u16(vsubq_u16(vreinterpretq_u16_s16(v_re), vreinterpretq_u16_s16(p_re)));
        int16x8_t r_im = vreinterpretq_s16_u16(vsubq_u16(vreinterpretq_u16_s16(v_im), vreinterpretq_u16_s16(p_im)));
        int16x8_t res_re = vreinterpretq_s16_u16(vaddq_u16(vreinterpretq_u16_s16(r_re), vreinterpretq_u16_s16(p_re)));
        int16x8_t res_im = vreinterpretq_s16_u16(vaddq_u16(vreinterpretq_u16_s16(r_im), vreinterpretq_u16_s16(p_im)));

        vst1q_s16(b->p_re,   p_re);   vst1q_s16(b->p_im,   p_im);
        vst1q_s16(b->r_re,   r_re);   vst1q_s16(b->r_im,   r_im);
        vst1q_s16(b->res_re, res_re); vst1q_s16(b->res_im, res_im);
    }
}
#endif /* LC_HAS_NEON */

/* ============================================================================
 * 5. [Test 2] 256 次元 Log-Polar FHRR 連想記憶 (Bundling -> Unbinding 想起)
 * ============================================================================ */
static inline uint32_t next_rng(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static bool run_fhrr_associative_memory_test(void) {
    enum { D = 256, N_ITEMS = 8 };
    static uint16_t keys[N_ITEMS][D];
    static uint16_t vals[N_ITEMS][D];
    /* 対数重み (Weber-Fechner 階層): Item 0 = -1 oct (0.5x), 1..6 = 0 oct (1.0x), 7 = +1 oct (2.0x) */
    const int16_t item_rho[N_ITEMS] = { -256, 0, 0, 0, 0, 0, 0, 256 };

    uint32_t rng = 0x20261123u;
    for (int m = 0; m < N_ITEMS; m++) {
        for (int d = 0; d < D; d++) {
            keys[m][d] = (uint16_t)(next_rng(&rng) & 0xFFFFu);
            vals[m][d] = (uint16_t)(next_rng(&rng) & 0xFFFFu);
        }
    }

    /* 1. 直交チャート上で 8 組の FHRR Binding (K_m ⊛ V_m) をモジュラ Bundling */
    int32_t bundle_re[D], bundle_im[D];
    memset(bundle_re, 0, sizeof(bundle_re));
    memset(bundle_im, 0, sizeof(bundle_im));
    for (int m = 0; m < N_ITEMS; m++) {
        for (int d = 0; d < D; d++) {
            /* 0-bit FHRR バインディング (位相加算 + 対数重み加算) -> 双対チャート変換 */
            lc_logc16_t kv = fhrr_bind_scalar((lc_logc16_t){ item_rho[m], keys[m][d] },
                                              (lc_logc16_t){ 0,           vals[m][d] });
            lc_c16_t    c  = log_to_cart_scalar(kv.rho, kv.theta);
            bundle_re[d] += c.re;
            bundle_im[d] += c.im;
        }
    }

    /* 2. 各キー K_q で 0-bit アンバインディング (位相回転 -theta_K) して全候補と内積照合 */
    double min_correct_sim = 1e9, max_noise_sim = -1e9;
    double sim_low = 0.0, sim_mid = 0.0, sim_high = 0.0;

    for (int q = 0; q < N_ITEMS; q++) {
        int    best_m   = -1;
        double best_sim = -1e9;
        for (int cand = 0; cand < N_ITEMS; cand++) {
            /* 内積 Re( Σ bundle[d] * exp(-i*(K_q[d] + V_cand[d])) ) */
            int64_t dot = 0;
            for (int d = 0; d < D; d++) {
                uint16_t rot = (uint16_t)(keys[q][d] + vals[cand][d]);
                uint32_t idx = (uint32_t)(rot >> LC_TRIG_SHIFT);
                dot += (int64_t)bundle_re[d] * g_cos_lut[idx]
                     + (int64_t)bundle_im[d] * g_sin_lut[idx];
            }
            double sim = (double)dot / ((double)D * (double)LC_Q14_ONE * (double)LC_Q14_ONE);
            if (sim > best_sim) { best_sim = sim; best_m = cand; }
            if (cand == q) {
                if (sim < min_correct_sim) min_correct_sim = sim;
                if (q == 0) sim_low  = sim;
                if (q == 1) sim_mid  = sim;
                if (q == 7) sim_high = sim;
            } else if (sim > max_noise_sim) {
                max_noise_sim = sim;
            }
        }
        if (best_m != q) return false;
    }

    printf("[Test 2] 256 次元 Log-Polar FHRR 連想記憶 (8 ペア重畳 -> 0-bit アンバインディング): 8/8 正解 (OK)\n");
    printf("         対数重み応答: -1 oct=%.2f | 0 oct=%.2f | +1 oct=%.2f (ノイズ上限=%.2f)\n",
           sim_low, sim_mid, sim_high, max_noise_sim);
    return (min_correct_sim > max_noise_sim * 1.25) && (sim_high > sim_mid * 1.5) && (sim_mid > sim_low * 1.5);
}

/* ============================================================================
 * 6. データ生成・0-bit 一致検証・スループットベンチマーク
 * ============================================================================ */
#define MAX_CELLS  16384
#define MAX_BLOCKS (MAX_CELLS / 8)

static void populate_blocks(lc_log_block8_t *blocks, int n_blocks) {
    static const int16_t rho_corners[] = {
        LC_LOG_ZERO, -32767, -8192, -4096, -3840, -1024, -256, -1, 0, 1, 128, 256, 512, 1024, 32767
    };
    const int nc = (int)(sizeof(rho_corners) / sizeof(rho_corners[0]));
    uint32_t rng = 0x20261123u;

    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
        for (int k = 0; k < 8; k++) {
            int idx = bi * 8 + k;
            if (idx < 512) {
                b->s_rho[k]   = rho_corners[(idx * 3 + 0) % nc];
                b->g_rho[k]   = rho_corners[(idx * 5 + 1) % nc];
                b->w_rho[k]   = rho_corners[(idx * 7 + 2) % nc];
                b->s_theta[k] = (uint16_t)((idx * 1021) & 0xFFFFu);
                b->g_theta[k] = (uint16_t)((idx * 4093) & 0xFFFFu);
                b->w_theta[k] = (uint16_t)((idx * 8191) & 0xFFFFu);
            } else {
                uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng), r2 = next_rng(&rng);
                b->s_rho[k]   = ((idx % 17) == 0) ? LC_LOG_ZERO : (int16_t)((int32_t)(r0 & 0x7FFu) - 1024);
                b->g_rho[k]   = ((idx % 23) == 0) ? LC_LOG_ZERO : (int16_t)((int32_t)(r1 & 0x3FFu) - 512);
                b->w_rho[k]   = (int16_t)((int32_t)(r2 & 0x3FFu) - 512);
                b->s_theta[k] = (uint16_t)(r0 >> 16);
                b->g_theta[k] = (uint16_t)(r1 >> 16);
                b->w_theta[k] = (uint16_t)(r2 >> 16);
            }
            b->v_re[k] = (int16_t)((idx * 257 - 16384) & 0xFFFF);
            b->v_im[k] = (int16_t)((idx * 131 +  8192) & 0xFFFF);
        }
    }
}

static bool verify_blocks_exact(const lc_log_block8_t *sc, const lc_log_block8_t *nm, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        for (int k = 0; k < 8; k++) {
            /* μ ∘ Δ = id の 0-bit 可逆性確認 */
            if (nm[bi].res_re[k] != nm[bi].v_re[k] || nm[bi].res_im[k] != nm[bi].v_im[k]) {
                fprintf(stderr, "FAIL: 双対チャート可逆性違反 (%d,%d)\n", bi, k);
                return false;
            }
            /* FHRR 位相アンバインディングの 0-bit 完全可逆性確認 */
            if (nm[bi].unb_theta[k] != nm[bi].se_theta[k]) {
                fprintf(stderr, "FAIL: FHRR 位相アンバインディング不一致 (%d,%d)\n", bi, k);
                return false;
            }
            /* 全フィールドのスカラー vs NEON 0-bit 一致確認 */
            if (sc[bi].se_rho[k]    != nm[bi].se_rho[k]    || sc[bi].se_theta[k]  != nm[bi].se_theta[k]  ||
                sc[bi].p_rho[k]     != nm[bi].p_rho[k]     || sc[bi].p_theta[k]   != nm[bi].p_theta[k]   ||
                sc[bi].unb_rho[k]   != nm[bi].unb_rho[k]   || sc[bi].unb_theta[k] != nm[bi].unb_theta[k] ||
                sc[bi].p_re[k]      != nm[bi].p_re[k]      || sc[bi].p_im[k]      != nm[bi].p_im[k]      ||
                sc[bi].r_re[k]      != nm[bi].r_re[k]      || sc[bi].r_im[k]      != nm[bi].r_im[k]) {
                fprintf(stderr, "FAIL: スカラー vs NEON 0-bit 不一致 (%d,%d)\n", bi, k);
                return false;
            }
        }
    }
    return true;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void run_bench(const char *label, int n_cells, int iters,
                      lc_log_block8_t *sc, lc_log_block8_t *nm) {
    int n_blocks = n_cells / 8;
    double tot = (double)n_cells * (double)iters;

    populate_blocks(sc, n_blocks);
    memcpy(nm, sc, (size_t)n_blocks * sizeof(lc_log_block8_t));

    /* ウォームアップ */
    aosoa8_dual_step_scalar(sc, n_blocks);
#if LC_HAS_NEON
    aosoa8_dual_step_neon(nm, n_blocks);
#endif

    double t0, dt_sc_b, dt_nm_b, dt_sc_d, dt_nm_d;

    t0 = now_ns();
    for (int it = 0; it < iters; it++) aosoa8_bind_only_scalar(sc, n_blocks);
    dt_sc_b = now_ns() - t0;

#if LC_HAS_NEON
    t0 = now_ns();
    for (int it = 0; it < iters; it++) aosoa8_bind_only_neon(nm, n_blocks);
    dt_nm_b = now_ns() - t0;
#else
    dt_nm_b = dt_sc_b;
#endif

    t0 = now_ns();
    for (int it = 0; it < iters; it++) aosoa8_dual_step_scalar(sc, n_blocks);
    dt_sc_d = now_ns() - t0;

#if LC_HAS_NEON
    t0 = now_ns();
    for (int it = 0; it < iters; it++) aosoa8_dual_step_neon(nm, n_blocks);
    dt_nm_d = now_ns() - t0;
#else
    dt_nm_d = dt_sc_d;
#endif

    if (!verify_blocks_exact(sc, nm, n_blocks)) exit(1);

    printf("%s (N=%d cells, %d KB, iters=%d)\n",
           label, n_cells, (n_blocks * (int)sizeof(lc_log_block8_t)) / 1024, iters);
    printf("  1. Pure Log-Polar FHRR Bind : Scalar %5.2f ns (%4.0f M/s) | NEON %5.2f ns (%5.0f M/s) | %5.2fx\n",
           dt_sc_b / tot, (tot * 1e3) / dt_sc_b, dt_nm_b / tot, (tot * 1e3) / dt_nm_b, dt_sc_b / dt_nm_b);
    printf("  2. Dual-Chart Bind+LUT+Δ/μ  : Scalar %5.2f ns (%4.0f M/s) | NEON %5.2f ns (%5.0f M/s) | %5.2fx\n\n",
           dt_sc_d / tot, (tot * 1e3) / dt_sc_d, dt_nm_d / tot, (tot * 1e3) / dt_nm_d, dt_sc_d / dt_nm_d);
}

int main(void) {
    static lc_log_block8_t LC_ALIGN64 blk_sc[MAX_BLOCKS];
    static lc_log_block8_t LC_ALIGN64 blk_nm[MAX_BLOCKS];

    init_dual_chart_luts();

    printf("=== 64-byte Log-Polar FHRR LiftingCell & 双対チャート ARM64 NEON 検証 ===\n");
    printf("sizeof(LiftingCellLog)=%zu B | sizeof(lc_log_block8_t)=%zu B | LUT=%.1f KB | NEON=%s\n\n",
           sizeof(LiftingCellLog), sizeof(lc_log_block8_t),
           (double)(sizeof(g_cos_lut) + sizeof(g_sin_lut) + sizeof(g_exp2_frac_lut)) / 1024.0,
           LC_HAS_NEON ? "ON (ARM64)" : "OFF");

    /* [Test 1] 吸収元 ⊥ (INT16_MIN) & 位相群可逆性の網羅検証 */
    for (int t = 0; t < 65536; t += 17) {
        lc_logc16_t a   = { (int16_t)((t % 512) - 256), (uint16_t)t };
        lc_logc16_t k   = { 128, (uint16_t)(t * 3 + 12345) };
        lc_logc16_t bnd = fhrr_bind_scalar(a, k);
        lc_logc16_t unb = fhrr_unbind_scalar(bnd, k);
        if (unb.rho != a.rho || unb.theta != a.theta) {
            fprintf(stderr, "FAIL: Test 1 FHRR 可逆性違反\n");
            return 1;
        }
        lc_logc16_t z = fhrr_bind_scalar((lc_logc16_t){ LC_LOG_ZERO, (uint16_t)t }, k);
        lc_c16_t   zc = log_to_cart_scalar(z.rho, z.theta);
        if (z.rho != LC_LOG_ZERO || zc.re != 0 || zc.im != 0) {
            fprintf(stderr, "FAIL: Test 1 吸収元 ⊥ 違反\n");
            return 1;
        }
    }
    printf("[Test 1] 吸収元 ⊥ (INT16_MIN) & 0-bit FHRR Bind/Unbind 可逆性: OK\n");

    /* [Test 2] 256 次元 Log-Polar FHRR 連想記憶 */
    if (!run_fhrr_associative_memory_test()) {
        fprintf(stderr, "FAIL: Test 2 FHRR 連想記憶テスト失敗\n");
        return 1;
    }

    /* [Test 3 & 4] 16,384 セル 0-bit 一致 + L1D / L2 スループット実測 */
    populate_blocks(blk_sc, MAX_BLOCKS);
    memcpy(blk_nm, blk_sc, sizeof(blk_nm));
    aosoa8_dual_step_scalar(blk_sc, MAX_BLOCKS);
#if LC_HAS_NEON
    aosoa8_dual_step_neon(blk_nm, MAX_BLOCKS);
#else
    aosoa8_dual_step_scalar(blk_nm, MAX_BLOCKS);
#endif
    if (!verify_blocks_exact(blk_sc, blk_nm, MAX_BLOCKS)) return 1;
    printf("[Test 3] 16,384 セル (境界値・⊥・全象限) Scalar vs ARM64 NEON 0-bit 一致: OK\n\n");

    run_bench("[Test 4-A] L1D 常駐ワークロード", 512,   40000, blk_sc, blk_nm);
    run_bench("[Test 4-B] L2  常駐ワークロード", 16384,  2000, blk_sc, blk_nm);

    printf(">>> ALL ASSERTIONS PASSED: 64-byte Log-Polar FHRR + 双対チャート 0-bit 厳密動作を確認 <<<\n");
    return 0;
}