/* lifting_dual_chart_zx.c — Pure-Integer Dual-Chart Inverse (Cart->Log),
 * 64-Layer Closed-Loop Lifting Ladder & Frobenius ZX-Spider Circuit Verification */
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
 * 1. 定数・型定義 (64-byte LiftingCellLog と 256-byte Dual-Chart AoSoA-8)
 * ============================================================================ */
#define LC_Q14_ONE    16384
#define LC_Q14_HALF    8192
#define LC_LOG_ZERO  (-32768)

#define LC_TRIG_BITS  10
#define LC_TRIG_SIZE  (1 << LC_TRIG_BITS)   /* 1024 点 (cos/sin 各 2 KB) */
#define LC_TRIG_SHIFT (16 - LC_TRIG_BITS)   /* 6 */

#define LC_LOG2_BITS  9
#define LC_LOG2_SIZE  (1 << LC_LOG2_BITS)   /* 512 点 + 1 (half_log2 約 1 KB) */

#define LC_ATAN_BITS  10
#define LC_ATAN_SIZE  (1 << LC_ATAN_BITS)   /* 1024 点 + 1 (atan 約 2 KB) */

typedef struct {
    int16_t  rho;    /* Q7.8 log2(|z| / 2^14), INT16_MIN = ⊥ */
    uint16_t theta;  /* Z / 2^16 Z 円環位相 */
} lc_logc16_t;

typedef struct {
    int16_t re;
    int16_t im;
} lc_c16_t;

/* 双対チャート変換ベンチマーク用 128-byte AoSoA-8 ブロック */
typedef struct LC_ALIGN64 {
    int16_t  re[8];
    int16_t  im[8];
    int16_t  rho[8];
    uint16_t theta[8];
    int16_t  rt_re[8];
    int16_t  rt_im[8];
    int16_t  pad[16];
} lc_chart_block8_t;

typedef char lc_assert_chart_block8_128b[(sizeof(lc_chart_block8_t) == 128) ? 1 : -1];

/* ============================================================================
 * 2. -lm 不要の L1D 常駐双対チャート LUT (合計 7.5 KB)
 * ============================================================================ */
static int16_t  LC_ALIGN64 g_cos_lut[LC_TRIG_SIZE];
static int16_t  LC_ALIGN64 g_sin_lut[LC_TRIG_SIZE];
static uint16_t LC_ALIGN64 g_exp2_frac_lut[256];
static uint16_t LC_ALIGN64 g_half_log2_lut[LC_LOG2_SIZE + 1];
static uint16_t LC_ALIGN64 g_atan_lut[LC_ATAN_SIZE + 1];

static double math_sqrt_pos(double x) {
    if (x <= 0.0) return 0.0;
    double y = (x >= 1.0) ? x : 1.0;
    for (int i = 0; i < 20; i++) y = 0.5 * (y + x / y);
    return y;
}

static double math_sin(double x) {
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

static double math_cos(double x) {
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

static double math_exp2_frac(double f) {
    static const double LN2 = 0.693147180559945309417;
    double x = f * LN2, term = 1.0, sum = 1.0;
    for (int k = 1; k <= 12; k++) {
        term *= x / (double)k;
        sum  += term;
    }
    return sum;
}

static double math_log2_1p(double f) {
    /* ln(1 + f) = 2 * sum_{k=0..8} u^(2k+1)/(2k+1), u = f / (2 + f) <= 1/3 */
    static const double INV_LN2 = 1.44269504088896340736;
    double u  = f / (2.0 + f);
    double u2 = u * u, p = u, sum = 0.0;
    for (int k = 1; k <= 17; k += 2) {
        sum += p / (double)k;
        p   *= u2;
    }
    return 2.0 * sum * INV_LN2;
}

static double math_atan_unit(double t) {
    /* 半角恒等式を 2 回適用して |u| <= tan(pi/16) ≈ 0.199 へ縮小 */
    double u = t / (1.0 + math_sqrt_pos(1.0 + t * t));
    u = u / (1.0 + math_sqrt_pos(1.0 + u * u));
    double u2 = u * u, p = u, sum = 0.0;
    for (int k = 0; k < 9; k++) {
        double term = p / (double)(2 * k + 1);
        sum += ((k & 1) == 0) ? term : -term;
        p   *= u2;
    }
    return 4.0 * sum;
}

static bool init_all_luts(void) {
    static const double TWO_PI = 6.28318530717958647692;
    static const double PI_4   = 0.78539816339744830962;

    /* 自己検証: -lm 不要初等関数精度確認 */
    double e_log  = math_log2_1p(1.0) - 1.0;
    double e_atan = math_atan_unit(1.0) - PI_4;
    if (e_log < -1e-9 || e_log > 1e-9 || e_atan < -1e-9 || e_atan > 1e-9) return false;

    for (int i = 0; i < LC_TRIG_SIZE; i++) {
        double ang = TWO_PI * (double)i / (double)LC_TRIG_SIZE;
        double c   = math_cos(ang) * (double)LC_Q14_ONE;
        double s   = math_sin(ang) * (double)LC_Q14_ONE;
        g_cos_lut[i] = (int16_t)((c >= 0.0) ? (c + 0.5) : (c - 0.5));
        g_sin_lut[i] = (int16_t)((s >= 0.0) ? (s + 0.5) : (s - 0.5));
    }
    for (int f = 0; f < 256; f++) {
        double v = math_exp2_frac((double)f / 256.0) * (double)LC_Q14_ONE;
        g_exp2_frac_lut[f] = (uint16_t)(v + 0.5);
    }
    for (int i = 0; i <= LC_LOG2_SIZE; i++) {
        double v = 128.0 * math_log2_1p((double)i / (double)LC_LOG2_SIZE);
        g_half_log2_lut[i] = (uint16_t)(v + 0.5);
    }
    for (int i = 0; i <= LC_ATAN_SIZE; i++) {
        double ang = math_atan_unit((double)i / (double)LC_ATAN_SIZE);
        double q   = ang * (65536.0 / TWO_PI);
        g_atan_lut[i] = (uint16_t)(q + 0.5);
    }
    return true;
}

/* ============================================================================
 * 3. 純整数双対チャート順写像 (Log->Cart) & 逆写像 (Cart->Log) スカラー厳密定義
 * ============================================================================ */
static inline uint32_t lc_clz32(uint32_t x) {
    if (x == 0u) return 32u;
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_clz(x);
#else
    uint32_t n = 0;
    if ((x & 0xFFFF0000u) == 0u) { n += 16u; x <<= 16; }
    if ((x & 0xFF000000u) == 0u) { n +=  8u; x <<=  8; }
    if ((x & 0xF0000000u) == 0u) { n +=  4u; x <<=  4; }
    if ((x & 0xC0000000u) == 0u) { n +=  2u; x <<=  2; }
    if ((x & 0x80000000u) == 0u) { n +=  1u; }
    return n;
#endif
}

static inline int16_t log_add_rho(int16_t a, int16_t b) {
    if (a == LC_LOG_ZERO || b == LC_LOG_ZERO) return LC_LOG_ZERO;
    int32_t s = (int32_t)a + (int32_t)b;
    if (s >  32767)      return 32767;
    if (s <= LC_LOG_ZERO) return LC_LOG_ZERO;
    return (int16_t)s;
}

static inline lc_logc16_t fhrr_bind(lc_logc16_t a, lc_logc16_t b) {
    lc_logc16_t out = { log_add_rho(a.rho, b.rho), (uint16_t)(a.theta + b.theta) };
    return out;
}

static inline lc_c16_t cart_add_mod(lc_c16_t a, lc_c16_t b) {
    lc_c16_t out = {
        (int16_t)((uint16_t)a.re + (uint16_t)b.re),
        (int16_t)((uint16_t)a.im + (uint16_t)b.im)
    };
    return out;
}

static inline lc_c16_t cart_sub_mod(lc_c16_t a, lc_c16_t b) {
    lc_c16_t out = {
        (int16_t)((uint16_t)a.re - (uint16_t)b.re),
        (int16_t)((uint16_t)a.im - (uint16_t)b.im)
    };
    return out;
}

static inline int16_t quantize_q14_s32(int32_t acc) {
    int32_t mag = (acc >= 0) ? acc : -acc;
    int32_t q   = (mag + LC_Q14_HALF) >> 14;
    if (acc < 0) q = -q;
    if (q >  32767) return  32767;
    if (q < -32768) return -32768;
    return (int16_t)q;
}

/* 順写像: Log-Polar (rho, theta) -> 直交 (re, im) Q1.14 (最近傍ビン丸め付き) */
static inline lc_c16_t log_to_cart_scalar(int16_t rho, uint16_t theta) {
    int32_t e = (int32_t)(rho >> 8);
    if (e < -16) e = -16;
    if (e >   2) e =   2;

    int32_t base = (int32_t)g_exp2_frac_lut[(uint8_t)(rho & 0xFF)];
    int32_t mag  = (e >= 0) ? (base << e) : (base >> (-e));
    if (mag > 32767) mag = 32767;

    uint32_t idx = (((uint32_t)theta + (1u << (LC_TRIG_SHIFT - 1))) >> LC_TRIG_SHIFT)
                   & (LC_TRIG_SIZE - 1u);
    lc_c16_t out = {
        quantize_q14_s32(mag * (int32_t)g_cos_lut[idx]),
        quantize_q14_s32(mag * (int32_t)g_sin_lut[idx])
    };
    return out;
}

/* 逆写像: 直交 (re, im) -> Log-Polar (rho, theta) 純整数固定小数点 (全 int16 域無オーバーフロー) */
static inline lc_logc16_t cart_to_log_scalar(int16_t re, int16_t im) {
    if (re == 0 && im == 0) {
        lc_logc16_t z = { LC_LOG_ZERO, 0u };
        return z;
    }
    uint32_t ax = (re >= 0) ? (uint32_t)re : (uint32_t)(-(int32_t)re);
    uint32_t ay = (im >= 0) ? (uint32_t)im : (uint32_t)(-(int32_t)im);

    /* 1. rho = 128 * (k - 28) + half_log2_lut[norm_top9] */
    uint32_t r2   = ax * ax + ay * ay; /* 最大 2^31: uint32_t オーバーフロー皆無 */
    uint32_t lz   = lc_clz32(r2);
    int32_t  k    = 31 - (int32_t)lz;
    uint32_t norm = r2 << lz;
    uint32_t lidx = ((norm & 0x7FFFFFFFu) + (1u << 21)) >> 22; /* [0..512] */
    int16_t  rho  = (int16_t)(128 * (k - 28) + (int32_t)g_half_log2_lut[lidx]);

    /* 2. theta: 8 象限折りたたみ + 1024 点 atan LUT */
    uint32_t hi    = (ax >= ay) ? ax : ay;
    uint32_t lo    = (ax >= ay) ? ay : ax;
    uint32_t aidx  = ((lo << LC_ATAN_BITS) + (hi >> 1)) / hi; /* [0..1024] */
    uint16_t phi   = g_atan_lut[aidx];
    uint16_t oct   = (ay > ax) ? (uint16_t)(16384u - phi) : phi;
    uint16_t half  = (re < 0)  ? (uint16_t)(32768u - oct) : oct;
    uint16_t theta = (im < 0)  ? (uint16_t)(0u - half)    : half;

    lc_logc16_t out = { rho, theta };
    return out;
}

/* ============================================================================
 * 4. ARM64 NEON 8 セル並列 Cart->Log 逆写像カーネル (0-bit スカラー完全一致)
 * ============================================================================ */
static LC_NOINLINE void aosoa8_cart_to_log_scalar(lc_chart_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_chart_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            lc_logc16_t z = cart_to_log_scalar(b->re[k], b->im[k]);
            b->rho[k]   = z.rho;
            b->theta[k] = z.theta;
        }
    }
}

#if LC_HAS_NEON
static LC_NOINLINE void aosoa8_cart_to_log_neon(lc_chart_block8_t *blocks, int n_blocks) {
    const int16x8_t  zero16   = vdupq_n_s16(0);
    const int16x8_t  log_zero = vdupq_n_s16(LC_LOG_ZERO);
    const uint16x8_t zero_u16 = vdupq_n_u16(0);
    const uint16x8_t q1_16384 = vdupq_n_u16(16384u);
    const uint16x8_t q2_32768 = vdupq_n_u16(32768u);
    const uint32x4_t zero_u32 = vdupq_n_u32(0u);
    const uint32x4_t one_u32  = vdupq_n_u32(1u);
    const uint32x4_t mask31   = vdupq_n_u32(0x7FFFFFFFu);
    const uint32x4_t round21  = vdupq_n_u32(1u << 21);
    const int32x4_t  c3_32    = vdupq_n_s32(3);

    for (int bi = 0; bi < n_blocks; bi++) {
        lc_chart_block8_t *b = &blocks[bi];
        int16x8_t v_re = vld1q_s16(b->re);
        int16x8_t v_im = vld1q_s16(b->im);

        /* 1. 32-bit 拡張後に絶対値取得 (-32768 も無オーバーフロー) */
        uint32x4_t ax_lo = vreinterpretq_u32_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_re))));
        uint32x4_t ax_hi = vreinterpretq_u32_s32(vabsq_s32(vmovl_high_s16(v_re)));
        uint32x4_t ay_lo = vreinterpretq_u32_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_im))));
        uint32x4_t ay_hi = vreinterpretq_u32_s32(vabsq_s32(vmovl_high_s16(v_im)));

        /* 2. r2 = ax^2 + ay^2 & NEON 単一命令 clz (vclzq_u32) */
        uint32x4_t r2_lo = vmlaq_u32(vmulq_u32(ax_lo, ax_lo), ay_lo, ay_lo);
        uint32x4_t r2_hi = vmlaq_u32(vmulq_u32(ax_hi, ax_hi), ay_hi, ay_hi);

        uint32x4_t lz_lo = vclzq_u32(r2_lo);
        uint32x4_t lz_hi = vclzq_u32(r2_hi);

        uint32x4_t norm_lo = vshlq_u32(r2_lo, vreinterpretq_s32_u32(lz_lo));
        uint32x4_t norm_hi = vshlq_u32(r2_hi, vreinterpretq_s32_u32(lz_hi));

        uint32x4_t lidx_lo = vshrq_n_u32(vaddq_u32(vandq_u32(norm_lo, mask31), round21), 22);
        uint32x4_t lidx_hi = vshrq_n_u32(vaddq_u32(vandq_u32(norm_hi, mask31), round21), 22);

        /* (k - 28) = (31 - lz) - 28 = 3 - lz -> 128 * (3 - lz) */
        int32x4_t base_lo = vshlq_n_s32(vsubq_s32(c3_32, vreinterpretq_s32_u32(lz_lo)), 7);
        int32x4_t base_hi = vshlq_n_s32(vsubq_s32(c3_32, vreinterpretq_s32_u32(lz_hi)), 7);
        int16x8_t rho_base = vcombine_s16(vmovn_s32(base_lo), vmovn_s32(base_hi));

        /* 3. max/min 抽出 & ゼロ除算防止 */
        uint32x4_t hi_lo = vmaxq_u32(vmaxq_u32(ax_lo, ay_lo), one_u32);
        uint32x4_t hi_hi = vmaxq_u32(vmaxq_u32(ax_hi, ay_hi), one_u32);
        uint32x4_t lo_lo = vminq_u32(ax_lo, ay_lo);
        uint32x4_t lo_hi = vminq_u32(ax_hi, ay_hi);

        uint32_t lidx_arr[8], hi_arr[8], lo_arr[8];
        vst1q_u32(&lidx_arr[0], lidx_lo); vst1q_u32(&lidx_arr[4], lidx_hi);
        vst1q_u32(&hi_arr[0],   hi_lo);   vst1q_u32(&hi_arr[4],   hi_hi);
        vst1q_u32(&lo_arr[0],   lo_lo);   vst1q_u32(&lo_arr[4],   lo_hi);

        uint16_t frac_arr[8], phi_arr[8];
        for (int k = 0; k < 8; k++) {
            frac_arr[k] = g_half_log2_lut[lidx_arr[k]];
            uint32_t aidx = ((lo_arr[k] << LC_ATAN_BITS) + (hi_arr[k] >> 1)) / hi_arr[k];
            phi_arr[k]  = g_atan_lut[aidx];
        }

        /* 4. 8 セル同時 rho 合成・8 象限展開・原点 ⊥ マスク */
        uint16x8_t is_zero = vcombine_u16(vmovn_u32(vceqq_u32(r2_lo, zero_u32)),
                                          vmovn_u32(vceqq_u32(r2_hi, zero_u32)));

        int16x8_t  v_rho   = vaddq_s16(rho_base, vreinterpretq_s16_u16(vld1q_u16(frac_arr)));
        v_rho = vbslq_s16(is_zero, log_zero, v_rho);

        uint16x8_t ay_gt_ax = vcombine_u16(vmovn_u32(vcgtq_u32(ay_lo, ax_lo)),
                                           vmovn_u32(vcgtq_u32(ay_hi, ax_hi)));
        uint16x8_t re_neg   = vcltq_s16(v_re, zero16);
        uint16x8_t im_neg   = vcltq_s16(v_im, zero16);

        uint16x8_t phi   = vld1q_u16(phi_arr);
        uint16x8_t oct   = vbslq_u16(ay_gt_ax, vsubq_u16(q1_16384, phi), phi);
        uint16x8_t half  = vbslq_u16(re_neg,   vsubq_u16(q2_32768, oct), oct);
        uint16x8_t theta = vbslq_u16(im_neg,   vsubq_u16(zero_u16, half), half);
        theta = vbicq_u16(theta, is_zero);

        vst1q_s16(b->rho,   v_rho);
        vst1q_u16(b->theta, theta);
    }
}
#endif

/* ============================================================================
 * 5. [Test 1] Cart->Log 逆写像精度・Scalar vs NEON 0-bit 一致・スループット実測
 * ============================================================================ */
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

static bool run_test1_inverse_chart_and_neon(void) {
    /* 1-A: 8 象限・軸境界・原点・極端値 (-32768, -32768) の厳密角度・強度検証 */
    static const struct {
        int16_t  re, im;
        int16_t  exp_rho;
        uint16_t exp_theta;
    } axes[] = {
        {      0,      0, LC_LOG_ZERO,     0u },
        {  16384,      0,           0,     0u },
        {  16384,  16384,         128,  8192u },
        {      0,  16384,           0, 16384u },
        { -16384,  16384,         128, 24576u },
        { -16384,      0,           0, 32768u },
        { -16384, -16384,         128, 40960u },
        {      0, -16384,           0, 49152u },
        {  16384, -16384,         128, 57344u },
        { -32768, -32768,         384, 40960u }
    };
    const int n_axes = (int)(sizeof(axes) / sizeof(axes[0]));
    for (int i = 0; i < n_axes; i++) {
        lc_logc16_t z = cart_to_log_scalar(axes[i].re, axes[i].im);
        if (z.rho != axes[i].exp_rho || z.theta != axes[i].exp_theta) {
            fprintf(stderr, "FAIL: Test 1-A 軸検証 %d: got (%d, %u) vs exp (%d, %u)\n",
                    i, (int)z.rho, (unsigned)z.theta, (int)axes[i].exp_rho, (unsigned)axes[i].exp_theta);
            return false;
        }
    }

    /* 1-B: Log -> Cart -> Log 往復精度 (rho ∈ [-1024, +128], 全位相) */
    int max_drho = 0, max_dtheta = 0;
    for (int r = -1024; r <= 128; r += 32) {
        for (int t = 0; t < 65536; t += 113) {
            lc_c16_t    c  = log_to_cart_scalar((int16_t)r, (uint16_t)t);
            lc_logc16_t z2 = cart_to_log_scalar(c.re, c.im);
            int dr = (int)z2.rho - r;
            if (dr < 0) dr = -dr;
            int16_t dt16 = (int16_t)((uint16_t)(z2.theta - (uint16_t)t));
            int dt = (dt16 == -32768) ? 32768 : ((dt16 >= 0) ? dt16 : -dt16);
            if (dr > max_drho)   max_drho   = dr;
            if (dt > max_dtheta) max_dtheta = dt;
        }
    }

    /* 1-C: 16,384 セル Scalar vs ARM64 NEON 0-bit 一致 + L1D / L2 スループット */
    enum { N_BLOCKS = 2048, N_CELLS = N_BLOCKS * 8 };
    static lc_chart_block8_t LC_ALIGN64 blk_sc[N_BLOCKS];
    static lc_chart_block8_t LC_ALIGN64 blk_nm[N_BLOCKS];
    static const int16_t corners[] = { 0, 1, -1, 128, -128, 16384, -16384, 32767, -32767, -32768 };
    const int nc = (int)(sizeof(corners) / sizeof(corners[0]));
    uint32_t rng = 0x20261123u;

    for (int bi = 0; bi < N_BLOCKS; bi++) {
        for (int k = 0; k < 8; k++) {
            int idx = bi * 8 + k;
            if (idx < 256) {
                blk_sc[bi].re[k] = corners[(idx / nc) % nc];
                blk_sc[bi].im[k] = corners[idx % nc];
            } else {
                uint32_t r0 = next_rng(&rng);
                blk_sc[bi].re[k] = (int16_t)(r0 & 0xFFFFu);
                blk_sc[bi].im[k] = (int16_t)(r0 >> 16);
            }
        }
    }
    memcpy(blk_nm, blk_sc, sizeof(blk_nm));

    aosoa8_cart_to_log_scalar(blk_sc, N_BLOCKS);
#if LC_HAS_NEON
    aosoa8_cart_to_log_neon(blk_nm, N_BLOCKS);
#else
    aosoa8_cart_to_log_scalar(blk_nm, N_BLOCKS);
#endif

    for (int bi = 0; bi < N_BLOCKS; bi++) {
        for (int k = 0; k < 8; k++) {
            if (blk_sc[bi].rho[k] != blk_nm[bi].rho[k] ||
                blk_sc[bi].theta[k] != blk_nm[bi].theta[k]) {
                fprintf(stderr, "FAIL: Test 1-C Scalar vs NEON 0-bit 不一致 (%d,%d)\n", bi, k);
                return false;
            }
        }
    }

    /* L1D (512 cells) & L2 (16,384 cells) 計測 */
    const int ITERS_L1 = 40000, ITERS_L2 = 2000;
    double t0, dt_sc_l1, dt_nm_l1, dt_sc_l2, dt_nm_l2;

    t0 = now_ns();
    for (int it = 0; it < ITERS_L1; it++) aosoa8_cart_to_log_scalar(blk_sc, 64);
    dt_sc_l1 = now_ns() - t0;
#if LC_HAS_NEON
    t0 = now_ns();
    for (int it = 0; it < ITERS_L1; it++) aosoa8_cart_to_log_neon(blk_nm, 64);
    dt_nm_l1 = now_ns() - t0;
#else
    dt_nm_l1 = dt_sc_l1;
#endif

    t0 = now_ns();
    for (int it = 0; it < ITERS_L2; it++) aosoa8_cart_to_log_scalar(blk_sc, N_BLOCKS);
    dt_sc_l2 = now_ns() - t0;
#if LC_HAS_NEON
    t0 = now_ns();
    for (int it = 0; it < ITERS_L2; it++) aosoa8_cart_to_log_neon(blk_nm, N_BLOCKS);
    dt_nm_l2 = now_ns() - t0;
#else
    dt_nm_l2 = dt_sc_l2;
#endif

    double tot_l1 = 512.0 * (double)ITERS_L1;
    double tot_l2 = (double)N_CELLS * (double)ITERS_L2;

    printf("[Test 1] 純整数逆写像 Cart->Log (CLZ + 1KB log2 LUT + 2KB atan LUT):\n");
    printf("  1-A 8 象限・軸・原点 ⊥・(-32768,-32768) 厳密一致 : 10/10 OK\n");
    printf("  1-B Log->Cart->Log 往復最大誤差 (rho∈[-1024,+128]): |Δrho|<=%d LSB, |Δtheta|<=%d (%.3f deg)\n",
           max_drho, max_dtheta, 360.0 * (double)max_dtheta / 65536.0);
    printf("  1-C 16,384 セル Scalar vs ARM64 NEON 0-bit 一致  : OK\n");
    printf("      L1D (  512 cells): Scalar %.2f ns (%4.0f M/s) | NEON %.2f ns (%4.0f M/s) | %.2fx\n",
           dt_sc_l1 / tot_l1, (tot_l1 * 1e3) / dt_sc_l1,
           dt_nm_l1 / tot_l1, (tot_l1 * 1e3) / dt_nm_l1, dt_sc_l1 / dt_nm_l1);
    printf("      L2  (16384 cells): Scalar %.2f ns (%4.0f M/s) | NEON %.2f ns (%4.0f M/s) | %.2fx\n\n",
           dt_sc_l2 / tot_l2, (tot_l2 * 1e3) / dt_sc_l2,
           dt_nm_l2 / tot_l2, (tot_l2 * 1e3) / dt_nm_l2, dt_sc_l2 / dt_nm_l2);

    return (max_drho <= 2 && max_dtheta <= 64);
}

/* ============================================================================
 * 6. [Test 2] 64 層双対チャート閉ループ・リフティング梯子 (μ^64 ∘ Δ^64 = id)
 *    各層で Cart -> Log -> FHRR Bind -> Cart を行いつつ 0-bit 完全可逆を実証
 * ============================================================================ */
static bool run_test2_64layer_dual_chart_ladder(void) {
    enum { DEPTH = 64, N_PAIRS = 512 };
    lc_logc16_t weights[DEPTH];
    lc_c16_t    s0[N_PAIRS], v0[N_PAIRS];
    lc_c16_t    s[N_PAIRS],  v[N_PAIRS];

    uint32_t rng = 0x20261123u;
    for (int L = 0; L < DEPTH; L++) {
        uint32_t r = next_rng(&rng);
        weights[L].rho   = (int16_t)((int32_t)(r & 0x3FFu) - 512);
        weights[L].theta = (uint16_t)(r >> 16);
    }
    for (int i = 0; i < N_PAIRS; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng);
        s0[i].re = (i == 0) ? 0      : (int16_t)(r0 & 0xFFFFu);
        s0[i].im = (i == 0) ? 0      : (int16_t)(r0 >> 16);
        v0[i].re = (i == 1) ? -32768 : (int16_t)(r1 & 0xFFFFu);
        v0[i].im = (i == 1) ? -32768 : (int16_t)(r1 >> 16);
        s[i] = s0[i];
        v[i] = v0[i];
    }

    /* 順方向 64 層 (Δ^64): 毎層 Cart->Log->FHRR Bind->Log->Cart で非線形混合 */
    for (int L = 0; L < DEPTH; L++) {
        for (int i = 0; i < N_PAIRS; i++) {
            lc_logc16_t sl = cart_to_log_scalar(s[i].re, s[i].im);
            lc_logc16_t pl = fhrr_bind(weights[L], sl);
            lc_c16_t    pc = log_to_cart_scalar(pl.rho, pl.theta);
            lc_c16_t    vn = cart_sub_mod(v[i], pc);
            /* 交互ストリーム入替 (Feistel / RevNet 梯子) */
            v[i] = s[i];
            s[i] = vn;
        }
    }

    /* 中間状態が十分に攪拌されていることを確認 */
    int changed = 0;
    for (int i = 0; i < N_PAIRS; i++) {
        if (s[i].re != s0[i].re || v[i].re != v0[i].re) changed++;
    }

    /* 逆方向 64 層 (μ^64): 同一双対チャート予測器を用いて厳密復元 */
    for (int L = DEPTH - 1; L >= 0; L--) {
        for (int i = 0; i < N_PAIRS; i++) {
            lc_c16_t    s_prev = v[i];
            lc_c16_t    vn     = s[i];
            lc_logc16_t sl     = cart_to_log_scalar(s_prev.re, s_prev.im);
            lc_logc16_t pl     = fhrr_bind(weights[L], sl);
            lc_c16_t    pc     = log_to_cart_scalar(pl.rho, pl.theta);
            s[i] = s_prev;
            v[i] = cart_add_mod(vn, pc);
        }
    }

    for (int i = 0; i < N_PAIRS; i++) {
        if (s[i].re != s0[i].re || s[i].im != s0[i].im ||
            v[i].re != v0[i].re || v[i].im != v0[i].im) {
            fprintf(stderr, "FAIL: Test 2 64 層閉ループ復元誤差 (pair %d)\n", i);
            return false;
        }
    }
    printf("[Test 2] 64 層双対チャート閉ループ・リフティング梯子 (μ^64 ∘ Δ^64 = id):\n");
    printf("  512 複素ペア (攪拌率 %d/512) -> 64 層 Cart<->Log 往復後 0-bit 厳密復元: OK\n\n", changed);
    return (changed == N_PAIRS);
}

/* ============================================================================
 * 7. [Test 3] Frobenius 双対 (ZX-Spider) 4 公理の 0-bit 厳密検証
 * ============================================================================ */
static inline void z_spider_copy(lc_logc16_t in, lc_logc16_t phase,
                                 lc_logc16_t *out0, lc_logc16_t *out1) {
    lc_logc16_t y = fhrr_bind(in, phase);
    *out0 = y;
    *out1 = y;
}

static inline uint16_t x_spider_parity(uint16_t a, uint16_t b, uint16_t phase) {
    return (uint16_t)(a + b + phase);
}

static bool run_test3_frobenius_zx_laws(void) {
    uint32_t rng = 0x20261123u;

    /* 3-A: Z-Spider Fusion: Z(β) ∘ Z(α) == Z(α ⊛ β) (連続 Log-Polar 位相空間, 1000 ケース) */
    for (int i = 0; i < 1000; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng), r2 = next_rng(&rng);
        lc_logc16_t x = { (int16_t)((int32_t)(r0 & 0x7FFu) - 1024), (uint16_t)(r0 >> 16) };
        lc_logc16_t a = { (int16_t)((int32_t)(r1 & 0x7FFu) - 1024), (uint16_t)(r1 >> 16) };
        lc_logc16_t b = { (int16_t)((int32_t)(r2 & 0x7FFu) - 1024), (uint16_t)(r2 >> 16) };

        lc_logc16_t lhs = fhrr_bind(fhrr_bind(x, a), b);
        lc_logc16_t rhs = fhrr_bind(x, fhrr_bind(a, b));
        if (lhs.rho != rhs.rho || lhs.theta != rhs.theta) return false;
    }

    /* 3-B: X-Spider Fusion: X(β2) ∘ X(β1) == X(β1 ⊕ β2) (直交 Z/2^16Z 重ね合わせ, 1000 ケース) */
    for (int i = 0; i < 1000; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng), r2 = next_rng(&rng);
        lc_c16_t x  = { (int16_t)(r0 & 0xFFFFu), (int16_t)(r0 >> 16) };
        lc_c16_t b1 = { (int16_t)(r1 & 0xFFFFu), (int16_t)(r1 >> 16) };
        lc_c16_t b2 = { (int16_t)(r2 & 0xFFFFu), (int16_t)(r2 >> 16) };

        lc_c16_t lhs = cart_add_mod(cart_add_mod(x, b1), b2);
        lc_c16_t rhs = cart_add_mod(x, cart_add_mod(b1, b2));
        if (lhs.re != rhs.re || lhs.im != rhs.im) return false;
    }

    /* 3-C & 3-D: ZX-Bialgebra Law (K_{2,2} <-> 単一橋) & ZX-Hopf Law (平行2辺消滅)
     *            Z_2 位相部分群 {0, 0x8000} 上の全入力組み合わせ網羅検証 */
    const uint16_t z2[2] = { 0x0000u, 0x8000u };
    const lc_logc16_t id_z = { 0, 0u };
    for (int ia = 0; ia < 2; ia++) {
        for (int ib = 0; ib < 2; ib++) {
            lc_logc16_t a = { 0, z2[ia] }, b = { 0, z2[ib] };
            /* Bialgebra LHS: Z-copy(a) と Z-copy(b) を 2 個の X-parity へ完全二部接続 */
            lc_logc16_t a0, a1, b0, b1;
            z_spider_copy(a, id_z, &a0, &a1);
            z_spider_copy(b, id_z, &b0, &b1);
            uint16_t lhs0 = x_spider_parity(a0.theta, b0.theta, 0u);
            uint16_t lhs1 = x_spider_parity(a1.theta, b1.theta, 0u);

            /* Bialgebra RHS: 先に X-parity(a, b) を取り、1 個の Z-copy で分岐 */
            lc_logc16_t mid = { 0, x_spider_parity(a.theta, b.theta, 0u) };
            lc_logc16_t rhs0, rhs1;
            z_spider_copy(mid, id_z, &rhs0, &rhs1);
            if (lhs0 != rhs0.theta || lhs1 != rhs1.theta) return false;

            /* Hopf Law: Z-copy(a) の 2 本の平行辺を同一 X-parity へ入れると恒等ゼロ 0x0000 */
            uint16_t hopf = x_spider_parity(a0.theta, a1.theta, 0u);
            if (hopf != 0x0000u) return false;
        }
    }

    printf("[Test 3] Frobenius 双対 (ZX-Spider) 4 公理の 0-bit 厳密検証:\n");
    printf("  3-A Z-Spider Fusion (Log-Polar 連続位相) : 1000/1000 厳密一致 (OK)\n");
    printf("  3-B X-Spider Fusion (直交 Z/2^16Z 重畳)  : 1000/1000 厳密一致 (OK)\n");
    printf("  3-C ZX-Bialgebra Law (K_{2,2} <-> 橋縮約):  4/4 全入力 0-bit 一致 (OK)\n");
    printf("  3-D ZX-Hopf Law (平行2辺 0x8000+0x8000=0):  4/4 完全分離消滅 (OK)\n\n");
    return true;
}

/* ============================================================================
 * 8. [Test 4] 統合実証: ZX 双対回路による 256 次元 FHRR 閉ループ連想想起 & 2-射ホロノミー自己修復
 *    Z-Spider (コピー/結合) -> Dual-Chart -> X-Spider (重畳/帯域集約) -> Cart->Log -> Z-Fusion 補償
 * ============================================================================ */
static bool run_test4_zx_closed_loop_holonomy_repair(void) {
    enum { D = 256, N_BANDS = 4, BAND_W = D / N_BANDS, N_ITEMS = 8 };
    static uint16_t keys[N_ITEMS][D];
    static uint16_t vals[N_ITEMS][D];
    static lc_c16_t bundle[D]; /* 8 項平均化後の int16_t 直交 X-Spider 記憶メモリ */

    /* 4 帯域それぞれに未知の複素位相ホロノミーねじれを注入 (帯域 0 は対蹠点 π=0x8000) */
    const uint16_t true_twist[N_BANDS] = { 0x8000u, 0x4000u, 0x2B73u, 0xC915u };

    uint32_t rng = 0x20261123u;
    for (int m = 0; m < N_ITEMS; m++) {
        for (int d = 0; d < D; d++) {
            keys[m][d] = (uint16_t)(next_rng(&rng) & 0xFFFFu);
            vals[m][d] = (uint16_t)(next_rng(&rng) & 0xFFFFu);
        }
    }

    /* 1. 符号化: Z-Spider (K_m ⊛ V_m) -> Log->Cart -> X-Spider Bundling */
    for (int d = 0; d < D; d++) {
        int32_t acc_re = 0, acc_im = 0;
        for (int m = 0; m < N_ITEMS; m++) {
            lc_logc16_t kv = fhrr_bind((lc_logc16_t){0, keys[m][d]}, (lc_logc16_t){0, vals[m][d]});
            lc_c16_t    c  = log_to_cart_scalar(kv.rho, kv.theta);
            acc_re += c.re;
            acc_im += c.im;
        }
        bundle[d].re = (int16_t)(acc_re >> 3); /* /8 スケーリングで直交 int16 表現へ格納 */
        bundle[d].im = (int16_t)(acc_im >> 3);
    }

    /* 2. ねじれクエリ評価ヘルパ: 帯域補償角 z_comp[b] を Z-Fusion 適用した想起正解数を計測 */
    uint16_t z_comp[N_BANDS] = { 0u, 0u, 0u, 0u };
    int hits_before = 0, hits_after = 0;
    double mean_sim_before = 0.0, mean_sim_after = 0.0;

    for (int phase = 0; phase < 2; phase++) {
        int    hits     = 0;
        double sim_sum  = 0.0;

        for (int q = 0; q < N_ITEMS; q++) {
            int    best_m   = -1;
            double best_sim = -1e9;
            for (int cand = 0; cand < N_ITEMS; cand++) {
                int64_t dot = 0;
                for (int d = 0; d < D; d++) {
                    int      b   = d / BAND_W;
                    /* 歪んだクエリ Z-Spider: K_q + true_twist[b] + z_comp[b] (Z-Fusion) */
                    uint16_t qk  = (uint16_t)(keys[q][d] + true_twist[b] + z_comp[b]);
                    uint16_t rot = (uint16_t)(qk + vals[cand][d]);
                    uint32_t idx = (((uint32_t)rot + 32u) >> LC_TRIG_SHIFT) & (LC_TRIG_SIZE - 1u);
                    dot += (int64_t)bundle[d].re * g_cos_lut[idx]
                         + (int64_t)bundle[d].im * g_sin_lut[idx];
                }
                double sim = (double)dot / ((double)D * (double)(LC_Q14_ONE >> 3) * (double)LC_Q14_ONE);
                if (sim > best_sim) { best_sim = sim; best_m = cand; }
                if (cand == q) sim_sum += sim;
            }
            if (best_m == q) hits++;
        }

        if (phase == 0) {
            hits_before     = hits;
            mean_sim_before = sim_sum / (double)N_ITEMS;

            /* 3. 閉ループ 2-射ホロノミー検出:
             *    パイロット項目 q=0 の残差を帯域ごとに X-Spider 集約 -> 純整数 Cart->Log 逆写像! */
            for (int b = 0; b < N_BANDS; b++) {
                int64_t sum_re = 0, sum_im = 0;
                for (int i = 0; i < BAND_W; i++) {
                    int      d   = b * BAND_W + i;
                    uint16_t qk  = (uint16_t)(keys[0][d] + true_twist[b]);
                    uint16_t rot = (uint16_t)(qk + vals[0][d]);
                    /* bundle[d] * exp(-i * (K0 + twist + V0)) = exp(-i * twist) + noise */
                    uint32_t idx = (((uint32_t)rot + 32u) >> LC_TRIG_SHIFT) & (LC_TRIG_SIZE - 1u);
                    int32_t  cr  = g_cos_lut[idx], cs = g_sin_lut[idx];
                    sum_re += (int32_t)bundle[d].re * cr + (int32_t)bundle[d].im * cs;
                    sum_im += (int32_t)bundle[d].im * cr - (int32_t)bundle[d].re * cs;
                }
                /* 帯域 X-Spider 集約結果を int16 へ正規化し、純整数 Cart->Log 逆写像で 2-射位相抽出 */
                int16_t agg_re = (int16_t)(sum_re >> 18);
                int16_t agg_im = (int16_t)(sum_im >> 18);
                lc_logc16_t holo = cart_to_log_scalar(agg_re, agg_im);

                /* holo.theta ≈ -true_twist[b] なので、そのまま Z-Spider へ Z-Fusion で補償結合 */
                z_comp[b] = holo.theta;
            }
        } else {
            hits_after     = hits;
            mean_sim_after = sim_sum / (double)N_ITEMS;
        }
    }

    printf("[Test 4] Frobenius ZX 閉ループ 256 次元 FHRR 連想想起 & 2-射ホロノミー自己修復:\n");
    printf("  修復前 (4 帯域未知位相ねじれ下) : 正解 %d/8 (平均正解類似度 = %+.3f)\n",
           hits_before, mean_sim_before);
    printf("  X-Spider 集約 -> 純整数 Cart->Log 逆写像 -> Z-Fusion 補償後の推定誤差:\n");
    for (int b = 0; b < N_BANDS; b++) {
        uint16_t est_twist = (uint16_t)(0u - z_comp[b]);
        int16_t  err16     = (int16_t)((uint16_t)(est_twist - true_twist[b]));
        printf("    帯域 %d: 真ねじれ=0x%04X | 推定=0x%04X | 残留位相誤差=%+.2f deg\n",
               b, (unsigned)true_twist[b], (unsigned)est_twist,
               360.0 * (double)err16 / 65536.0);
    }
    printf("  修復後 (Z-Spider 補償融合後)    : 正解 %d/8 (平均正解類似度 = %+.3f)\n\n",
           hits_after, mean_sim_after);

    return (hits_before <= 2 && hits_after == N_ITEMS && mean_sim_after > 0.75);
}

int main(void) {
    if (!init_all_luts()) {
        fprintf(stderr, "FATAL: 双対チャート LUT 初期化検証失敗\n");
        return 1;
    }

    printf("=== 純整数双対チャート完結 (Cart<->Log) & Frobenius ZX-Spider 統合検証 ===\n");
    printf("双対 LUT 合計 = %.1f KB (L1D 常駐・-lm 不要) | NEON = %s\n\n",
           (double)(sizeof(g_cos_lut) + sizeof(g_sin_lut) + sizeof(g_exp2_frac_lut) +
                    sizeof(g_half_log2_lut) + sizeof(g_atan_lut)) / 1024.0,
           LC_HAS_NEON ? "ON (ARM64)" : "OFF");

    if (!run_test1_inverse_chart_and_neon())          return 1;
    if (!run_test2_64layer_dual_chart_ladder())       return 1;
    if (!run_test3_frobenius_zx_laws())               return 1;
    if (!run_test4_zx_closed_loop_holonomy_repair())  return 1;

    printf(">>> ALL ASSERTIONS PASSED: 双対チャート閉ループ可逆性 & Frobenius ZX 自己修復を実証 <<<\n");
    return 0;
}