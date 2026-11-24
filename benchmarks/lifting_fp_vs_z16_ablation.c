#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * 1. -lm 不要の倍精度初等関数（exp / ln / tanh / atanh）と IEEE 754 ビット丸め模擬
 *    FP32 (m=23), FP16 (binary16, m=10, 非正規化対応), BF16 (bfloat16, m=7)
 * ============================================================================ */
static double exp_fast(double x) {
    static const double LN2     = 0.693147180559945309417;
    static const double INV_LN2 = 1.44269504088896340736;
    if (x >  700.0) return 1e300;
    if (x < -700.0) return 0.0;

    int    k = (x >= 0.0) ? (int)(x * INV_LN2 + 0.5) : (int)(x * INV_LN2 - 0.5);
    double r = x - (double)k * LN2;

    double sum = 1.0, term = 1.0;
    for (int i = 1; i <= 12; i++) {
        term *= r / (double)i;
        sum  += term;
    }

    uint64_t pow2_bits = (uint64_t)(k + 1023) << 52;
    double   pow2;
    memcpy(&pow2, &pow2_bits, sizeof(pow2));
    return sum * pow2;
}

static double ln_fast(double z) {
    static const double LN2 = 0.693147180559945309417;
    uint64_t bits;
    memcpy(&bits, &z, sizeof(bits));

    int      e      = (int)((bits >> 52) & 0x7FFu) - 1023;
    uint64_t m_bits = (bits & 0x000FFFFFFFFFFFFFULL) | (1023ULL << 52);
    double   m;
    memcpy(&m, &m_bits, sizeof(m));

    if (m > 1.4142135623730951) {
        m *= 0.5;
        e += 1;
    }
    double t   = (m - 1.0) / (m + 1.0);
    double t2  = t * t;
    double p   = t;
    double sum = 0.0;
    for (int k = 1; k <= 15; k += 2) {
        sum += p / (double)k;
        p   *= t2;
    }
    return (double)e * LN2 + 2.0 * sum;
}

static inline double tanh_exact(double x) {
    double ax = (x >= 0.0) ? x : -x;
    if (ax > 18.0) return (x >= 0.0) ? 1.0 : -1.0;
    double em = exp_fast(-2.0 * ax);
    double th = (1.0 - em) / (1.0 + em);
    return (x >= 0.0) ? th : -th;
}

static inline double two_atanh_clamped(double mu, double max_llr) {
    double am = (mu >= 0.0) ? mu : -mu;
    if (am >= 1.0 - 1e-12) return (mu >= 0.0) ? max_llr : -max_llr;
    double v = ln_fast((1.0 + am) / (1.0 - am));
    if (v > max_llr) v = max_llr;
    return (mu >= 0.0) ? v : -v;
}

typedef enum {
    FP_FMT_FP32 = 0,
    FP_FMT_FP16 = 1,
    FP_FMT_BF16 = 2
} FPFormat;

static const char *fp_fmt_name(FPFormat fmt) {
    switch (fmt) {
        case FP_FMT_FP32: return "FP32 (23b)";
        case FP_FMT_FP16: return "FP16 (10b)";
        case FP_FMT_BF16: return "BF16 ( 7b)";
    }
    return "?";
}

/* IEEE 754 bfloat16 厳密丸め (round-to-nearest-even) */
static inline float quantize_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) return x; /* Inf / NaN */
    uint32_t lsb  = (u >> 16) & 1u;
    uint32_t bias = 0x7FFFu + lsb;
    u = (u + bias) & 0xFFFF0000u;
    float out;
    memcpy(&out, &u, sizeof(out));
    return out;
}

/* IEEE 754 binary16 厳密丸め (round-to-nearest-even, サブノーマル対応) */
static inline float quantize_fp16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    uint32_t sign = (u >> 31) & 1u;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t mant = u & 0x7FFFFFu;

    if (exp == 128) return x; /* Inf / NaN */
    if (exp > 15) {
        uint32_t inf = (sign << 31) | 0x7F800000u;
        float out;
        memcpy(&out, &inf, sizeof(out));
        return out;
    }
    if (exp >= -14) {
        uint32_t m10 = mant >> 13;
        uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1u) != 0u)) {
            m10++;
            if (m10 == 1024u) {
                m10 = 0u;
                exp++;
                if (exp > 15) {
                    uint32_t inf = (sign << 31) | 0x7F800000u;
                    float out;
                    memcpy(&out, &inf, sizeof(out));
                    return out;
                }
            }
        }
        uint32_t res = (sign << 31) | ((uint32_t)(exp + 127) << 23) | (m10 << 13);
        float out;
        memcpy(&out, &res, sizeof(out));
        return out;
    }
    if (exp >= -25) {
        uint32_t sig   = 0x800000u | mant;
        uint32_t shift = (uint32_t)(-14 - exp) + 13u; /* 14..24 */
        uint32_t q     = sig >> shift;
        uint32_t mask  = (1u << shift) - 1u;
        uint32_t half  = 1u << (shift - 1u);
        uint32_t rem   = sig & mask;
        if (rem > half || (rem == half && (q & 1u) != 0u)) q++;
        if (q == 0u) return sign ? -0.0f : 0.0f;
        float val = (float)q * (1.0f / 16777216.0f); /* 2^-24 は厳密表現 */
        return sign ? -val : val;
    }
    return sign ? -0.0f : 0.0f;
}

static inline float fp_q(FPFormat fmt, float x) {
    if (fmt == FP_FMT_BF16) return quantize_bf16(x);
    if (fmt == FP_FMT_FP16) return quantize_fp16(x);
    return x;
}

static inline float fp_add(FPFormat fmt, float a, float b) { return fp_q(fmt, a + b); }
static inline float fp_sub(FPFormat fmt, float a, float b) { return fp_q(fmt, a - b); }
static inline float fp_mul(FPFormat fmt, float a, float b) { return fp_q(fmt, a * b); }

static inline double dabs(double x) { return (x >= 0.0) ? x : -x; }

/* ============================================================================
 * 2. Z / 2^16 Z 厳密固定小数点 (Q1.14) ガウス整数基本演算 & LUT
 * ============================================================================ */
#define Q14_ONE   16384
#define Q14_HALF   8192
#define Q14_NOT  -32768
#define LLR_SCALE 256
#define LLR_MAX   2048   /* 8.0 in Q8 */

typedef struct {
    int16_t re;
    int16_t im;
} c16_t;

static inline int16_t quantize_q14(int64_t acc) {
    int64_t rounded = (acc >= 0) ? ((acc + Q14_HALF) >> 14)
                                 : -((-acc + Q14_HALF) >> 14);
    if (rounded > 32767)  return 32767;
    if (rounded < -32768) return -32768;
    return (int16_t)rounded;
}

static inline c16_t c16_mul_q14(c16_t a, c16_t b) {
    int64_t re_acc = (int64_t)a.re * b.re - (int64_t)a.im * b.im;
    int64_t im_acc = (int64_t)a.re * b.im + (int64_t)a.im * b.re;
    c16_t out = { quantize_q14(re_acc), quantize_q14(im_acc) };
    return out;
}

static inline c16_t c16_sub_mod(c16_t a, c16_t b) {
    c16_t out = {
        (int16_t)((uint16_t)a.re - (uint16_t)b.re),
        (int16_t)((uint16_t)a.im - (uint16_t)b.im)
    };
    return out;
}

static inline c16_t c16_add_mod(c16_t a, c16_t b) {
    c16_t out = {
        (int16_t)((uint16_t)a.re + (uint16_t)b.re),
        (int16_t)((uint16_t)a.im + (uint16_t)b.im)
    };
    return out;
}

static inline int32_t c16_l1_norm(c16_t a) {
    int32_t ar = (a.re >= 0) ? a.re : -(int32_t)a.re;
    int32_t ai = (a.im >= 0) ? a.im : -(int32_t)a.im;
    return ar + ai;
}

static inline int32_t sym_rshift64(int64_t val, int32_t shift) {
    if (val >= 0) return (int32_t)(val >> shift);
    return -(int32_t)((-val) >> shift);
}

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline uint8_t popcount8(uint8_t x) {
    uint8_t c = 0;
    while (x != 0) { c += (uint8_t)(x & 1u); x >>= 1; }
    return c;
}

static void update_complex_weight_mod(c16_t *w, c16_t err, c16_t s, uint8_t lr_shift) {
    if ((err.re == 0 && err.im == 0) || (s.re == 0 && s.im == 0)) return;
    int64_t grad_re = (int64_t)err.re * s.re + (int64_t)err.im * s.im;
    int64_t grad_im = (int64_t)err.im * s.re - (int64_t)err.re * s.im;
    int32_t shift   = 14 + (int32_t)lr_shift;
    int32_t step_re = sym_rshift64(grad_re, shift);
    int32_t step_im = sym_rshift64(grad_im, shift);
    if (step_re == 0 && grad_re != 0) step_re = (grad_re > 0) ? 1 : -1;
    if (step_im == 0 && grad_im != 0) step_im = (grad_im > 0) ? 1 : -1;
    w->re = (int16_t)((uint16_t)w->re + (uint16_t)step_re);
    w->im = (int16_t)((uint16_t)w->im + (uint16_t)step_im);
}

static int16_t g_lut_tanh_q14[LLR_MAX + 1];
static int16_t g_lut_atanh_q8[Q14_ONE + 1];

static void init_q14_luts(void) {
    for (int L = 0; L <= LLR_MAX; L++) {
        double th = tanh_exact((double)L / (double)(2 * LLR_SCALE));
        int32_t q = (int32_t)(th * (double)Q14_ONE + 0.5);
        if (q > Q14_ONE - 1) q = Q14_ONE - 1;
        g_lut_tanh_q14[L] = (int16_t)q;
    }
    g_lut_atanh_q8[0] = 0;
    for (int q = 1; q <= Q14_ONE; q++) {
        double mu = (double)q / (double)Q14_ONE;
        double ll = two_atanh_clamped(mu, (double)LLR_MAX / (double)LLR_SCALE);
        int32_t L = (int32_t)(ll * (double)LLR_SCALE + 0.5);
        if (L > LLR_MAX) L = LLR_MAX;
        g_lut_atanh_q8[q] = (int16_t)L;
    }
}

static inline int16_t llr_to_phase_q14(int32_t llr) {
    if (llr >= 0) return g_lut_tanh_q14[(llr > LLR_MAX) ? LLR_MAX : llr];
    int32_t m = -llr;
    return (int16_t)(-g_lut_tanh_q14[(m > LLR_MAX) ? LLR_MAX : m]);
}

static inline int32_t phase_q14_to_llr(int16_t mu) {
    if (mu >= 0) return (int32_t)g_lut_atanh_q8[(mu > Q14_ONE) ? Q14_ONE : mu];
    int32_t m = -(int32_t)mu;
    return -(int32_t)g_lut_atanh_q8[(m > Q14_ONE) ? Q14_ONE : m];
}

/* ============================================================================
 * 3. 決定論的乱数・正規分布生成器
 * ============================================================================ */
static inline uint32_t next_rng(uint32_t *st) {
    uint32_t x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

static double next_gauss_std(uint32_t *st) {
    double acc = -6.0;
    for (int i = 0; i < 12; i++) {
        acc += (double)(next_rng(st) >> 8) * (1.0 / 16777216.0);
    }
    return acc;
}

/* ============================================================================
 * 4. [Phase 1] 多層可逆リフティング往復 (64層 μ∘Δ) & 並列縮約の非結合性実験
 * ============================================================================ */
static void run_phase1_reversibility_and_associativity(void) {
    printf("[Phase 1-A] 64 層可逆リフティング往復 (μ∘Δ = id) の予測振幅スケール別散逸:\n");
    printf("  初期信号 v0 = 1.234567 (Q1.14 では整数 v0 = 20227) を 64 層減算->加算往復\n");
    printf("  %-11s | %-15s | %-16s | %-16s | %s\n",
           "予測振幅 P", "Z/2^16Z (Q1.14)", "FP32 (23b)", "FP16 (10b)", "BF16 ( 7b)");
    printf("  ------------+-----------------+------------------+------------------+------------------\n");

    const float scales[] = { 1.0f, 8.0f, 64.0f, 512.0f, 4096.0f };
    const int   n_scales = (int)(sizeof(scales) / sizeof(scales[0]));
    const int   DEPTH    = 64;

    uint32_t rng = 0x11232026u;
    float    unit_p[64];
    int16_t  q14_p[64];
    for (int k = 0; k < DEPTH; k++) {
        double u  = 0.5 + (double)(next_rng(&rng) & 0xFFFFu) / 65536.0; /* [0.5, 1.5) */
        unit_p[k] = (float)(((k & 1) == 0) ? u : -u);
        q14_p[k]  = (int16_t)(next_rng(&rng) & 0xFFFFu); /* 全 16-bit 振幅・ラップアラウンド込み */
    }

    const float   v0_f   = 1.234567f;
    const int16_t v0_q14 = 20227;

    for (int si = 0; si < n_scales; si++) {
        float P = scales[si];

        /* 1. Z/2^16Z 厳密固定小数点: 64 層 Δ -> 64 層 μ */
        int16_t cur_q = v0_q14;
        for (int k = 0; k < DEPTH; k++) {
            cur_q = (int16_t)((uint16_t)cur_q - (uint16_t)q14_p[k]);
        }
        for (int k = DEPTH - 1; k >= 0; k--) {
            cur_q = (int16_t)((uint16_t)cur_q + (uint16_t)q14_p[k]);
        }
        int32_t err_q = (int32_t)cur_q - (int32_t)v0_q14;
        if (err_q != 0) {
            fprintf(stderr, "FATAL: Z/2^16Z 可逆性不変条件違反\n");
            exit(1);
        }

        /* 2. IEEE 754 (FP32 / FP16 / BF16): 64 層 Δ -> 64 層 μ */
        double err_fp[3];
        for (int fi = 0; fi < 3; fi++) {
            FPFormat fmt = (FPFormat)fi;
            float v_init = fp_q(fmt, v0_f);
            float cur    = v_init;
            float p_arr[64];
            for (int k = 0; k < DEPTH; k++) {
                p_arr[k] = fp_mul(fmt, fp_q(fmt, P), fp_q(fmt, unit_p[k]));
                cur      = fp_sub(fmt, cur, p_arr[k]);
            }
            for (int k = DEPTH - 1; k >= 0; k--) {
                cur = fp_add(fmt, cur, p_arr[k]);
            }
            err_fp[fi] = dabs((double)cur - (double)v_init);
        }

        printf("  P = %6.0f  | 0.000000 (0bit) | %8.6f (%4.0f%%) | %8.6f (%4.0f%%) | %8.6f (%4.0f%%)\n",
               (double)P,
               err_fp[0], 100.0 * err_fp[0] / (double)v0_f,
               err_fp[1], 100.0 * err_fp[1] / (double)v0_f,
               err_fp[2], 100.0 * err_fp[2] / (double)v0_f);
    }
    printf("  (括弧内は初期信号 v0 に対する相対破壊率。100%% は信号完全消滅)\n\n");

    /* [Phase 1-B] 並列縮約順序 (順方向 vs 逆方向 vs 木縮約) による非結合性ドリフト */
    printf("[Phase 1-B] 1,024 要素並列縮約の非結合性ドリフト (外れ値チャネル 1 本 = 512.0 混入):\n");
    const int N_RED = 1024;
    static float   vec_f[1024];
    static int16_t vec_q[1024];

    vec_f[0] = 512.0f; /* Transformer における Massive Activation 外れ値チャネル模擬 */
    vec_q[0] = (int16_t)30000;
    double exact_sum = 512.0;
    for (int i = 1; i < N_RED; i++) {
        float x = (float)(0.05 + 0.05 * (double)(next_rng(&rng) & 0xFFu) / 256.0);
        vec_f[i]   = x;
        vec_q[i]   = (int16_t)(next_rng(&rng) & 0xFFFFu);
        exact_sum += (double)x;
    }

    /* Z/2^16Z: 順方向・逆方向・ペアワイズ木 */
    int16_t q_fwd = 0, q_rev = 0;
    static int16_t q_tree[1024];
    for (int i = 0; i < N_RED; i++) {
        q_fwd     = (int16_t)((uint16_t)q_fwd + (uint16_t)vec_q[i]);
        q_rev     = (int16_t)((uint16_t)q_rev + (uint16_t)vec_q[N_RED - 1 - i]);
        q_tree[i] = vec_q[i];
    }
    for (int stride = 1; stride < N_RED; stride <<= 1) {
        for (int i = 0; i < N_RED; i += stride * 2) {
            q_tree[i] = (int16_t)((uint16_t)q_tree[i] + (uint16_t)q_tree[i + stride]);
        }
    }
    if (q_fwd != q_rev || q_fwd != q_tree[0]) {
        fprintf(stderr, "FATAL: Z/2^16Z 結合律違反\n");
        exit(1);
    }

    printf("  %-13s | %-11s | %-11s | %-11s | %-13s | %s\n",
           "数値フォーマット", "順方向加算", "逆方向加算", "ペア木縮約", "順序間最大乖離", "対厳密値誤差(順)");
    printf("  --------------+-------------+-------------+-------------+---------------+----------------\n");
    printf("  %-13s | %11d | %11d | %11d | %13d | 0 (完全一致)\n",
           "Z/2^16Z", (int)q_fwd, (int)q_rev, (int)q_tree[0], 0);

    for (int fi = 0; fi < 3; fi++) {
        FPFormat fmt = (FPFormat)fi;
        float s_fwd = 0.0f, s_rev = 0.0f;
        static float s_tree[1024];
        for (int i = 0; i < N_RED; i++) {
            s_fwd     = fp_add(fmt, s_fwd, fp_q(fmt, vec_f[i]));
            s_rev     = fp_add(fmt, s_rev, fp_q(fmt, vec_f[N_RED - 1 - i]));
            s_tree[i] = fp_q(fmt, vec_f[i]);
        }
        for (int stride = 1; stride < N_RED; stride <<= 1) {
            for (int i = 0; i < N_RED; i += stride * 2) {
                s_tree[i] = fp_add(fmt, s_tree[i], s_tree[i + stride]);
            }
        }
        double d1 = dabs((double)s_fwd - (double)s_rev);
        double d2 = dabs((double)s_fwd - (double)s_tree[0]);
        double d3 = dabs((double)s_rev - (double)s_tree[0]);
        double max_gap = (d1 > d2) ? d1 : d2;
        if (d3 > max_gap) max_gap = d3;

        printf("  %-13s | %11.4f | %11.4f | %11.4f | %13.5f | %12.5f\n",
               fp_fmt_name(fmt), (double)s_fwd, (double)s_rev, (double)s_tree[0],
               max_gap, dabs((double)s_fwd - exact_sum));
    }
    printf("  (理論厳密和 = %.4f: 順序を変えただけで BF16/FP16 が大きく乖離)\n\n", exact_sum);
}

/* ============================================================================
 * 5. [Phase 2] 動的カリー化論理合成 (Toffoli3 / Carry) の Swamping 境界比較
 * ============================================================================ */
typedef bool (*BoolFn3)(bool, bool, bool);
static bool fn_toffoli3(bool a, bool b, bool c) { return (a && b && c); }
static bool fn_carry(bool a, bool b, bool c)    { return (a && b) || (b && c) || (c && a); }

typedef struct {
    int    epochs_to_exact;  /* 残差完全ゼロ到達エポック (-1 なら未収束/Swamping) */
    int    num_cells;
    int    logic_ok;         /* 8 パターン中の論理正答数 (0..8) */
    double max_res;          /* 最終最大残差 (0.0 が完全一致) */
    double max_rev_err;      /* 最終可逆復元誤差 |μ∘Δ(v) - v| */
} SynthResult;

/* Z/2^16Z 厳密固定小数点 LiftingCell 動的カリー化合成 */
static SynthResult run_synth_z16(BoolFn3 fn, uint8_t lr_shift, int max_epochs) {
    c16_t   w[16];
    uint8_t masks[16];
    int     in_s[16], in_g[16];
    int     nc = 3;

    memset(w, 0, sizeof(w));
    for (int i = 0; i < 3; i++) {
        masks[i] = (uint8_t)(1u << i);
        in_s[i]  = -1 - i; /* 外部入力ポート */
        in_g[i]  = -99;
    }

    int conv_ep = -1;
    for (int ep = 1; ep <= max_epochs; ep++) {
        int64_t known_err = 0, unknown_err = 0;
        int64_t pair_score[16][16];
        memset(pair_score, 0, sizeof(pair_score));

        for (int m = 0; m < 8; m++) {
            c16_t s_eff[16], r_arr[16];
            c16_t in_p[3] = {
                { (m & 1) ? Q14_ONE : 0, 0 },
                { (m & 2) ? Q14_ONE : 0, 0 },
                { (m & 4) ? Q14_ONE : 0, 0 }
            };
            c16_t cur_r = fn((m & 1) != 0, (m & 2) != 0, (m & 4) != 0)
                          ? (c16_t){0, Q14_NOT} : (c16_t){0, 0};

            for (int i = 0; i < nc; i++) {
                c16_t s = (in_s[i] < 0) ? in_p[-1 - in_s[i]] : s_eff[in_s[i]];
                c16_t g = (in_g[i] == -99) ? (c16_t){Q14_ONE, 0} : s_eff[in_g[i]];
                s_eff[i] = c16_mul_q14(g, s);
                c16_t p  = c16_mul_q14(w[i], s_eff[i]);
                cur_r    = c16_sub_mod(cur_r, p);
                r_arr[i] = cur_r;
            }
            for (int i = 0; i < nc; i++) {
                if (c16_l1_norm(s_eff[i]) > 0 && masks[i] == (uint8_t)m) {
                    update_complex_weight_mod(&w[i], r_arr[i], s_eff[i], lr_shift);
                }
            }
            int32_t en = c16_l1_norm(cur_r);
            if (en == 0) continue;

            bool has_m = false;
            for (int i = 0; i < nc; i++) if (masks[i] == (uint8_t)m) has_m = true;
            if (has_m) {
                known_err += en;
            } else {
                unknown_err += en;
                for (int u = 0; u < nc; u++) {
                    if (c16_l1_norm(s_eff[u]) == 0) continue;
                    for (int v = u + 1; v < nc; v++) {
                        if (c16_l1_norm(s_eff[v]) == 0) continue;
                        if ((masks[u] & masks[v]) != 0u) continue;
                        uint8_t um = (uint8_t)(masks[u] | masks[v]);
                        bool dup = false;
                        for (int k = 0; k < nc; k++) if (masks[k] == um) dup = true;
                        if (!dup) pair_score[u][v] += en;
                    }
                }
            }
        }

        if (known_err == 0 && unknown_err == 0) {
            conv_ep = ep;
            break;
        }
        if (known_err == 0 && unknown_err > 0 && nc < 16) {
            int bu = -1, bv = -1;
            int64_t bs = 0;
            uint8_t bb = 0;
            for (int u = 0; u < nc; u++) {
                for (int v = u + 1; v < nc; v++) {
                    if (pair_score[u][v] <= 0) continue;
                    uint8_t ub = popcount8((uint8_t)(masks[u] | masks[v]));
                    if (pair_score[u][v] > bs || (pair_score[u][v] == bs && ub > bb)) {
                        bs = pair_score[u][v]; bb = ub; bu = u; bv = v;
                    }
                }
            }
            if (bu >= 0) {
                in_s[nc]  = bu;
                in_g[nc]  = bv;
                masks[nc] = (uint8_t)(masks[bu] | masks[bv]);
                w[nc]     = (c16_t){0, 0};
                nc++;
            }
        }
    }

    /* 最終評価: 8 パターンすべての残差・論理一致・可逆復元検証 */
    SynthResult res = { conv_ep, nc, 0, 0.0, 0.0 };
    for (int m = 0; m < 8; m++) {
        c16_t s_eff[16], p_arr[16];
        c16_t in_p[3] = {
            { (m & 1) ? Q14_ONE : 0, 0 },
            { (m & 2) ? Q14_ONE : 0, 0 },
            { (m & 4) ? Q14_ONE : 0, 0 }
        };
        bool  exp_b = fn((m & 1) != 0, (m & 2) != 0, (m & 4) != 0);
        c16_t tgt   = exp_b ? (c16_t){0, Q14_NOT} : (c16_t){0, 0};
        c16_t cur_r = tgt;
        for (int i = 0; i < nc; i++) {
            c16_t s  = (in_s[i] < 0) ? in_p[-1 - in_s[i]] : s_eff[in_s[i]];
            c16_t g  = (in_g[i] == -99) ? (c16_t){Q14_ONE, 0} : s_eff[in_g[i]];
            s_eff[i] = c16_mul_q14(g, s);
            p_arr[i] = c16_mul_q14(w[i], s_eff[i]);
            cur_r    = c16_sub_mod(cur_r, p_arr[i]);
        }
        /* MODE_EVAL (td_r = 0 から合流) & MODE_EXACT (td_r = cur_r から合流) */
        c16_t pred_v = {0, 0}, rev_v = cur_r;
        for (int i = nc - 1; i >= 0; i--) {
            pred_v = c16_add_mod(pred_v, p_arr[i]);
            rev_v  = c16_add_mod(rev_v,  p_arr[i]);
        }
        double r_norm = (double)c16_l1_norm(cur_r) / 32768.0;
        if (r_norm > res.max_res) res.max_res = r_norm;
        double rev_e = (double)c16_l1_norm(c16_sub_mod(rev_v, tgt)) / 32768.0;
        if (rev_e > res.max_rev_err) res.max_rev_err = rev_e;

        bool pred_b = (c16_l1_norm(c16_sub_mod(pred_v, (c16_t){0, Q14_NOT})) <
                       c16_l1_norm(pred_v));
        if (pred_b == exp_b) res.logic_ok++;
    }
    return res;
}

/* IEEE 754 浮動小数点 (FP32 / FP16 / BF16) LiftingCell 動的カリー化合成 */
static SynthResult run_synth_fp(FPFormat fmt, BoolFn3 fn, uint8_t lr_shift, int max_epochs) {
    float   w[16];
    uint8_t masks[16];
    int     in_s[16], in_g[16];
    int     nc = 3;

    memset(w, 0, sizeof(w));
    for (int i = 0; i < 3; i++) {
        masks[i] = (uint8_t)(1u << i);
        in_s[i]  = -1 - i;
        in_g[i]  = -99;
    }

    float eta = fp_q(fmt, 1.0f / (float)(1u << lr_shift));
    int   conv_ep = -1;
    int   stagnation = 0;

    for (int ep = 1; ep <= max_epochs; ep++) {
        double known_err = 0.0, unknown_err = 0.0;
        double pair_score[16][16];
        memset(pair_score, 0, sizeof(pair_score));
        bool any_w_changed = false;

        for (int m = 0; m < 8; m++) {
            float s_eff[16], r_arr[16];
            float in_p[3] = {
                (m & 1) ? 1.0f : 0.0f,
                (m & 2) ? 1.0f : 0.0f,
                (m & 4) ? 1.0f : 0.0f
            };
            float cur_r = fn((m & 1) != 0, (m & 2) != 0, (m & 4) != 0) ? 1.0f : 0.0f;

            for (int i = 0; i < nc; i++) {
                float s  = (in_s[i] < 0) ? in_p[-1 - in_s[i]] : s_eff[in_s[i]];
                float g  = (in_g[i] == -99) ? 1.0f : s_eff[in_g[i]];
                s_eff[i] = fp_mul(fmt, g, s);
                float p  = fp_mul(fmt, w[i], s_eff[i]);
                cur_r    = fp_sub(fmt, cur_r, p);
                r_arr[i] = cur_r;
            }
            for (int i = 0; i < nc; i++) {
                if (s_eff[i] != 0.0f && masks[i] == (uint8_t)m) {
                    float grad  = fp_mul(fmt, r_arr[i], s_eff[i]);
                    float delta = fp_mul(fmt, eta, grad);
                    float w_new = fp_add(fmt, w[i], delta);
                    if (w_new != w[i]) any_w_changed = true;
                    w[i] = w_new;
                }
            }
            double en = dabs((double)cur_r);
            if (en == 0.0) continue;

            bool has_m = false;
            for (int i = 0; i < nc; i++) if (masks[i] == (uint8_t)m) has_m = true;
            if (has_m) {
                known_err += en;
            } else {
                unknown_err += en;
                for (int u = 0; u < nc; u++) {
                    if (s_eff[u] == 0.0f) continue;
                    for (int v = u + 1; v < nc; v++) {
                        if (s_eff[v] == 0.0f) continue;
                        if ((masks[u] & masks[v]) != 0u) continue;
                        uint8_t um = (uint8_t)(masks[u] | masks[v]);
                        bool dup = false;
                        for (int k = 0; k < nc; k++) if (masks[k] == um) dup = true;
                        if (!dup) pair_score[u][v] += en;
                    }
                }
            }
        }

        if (known_err == 0.0 && unknown_err == 0.0) {
            conv_ep = ep;
            break;
        }
        if (!any_w_changed) stagnation++;
        else                stagnation = 0;

        /* 既知セル収束、または Swamping で重み更新が完全停止した時点で次段カリー化を許可 */
        if ((known_err == 0.0 || stagnation >= 4) && unknown_err > 0.0 && nc < 16) {
            int bu = -1, bv = -1;
            double bs = 0.0;
            uint8_t bb = 0;
            for (int u = 0; u < nc; u++) {
                for (int v = u + 1; v < nc; v++) {
                    if (pair_score[u][v] <= 0.0) continue;
                    uint8_t ub = popcount8((uint8_t)(masks[u] | masks[v]));
                    if (pair_score[u][v] > bs || (pair_score[u][v] == bs && ub > bb)) {
                        bs = pair_score[u][v]; bb = ub; bu = u; bv = v;
                    }
                }
            }
            if (bu >= 0) {
                in_s[nc]   = bu;
                in_g[nc]   = bv;
                masks[nc]  = (uint8_t)(masks[bu] | masks[bv]);
                w[nc]      = 0.0f;
                nc++;
                stagnation = 0;
                continue;
            }
        }
        /* 全必要セルが揃ったうえで重みが 1 bit も動かなくなったら完全 Swamping 停止 */
        if (stagnation >= 8 && unknown_err == 0.0) {
            break;
        }
    }

    SynthResult res = { conv_ep, nc, 0, 0.0, 0.0 };
    for (int m = 0; m < 8; m++) {
        float s_eff[16], p_arr[16];
        float in_p[3] = {
            (m & 1) ? 1.0f : 0.0f,
            (m & 2) ? 1.0f : 0.0f,
            (m & 4) ? 1.0f : 0.0f
        };
        bool  exp_b = fn((m & 1) != 0, (m & 2) != 0, (m & 4) != 0);
        float tgt   = exp_b ? 1.0f : 0.0f;
        float cur_r = tgt;
        for (int i = 0; i < nc; i++) {
            float s  = (in_s[i] < 0) ? in_p[-1 - in_s[i]] : s_eff[in_s[i]];
            float g  = (in_g[i] == -99) ? 1.0f : s_eff[in_g[i]];
            s_eff[i] = fp_mul(fmt, g, s);
            p_arr[i] = fp_mul(fmt, w[i], s_eff[i]);
            cur_r    = fp_sub(fmt, cur_r, p_arr[i]);
        }
        float pred_v = 0.0f, rev_v = cur_r;
        for (int i = nc - 1; i >= 0; i--) {
            pred_v = fp_add(fmt, pred_v, p_arr[i]);
            rev_v  = fp_add(fmt, rev_v,  p_arr[i]);
        }
        double r_norm = dabs((double)cur_r);
        if (r_norm > res.max_res) res.max_res = r_norm;
        double rev_e = dabs((double)rev_v - (double)tgt);
        if (rev_e > res.max_rev_err) res.max_rev_err = rev_e;

        bool pred_b = (pred_v >= 0.5f);
        if (pred_b == exp_b) res.logic_ok++;
    }
    return res;
}

static void format_synth_cell(const SynthResult *r, char buf[32]) {
    if (r->epochs_to_exact > 0) {
        snprintf(buf, 32, "0-err (%4dep, %d/8)", r->epochs_to_exact, r->logic_ok);
    } else {
        snprintf(buf, 32, "r=%.3f (%d/8)%s", r->max_res, r->logic_ok,
                 (r->logic_ok < 8) ? " FAIL" : "");
    }
}

static void run_phase2_swamping_boundary(void) {
    const uint8_t shifts[]  = { 2, 4, 6, 8, 10 };
    const int     n_shifts  = (int)(sizeof(shifts) / sizeof(shifts[0]));
    const int     MAX_EP    = 40000;

    const char *names[2] = { "Toffoli3 (A*B*C)", "Carry (多数決 3入力)" };
    BoolFn3     fns[2]   = { fn_toffoli3, fn_carry };

    for (int fi = 0; fi < 2; fi++) {
        printf("[Phase 2-%c] 動的カリー化論理合成 Swamping 境界: %s (最大 %d epochs)\n",
               (fi == 0) ? 'A' : 'B', names[fi], MAX_EP);
        printf("  %-13s | %-19s | %-19s | %-19s | %s\n",
               "学習率 η", "Z/2^16Z (Q1.14)", "FP32 (23b)", "FP16 (10b)", "BF16 ( 7b)");
        printf("  --------------+---------------------+---------------------+---------------------+---------------------\n");

        for (int si = 0; si < n_shifts; si++) {
            uint8_t sh = shifts[si];
            SynthResult rz   = run_synth_z16(fns[fi], sh, MAX_EP);
            SynthResult rf32 = run_synth_fp(FP_FMT_FP32, fns[fi], sh, MAX_EP);
            SynthResult rf16 = run_synth_fp(FP_FMT_FP16, fns[fi], sh, MAX_EP);
            SynthResult rbf  = run_synth_fp(FP_FMT_BF16, fns[fi], sh, MAX_EP);

            if (rz.epochs_to_exact < 0 || rz.logic_ok != 8 || rz.max_rev_err != 0.0) {
                fprintf(stderr, "FAIL: Phase 2 Z/2^16Z 未収束 (lr_shift=%u)\n", (unsigned)sh);
                exit(1);
            }

            char b_z[32], b_32[32], b_16[32], b_bf[32];
            format_synth_cell(&rz,   b_z);
            format_synth_cell(&rf32, b_32);
            format_synth_cell(&rf16, b_16);
            format_synth_cell(&rbf,  b_bf);

            printf("  2^-%-2u (%.4f) | %-19s | %-19s | %-19s | %s\n",
                   (unsigned)sh, 1.0 / (double)(1u << sh), b_z, b_32, b_16, b_bf);
        }
        printf("\n");
    }
}

/* ============================================================================
 * 6. [Phase 3] 192-bit QC-LDPC 軟判定復号 (Q1.14 vs FP32 vs FP16 vs BF16)
 * ============================================================================ */
#define LDPC_Z    32
#define LDPC_ROWS 3
#define LDPC_COLS 6
#define LDPC_N    (LDPC_Z * LDPC_COLS) /* 192 */
#define LDPC_M    (LDPC_Z * LDPC_ROWS) /*  96 */
#define LDPC_E    (LDPC_M * LDPC_COLS) /* 576 */

typedef struct {
    uint8_t H[LDPC_M][LDPC_N];
    uint8_t rref[LDPC_M][LDPC_N];
    int     pivot_col[LDPC_M];
    int     free_col[LDPC_N];
    int     rank;
    int     k_info;

    int     chk_vars[LDPC_M][LDPC_COLS];
    int     chk_edges[LDPC_M][LDPC_COLS];
    int     var_edges[LDPC_N][LDPC_ROWS];
    int     edge_var[LDPC_E];
} LDPCGraph;

static void ldpc_init(LDPCGraph *g) {
    static const int shifts[LDPC_ROWS][LDPC_COLS] = {
        {  0,  0,  0,  0,  0,  0 },
        {  0,  1,  3,  7, 12, 20 },
        {  0,  2,  7, 15, 25,  6 }
    };
    memset(g->H, 0, sizeof(g->H));
    int v_deg[LDPC_N];
    memset(v_deg, 0, sizeof(v_deg));

    int eid = 0;
    for (int br = 0; br < LDPC_ROWS; br++) {
        for (int z = 0; z < LDPC_Z; z++) {
            int row = br * LDPC_Z + z;
            for (int bc = 0; bc < LDPC_COLS; bc++) {
                int col = bc * LDPC_Z + ((z + shifts[br][bc]) % LDPC_Z);
                g->H[row][col]         = 1u;
                g->chk_vars[row][bc]   = col;
                g->chk_edges[row][bc]  = eid;
                g->var_edges[col][v_deg[col]++] = eid;
                g->edge_var[eid]       = col;
                eid++;
            }
        }
    }

    memcpy(g->rref, g->H, sizeof(g->H));
    bool is_piv[LDPC_N];
    memset(is_piv, 0, sizeof(is_piv));
    int r = 0;
    for (int c = 0; c < LDPC_N && r < LDPC_M; c++) {
        int piv = -1;
        for (int i = r; i < LDPC_M; i++) {
            if (g->rref[i][c] != 0u) { piv = i; break; }
        }
        if (piv < 0) continue;
        if (piv != r) {
            for (int j = 0; j < LDPC_N; j++) {
                uint8_t tmp = g->rref[r][j];
                g->rref[r][j]   = g->rref[piv][j];
                g->rref[piv][j] = tmp;
            }
        }
        for (int i = 0; i < LDPC_M; i++) {
            if (i != r && g->rref[i][c] != 0u) {
                for (int j = c; j < LDPC_N; j++) g->rref[i][j] ^= g->rref[r][j];
            }
        }
        g->pivot_col[r] = c;
        is_piv[c]       = true;
        r++;
    }
    g->rank   = r;
    g->k_info = 0;
    for (int c = 0; c < LDPC_N; c++) {
        if (!is_piv[c]) g->free_col[g->k_info++] = c;
    }
}

static void ldpc_encode(const LDPCGraph *g, const uint8_t info[LDPC_N], uint8_t cw[LDPC_N]) {
    memset(cw, 0, LDPC_N);
    for (int i = 0; i < g->k_info; i++) cw[g->free_col[i]] = info[i] & 1u;
    for (int r = 0; r < g->rank; r++) {
        int     pc  = g->pivot_col[r];
        uint8_t acc = 0u;
        for (int i = 0; i < g->k_info; i++) {
            int fc = g->free_col[i];
            if (g->rref[r][fc]) acc ^= cw[fc];
        }
        cw[pc] = acc;
    }
}

static bool ldpc_syndrome_ok(const LDPCGraph *g, const uint8_t bits[LDPC_N]) {
    for (int r = 0; r < LDPC_M; r++) {
        uint8_t parity = 0u;
        for (int k = 0; k < LDPC_COLS; k++) parity ^= bits[g->chk_vars[r][k]];
        if (parity != 0u) return false;
    }
    return true;
}

/* Z/2^16Z (Q1.14) 固定小数点 LiftingCell 軟判定復号 */
static int ldpc_decode_q14(const LDPCGraph *g, const int32_t ch_llr[LDPC_N],
                           int max_iters, uint8_t out[LDPC_N],
                           int64_t *sat_cnt, int64_t *mu_total) {
    int32_t v2c[LDPC_E], c2v[LDPC_E];
    for (int v = 0; v < LDPC_N; v++) out[v] = (ch_llr[v] < 0) ? 1u : 0u;
    if (ldpc_syndrome_ok(g, out)) return 0;

    for (int e = 0; e < LDPC_E; e++) {
        v2c[e] = clamp_i32(ch_llr[g->edge_var[e]], -LLR_MAX, LLR_MAX);
        c2v[e] = 0;
    }

    for (int iter = 1; iter <= max_iters; iter++) {
        for (int r = 0; r < LDPC_M; r++) {
            c16_t mu[LDPC_COLS], pref[LDPC_COLS], suff[LDPC_COLS];
            for (int k = 0; k < LDPC_COLS; k++) {
                int16_t m = llr_to_phase_q14(v2c[g->chk_edges[r][k]]);
                mu[k] = (c16_t){ m, 0 };
                (*mu_total)++;
                if (m >= Q14_ONE || m <= -Q14_ONE) (*sat_cnt)++;
            }
            pref[0] = (c16_t){ Q14_ONE, 0 };
            for (int k = 1; k < LDPC_COLS; k++) {
                pref[k] = c16_mul_q14(pref[k - 1], mu[k - 1]);
            }
            suff[LDPC_COLS - 1] = (c16_t){ Q14_ONE, 0 };
            for (int k = LDPC_COLS - 2; k >= 0; k--) {
                suff[k] = c16_mul_q14(suff[k + 1], mu[k + 1]);
            }
            for (int k = 0; k < LDPC_COLS; k++) {
                c16_t   ext = c16_mul_q14(pref[k], suff[k]);
                int32_t nl  = phase_q14_to_llr(ext.re);
                int     eid = g->chk_edges[r][k];
                if (iter > 1) nl = (nl * 3 + c2v[eid]) >> 2;
                c2v[eid] = clamp_i32(nl, -LLR_MAX, LLR_MAX);
            }
        }
        for (int v = 0; v < LDPC_N; v++) {
            int32_t tot = ch_llr[v];
            for (int k = 0; k < LDPC_ROWS; k++) tot += c2v[g->var_edges[v][k]];
            out[v] = (tot < 0) ? 1u : 0u;
            for (int k = 0; k < LDPC_ROWS; k++) {
                int eid = g->var_edges[v][k];
                v2c[eid] = clamp_i32(tot - c2v[eid], -LLR_MAX, LLR_MAX);
            }
        }
        if (ldpc_syndrome_ok(g, out)) return iter;
    }
    return -1;
}

/* IEEE 754 (FP32 / FP16 / BF16) 軟判定復号 (全中間メッセージ・乗算・加算を各精度へ厳密丸め) */
static int ldpc_decode_fp(const LDPCGraph *g, FPFormat fmt, const float ch_llr_in[LDPC_N],
                          int max_iters, uint8_t out[LDPC_N],
                          int64_t *sat_cnt, int64_t *mu_total) {
    const float MAX_L = 8.0f;
    float ch_llr[LDPC_N], v2c[LDPC_E], c2v[LDPC_E];

    for (int v = 0; v < LDPC_N; v++) {
        float c = ch_llr_in[v];
        if (c >  MAX_L) c =  MAX_L;
        if (c < -MAX_L) c = -MAX_L;
        ch_llr[v] = fp_q(fmt, c);
        out[v]    = (ch_llr[v] < 0.0f) ? 1u : 0u;
    }
    if (ldpc_syndrome_ok(g, out)) return 0;

    for (int e = 0; e < LDPC_E; e++) {
        v2c[e] = ch_llr[g->edge_var[e]];
        c2v[e] = 0.0f;
    }

    const float damp_new = fp_q(fmt, 0.75f);
    const float damp_old = fp_q(fmt, 0.25f);

    for (int iter = 1; iter <= max_iters; iter++) {
        for (int r = 0; r < LDPC_M; r++) {
            float mu[LDPC_COLS], pref[LDPC_COLS], suff[LDPC_COLS];
            for (int k = 0; k < LDPC_COLS; k++) {
                float m = fp_q(fmt, (float)tanh_exact(0.5 * (double)v2c[g->chk_edges[r][k]]));
                mu[k]   = m;
                (*mu_total)++;
                if (m >= 1.0f || m <= -1.0f) (*sat_cnt)++;
            }
            pref[0] = 1.0f;
            for (int k = 1; k < LDPC_COLS; k++) {
                pref[k] = fp_mul(fmt, pref[k - 1], mu[k - 1]);
            }
            suff[LDPC_COLS - 1] = 1.0f;
            for (int k = LDPC_COLS - 2; k >= 0; k--) {
                suff[k] = fp_mul(fmt, suff[k + 1], mu[k + 1]);
            }
            for (int k = 0; k < LDPC_COLS; k++) {
                float ext = fp_mul(fmt, pref[k], suff[k]);
                float nl  = fp_q(fmt, (float)two_atanh_clamped((double)ext, (double)MAX_L));
                int   eid = g->chk_edges[r][k];
                if (iter > 1) {
                    nl = fp_add(fmt, fp_mul(fmt, nl, damp_new), fp_mul(fmt, c2v[eid], damp_old));
                }
                if (nl >  MAX_L) nl =  MAX_L;
                if (nl < -MAX_L) nl = -MAX_L;
                c2v[eid] = fp_q(fmt, nl);
            }
        }
        for (int v = 0; v < LDPC_N; v++) {
            float tot = ch_llr[v];
            for (int k = 0; k < LDPC_ROWS; k++) {
                tot = fp_add(fmt, tot, c2v[g->var_edges[v][k]]);
            }
            out[v] = (tot < 0.0f) ? 1u : 0u;
            for (int k = 0; k < LDPC_ROWS; k++) {
                int   eid = g->var_edges[v][k];
                float ext = fp_sub(fmt, tot, c2v[eid]);
                if (ext >  MAX_L) ext =  MAX_L;
                if (ext < -MAX_L) ext = -MAX_L;
                v2c[eid] = fp_q(fmt, ext);
            }
        }
        if (ldpc_syndrome_ok(g, out)) return iter;
    }
    return -1;
}

static void run_phase3_ldpc_precision_ablation(void) {
    static LDPCGraph g;
    ldpc_init(&g);

    printf("[Phase 3] 192-bit QC-LDPC 軟判定復号 精度対照比較 (各 200 同一ノイズフレーム, 最大 40 iters):\n");
    printf("  同一 16-bit メモリ幅 (Z/2^16Z Q1.14 vs FP16 vs BF16) + 32-bit 参照 (FP32)\n");
    printf("  %-8s | %-18s | %-18s | %-18s | %-22s\n",
           "Eb/N0", "Z/2^16Z Q1.14(16b)", "FP32 (32b 参照)", "FP16 (16b 模擬)", "BF16 (16b 模擬 / 飽和率)");
    printf("  ---------+--------------------+--------------------+--------------------+------------------------\n");

    const double eb_db[]  = { 1.5,    2.0,    2.5,    3.0,    3.5,    4.0    };
    const double sigmas[] = { 0.8328, 0.7862, 0.7422, 0.7007, 0.6615, 0.6245 };
    const int    n_snr    = (int)(sizeof(eb_db) / sizeof(eb_db[0]));
    const int    N_FR     = 200;
    uint32_t     rng      = 0x20261123u;

    for (int si = 0; si < n_snr; si++) {
        double sigma    = sigmas[si];
        double inv_var2 = 2.0 / (sigma * sigma);

        int     ber_cnt[4]  = {0, 0, 0, 0}; /* 0:Q14, 1:FP32, 2:FP16, 3:BF16 */
        int     fer_cnt[4]  = {0, 0, 0, 0};
        int64_t sat_cnt[4]  = {0, 0, 0, 0};
        int64_t mu_tot[4]   = {0, 0, 0, 0};

        for (int fr = 0; fr < N_FR; fr++) {
            uint8_t info[LDPC_N], cw[LDPC_N], dec[LDPC_N];
            int32_t llr_q[LDPC_N];
            float   llr_f[LDPC_N];

            for (int i = 0; i < g.k_info; i++) info[i] = (uint8_t)(next_rng(&rng) & 1u);
            ldpc_encode(&g, info, cw);

            for (int i = 0; i < LDPC_N; i++) {
                double tx  = (cw[i] == 0u) ? 1.0 : -1.0;
                double rx  = tx + sigma * next_gauss_std(&rng);
                double llr = rx * inv_var2;
                llr_f[i]   = (float)llr;
                llr_q[i]   = clamp_i32((int32_t)(llr * (double)LLR_SCALE), -LLR_MAX, LLR_MAX);
            }

            /* 0: Z/2^16Z Q1.14 */
            int it0 = ldpc_decode_q14(&g, llr_q, 40, dec, &sat_cnt[0], &mu_tot[0]);
            int e0  = 0;
            for (int i = 0; i < LDPC_N; i++) if (dec[i] != cw[i]) e0++;
            ber_cnt[0] += e0;
            if (it0 < 0 || e0 > 0) fer_cnt[0]++;

            /* 1..3: FP32 / FP16 / BF16 */
            for (int fi = 0; fi < 3; fi++) {
                int it = ldpc_decode_fp(&g, (FPFormat)fi, llr_f, 40, dec,
                                        &sat_cnt[1 + fi], &mu_tot[1 + fi]);
                int e  = 0;
                for (int i = 0; i < LDPC_N; i++) if (dec[i] != cw[i]) e++;
                ber_cnt[1 + fi] += e;
                if (it < 0 || e > 0) fer_cnt[1 + fi]++;
            }
        }

        double tot_bits = (double)(N_FR * LDPC_N);
        double bf_sat   = (mu_tot[3] > 0) ? (100.0 * (double)sat_cnt[3] / (double)mu_tot[3]) : 0.0;

        printf("  %4.1f dB  | %.5f (F=%.3f)   | %.5f (F=%.3f)   | %.5f (F=%.3f)   | %.5f (F=%.3f, s=%4.1f%%)\n",
               eb_db[si],
               (double)ber_cnt[0] / tot_bits, (double)fer_cnt[0] / (double)N_FR,
               (double)ber_cnt[1] / tot_bits, (double)fer_cnt[1] / (double)N_FR,
               (double)ber_cnt[2] / tot_bits, (double)fer_cnt[2] / (double)N_FR,
               (double)ber_cnt[3] / tot_bits, (double)fer_cnt[3] / (double)N_FR,
               bf_sat);

        if (si == n_snr - 1 && fer_cnt[0] > 5) {
            fprintf(stderr, "FAIL: Phase 3 Q1.14 4.0 dB 復号精度不足\n");
            exit(1);
        }
    }
    printf("  (F=FER フレーム誤り率, s=チェックノードで |tanh(L/2)| が BF16 丸めにより 1.0 へ飽和した割合)\n");
}

/* ============================================================================
 * 7. Phase 0: 丸めエミュレータ・初等関数の単体自己検証 & メインエントリ
 * ============================================================================ */
static void verify_math_and_ieee_emulators(void) {
    /* 1. 初等関数精度検証 */
    double e1 = dabs(exp_fast(1.0) - 2.718281828459045);
    double l1 = dabs(ln_fast(2.718281828459045) - 1.0);
    double rt = dabs(two_atanh_clamped(tanh_exact(1.75), 8.0) - 3.5);
    if (e1 > 1e-11 || l1 > 1e-11 || rt > 1e-10) {
        fprintf(stderr, "FATAL: 初等関数精度検証失敗 (e1=%g, l1=%g, rt=%g)\n", e1, l1, rt);
        exit(1);
    }

    /* 2. BF16 ties-to-even & Swamping 境界検証: ulp(1.0) = 2^-7 = 0.0078125 */
    if (quantize_bf16(1.0f + 0.00390625f) != 1.0f ||      /* ちょうど half-ULP -> 偶数 1.0 へ */
        quantize_bf16(1.0f + 0.0040f)     != 1.0078125f) { /* half-ULP 超 -> 次表現へ */
        fprintf(stderr, "FATAL: BF16 丸め検証失敗\n");
        exit(1);
    }

    /* 3. FP16 ties-to-even & サブノーマル検証: ulp(1.0) = 2^-10 = 0.0009765625 */
    if (quantize_fp16(1.0f + 0.00048828125f) != 1.0f ||
        quantize_fp16(1.0f + 0.0005f)        != 1.0009765625f ||
        quantize_fp16(1.0f / 16777216.0f)    != (1.0f / 16777216.0f)) {
        fprintf(stderr, "FATAL: FP16 丸め検証失敗\n");
        exit(1);
    }
}

int main(void) {
    verify_math_and_ieee_emulators();
    init_q14_luts();

    printf("=== Z/2^16Z 厳密固定小数点 vs IEEE 754 (FP32 / FP16 / BF16) 対照実験 ===\n");
    printf("[Phase 0] IEEE 754 (FP16/BF16 ties-to-even) & -lm 不要初等関数 自己検証: OK\n\n");

    run_phase1_reversibility_and_associativity();
    run_phase2_swamping_boundary();
    run_phase3_ldpc_precision_ablation();

    printf("\n>>> ALL ASSERTIONS PASSED: 厳密固定小数点 vs IEEE 754 浮動小数点の劣化境界計測完了 <<<\n");
    return 0;
}