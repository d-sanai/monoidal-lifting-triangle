/* lifting_v2_ladder256_ffi.c — Multi-Channel (256-D x 64-Layer) O(1)-Memory
 * Backprop-Free Local 2-Morphism Learner Extension for liblifting_core_v2.dylib */
#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "lifting_core_v2.h"

#if defined(__GNUC__) || defined(__clang__)
#define LC_API      __attribute__((visibility("default")))
#define LC_ALIGN128 __attribute__((aligned(128)))
#else
#define LC_API
#define LC_ALIGN128
#endif

#define LC_L256_MAX_DIM     512
#define LC_L256_MAX_BLOCKS  (LC_L256_MAX_DIM / 8)
#define LC_L256_MAX_THREADS 64

static inline int32_t lc_l256_clamp32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

#if !LC_HAS_ARM_NEON
static inline int16_t lc_l256_asr1_s16(int16_t x) {
    if (x == LC_LOG_ZERO) return LC_LOG_ZERO;
    return (x >= 0) ? (int16_t)(x >> 1) : (int16_t)(-(((-(int32_t)x) + 1) >> 1));
}
#endif

static double lc_l256_sqrt(double x) {
    if (x <= 0.0) return 0.0;
    double y = (x >= 1.0) ? x : 1.0;
    for (int i = 0; i < 24; i++) y = 0.5 * (y + x / y);
    return y;
}

/* ============================================================================
 * 1. 単一サンプル D 次元 (D = 8 * n_blk) 予測・順逆リフティング (ARM64 NEON 駆動)
 * ============================================================================ */
static void lc_l256_prep_features(const lc_c16_t *s_row, int dim, int shift, uint8_t mode,
                                  lc_chart_block8_t *ch, lc_log_block8_t *lb) {
    lc_c16_t shifted[LC_L256_MAX_DIM];
    int sh = shift % dim;
    if (sh < 0) sh += dim;

    if (sh == 0) {
        memcpy(shifted, s_row, (size_t)dim * sizeof(lc_c16_t));
    } else {
        int tail = dim - sh;
        memcpy(&shifted[0],    &s_row[sh], (size_t)tail * sizeof(lc_c16_t));
        memcpy(&shifted[tail], &s_row[0],  (size_t)sh   * sizeof(lc_c16_t));
    }

    int n_blk = dim >> 3;
    for (int bi = 0; bi < n_blk; bi++) {
#if LC_HAS_ARM_NEON
        int16_t raw[16];
        memcpy(raw, &shifted[bi * 8], sizeof(raw));
        int16x8x2_t pair = vld2q_s16(raw);
        vst1q_s16(ch[bi].re, pair.val[0]);
        vst1q_s16(ch[bi].im, pair.val[1]);
#else
        for (int k = 0; k < 8; k++) {
            ch[bi].re[k] = shifted[bi * 8 + k].re;
            ch[bi].im[k] = shifted[bi * 8 + k].im;
        }
#endif
    }

    /* lifting_core_v2.h の ARM64 NEON 純整数 Cart->Log 逆写像を一括呼び出し */
    lc_chart_aosoa8_cart_to_log(ch, n_blk);

#if LC_HAS_ARM_NEON
    const int16x8_t  z_log = vdupq_n_s16(LC_LOG_ZERO);
    const uint16x8_t z_u16 = vdupq_n_u16(0u);
    for (int bi = 0; bi < n_blk; bi++) {
        int16x8_t  r = vld1q_s16(ch[bi].rho);
        uint16x8_t t = vld1q_u16(ch[bi].theta);
        if (mode == 1u) {
            uint16x8_t is_z = vceqq_s16(r, z_log);
            r = vbslq_s16(is_z, z_log, vshrq_n_s16(r, 1));
            t = vshlq_n_u16(t, 1);
        } else if (mode == 2u) {
            uint16x8_t is_z = vceqq_s16(r, z_log);
            r = vbslq_s16(is_z, z_log, vshrq_n_s16(r, 1));
            t = vsubq_u16(z_u16, vshlq_n_u16(t, 1));
        } else if (mode == 3u) {
            t = z_u16;
        }
        vst1q_s16(lb[bi].s_rho,   r);
        vst1q_u16(lb[bi].s_theta, t);
    }
#else
    for (int bi = 0; bi < n_blk; bi++) {
        for (int k = 0; k < 8; k++) {
            int16_t  r = ch[bi].rho[k];
            uint16_t t = ch[bi].theta[k];
            if (mode == 1u) {
                r = lc_l256_asr1_s16(r);
                t = (uint16_t)(t << 1);
            } else if (mode == 2u) {
                r = lc_l256_asr1_s16(r);
                t = (uint16_t)(0u - (uint16_t)(t << 1));
            } else if (mode == 3u) {
                t = 0u;
            }
            lb[bi].s_rho[k]   = r;
            lb[bi].s_theta[k] = t;
        }
    }
#endif
}

static void lc_l256_load_weights_and_dual(const lc_logc16_t *w_row, int dim,
                                          lc_log_block8_t *lb) {
    int n_blk = dim >> 3;
    for (int bi = 0; bi < n_blk; bi++) {
#if LC_HAS_ARM_NEON
        if (w_row != NULL) {
            int16_t raw[16];
            memcpy(raw, &w_row[bi * 8], sizeof(raw));
            int16x8x2_t wp = vld2q_s16(raw);
            vst1q_s16(lb[bi].w_rho,   wp.val[0]);
            vst1q_u16(lb[bi].w_theta, vreinterpretq_u16_s16(wp.val[1]));
        } else {
            vst1q_s16(lb[bi].w_rho,   vdupq_n_s16(0));
            vst1q_u16(lb[bi].w_theta, vdupq_n_u16(0u));
        }
#else
        for (int k = 0; k < 8; k++) {
            lb[bi].w_rho[k]   = (w_row != NULL) ? w_row[bi * 8 + k].rho   : 0;
            lb[bi].w_theta[k] = (w_row != NULL) ? w_row[bi * 8 + k].theta : 0u;
        }
#endif
    }
    /* lifting_core_v2.h の ARM64 NEON Log-Polar Bind + Log->Cart LUT を一括呼び出し */
    lc_log_aosoa8_dual_step(lb, n_blk);
}

static void lc_l256_step_sample(lc_c16_t *s_row, lc_c16_t *v_row, int dim,
                                const lc_logc16_t *w_row, const lc_c16_t *b_row,
                                uint8_t mode, int shift, bool is_inverse,
                                lc_chart_block8_t *ch, lc_log_block8_t *lb) {
    int n_blk = dim >> 3;
    if (!is_inverse) {
        lc_l256_prep_features(s_row, dim, shift, mode, ch, lb);
        lc_l256_load_weights_and_dual(w_row, dim, lb);

        for (int bi = 0; bi < n_blk; bi++) {
#if LC_HAS_ARM_NEON
            int16_t raw_s[16], raw_v[16], raw_b[16], raw_out[16];
            memcpy(raw_s, &s_row[bi * 8], sizeof(raw_s));
            memcpy(raw_v, &v_row[bi * 8], sizeof(raw_v));
            memcpy(raw_b, &b_row[bi * 8], sizeof(raw_b));
            int16x8x2_t vp = vld2q_s16(raw_v);
            int16x8x2_t bp = vld2q_s16(raw_b);
            uint16x8_t pre = vaddq_u16(vreinterpretq_u16_s16(vld1q_s16(lb[bi].p_re)),
                                       vreinterpretq_u16_s16(bp.val[0]));
            uint16x8_t pim = vaddq_u16(vreinterpretq_u16_s16(vld1q_s16(lb[bi].p_im)),
                                       vreinterpretq_u16_s16(bp.val[1]));
            int16x8x2_t vn;
            vn.val[0] = vreinterpretq_s16_u16(vsubq_u16(vreinterpretq_u16_s16(vp.val[0]), pre));
            vn.val[1] = vreinterpretq_s16_u16(vsubq_u16(vreinterpretq_u16_s16(vp.val[1]), pim));
            vst2q_s16(raw_out, vn);
            memcpy(&v_row[bi * 8], raw_s,   sizeof(raw_s));
            memcpy(&s_row[bi * 8], raw_out, sizeof(raw_out));
#else
            for (int k = 0; k < 8; k++) {
                int idx = bi * 8 + k;
                lc_c16_t p  = lc_c16_add_mod((lc_c16_t){lb[bi].p_re[k], lb[bi].p_im[k]}, b_row[idx]);
                lc_c16_t vn = lc_c16_sub_mod(v_row[idx], p);
                v_row[idx]  = s_row[idx];
                s_row[idx]  = vn;
            }
#endif
        }
    } else {
        /* 逆写像: s_prev = v_row, vn = s_row -> v_prev = vn ⊕ (P(s_prev) ⊕ b) */
        lc_l256_prep_features(v_row, dim, shift, mode, ch, lb);
        lc_l256_load_weights_and_dual(w_row, dim, lb);

        for (int bi = 0; bi < n_blk; bi++) {
#if LC_HAS_ARM_NEON
            int16_t raw_sp[16], raw_vn[16], raw_b[16], raw_out[16];
            memcpy(raw_sp, &v_row[bi * 8], sizeof(raw_sp));
            memcpy(raw_vn, &s_row[bi * 8], sizeof(raw_vn));
            memcpy(raw_b,  &b_row[bi * 8], sizeof(raw_b));
            int16x8x2_t vnp = vld2q_s16(raw_vn);
            int16x8x2_t bp  = vld2q_s16(raw_b);
            uint16x8_t pre = vaddq_u16(vreinterpretq_u16_s16(vld1q_s16(lb[bi].p_re)),
                                       vreinterpretq_u16_s16(bp.val[0]));
            uint16x8_t pim = vaddq_u16(vreinterpretq_u16_s16(vld1q_s16(lb[bi].p_im)),
                                       vreinterpretq_u16_s16(bp.val[1]));
            int16x8x2_t vprev;
            vprev.val[0] = vreinterpretq_s16_u16(vaddq_u16(vreinterpretq_u16_s16(vnp.val[0]), pre));
            vprev.val[1] = vreinterpretq_s16_u16(vaddq_u16(vreinterpretq_u16_s16(vnp.val[1]), pim));
            vst2q_s16(raw_out, vprev);
            memcpy(&s_row[bi * 8], raw_sp,  sizeof(raw_sp));
            memcpy(&v_row[bi * 8], raw_out, sizeof(raw_out));
#else
            for (int k = 0; k < 8; k++) {
                int idx = bi * 8 + k;
                lc_c16_t s_prev = v_row[idx];
                lc_c16_t vn     = s_row[idx];
                lc_c16_t p      = lc_c16_add_mod((lc_c16_t){lb[bi].p_re[k], lb[bi].p_im[k]}, b_row[idx]);
                s_row[idx] = s_prev;
                v_row[idx] = lc_c16_add_mod(vn, p);
            }
#endif
        }
    }
}

/* ============================================================================
 * 2. 128-byte 整列・ゼロヒープ pthread 並列プール (0-bit 決定論的リダクション)
 * ============================================================================ */
typedef enum {
    LC_L256_JOB_FORWARD = 0,
    LC_L256_JOB_INVERSE = 1,
    LC_L256_JOB_REV_ACC = 2
} lc_l256_job_t;

typedef struct LC_ALIGN128 {
    int64_t sum_gre[LC_L256_MAX_DIM];
    int64_t sum_gim[LC_L256_MAX_DIM];
    int64_t sum_u2[LC_L256_MAX_DIM];
    int64_t sum_dre[LC_L256_MAX_DIM];
    int64_t sum_dim[LC_L256_MAX_DIM];
} lc_l256_acc_t;

typedef struct {
    lc_l256_job_t      job;
    lc_c16_t          *up_s;
    lc_c16_t          *up_v;
    lc_c16_t          *dn_s;
    lc_c16_t          *dn_v;
    const lc_logc16_t *w;
    const lc_c16_t    *b;
    const uint8_t     *modes;
    const int32_t     *shifts;
    int                n_samples;
    int                dim;
    int                depth;
    int                cur_layer;
    int                num_threads;

    LC_ALIGN128 pthread_mutex_t mtx;
    pthread_cond_t  cv_start;
    pthread_cond_t  cv_done;
    uint64_t        epoch;
    int             max_workers;
    int             done_workers;
    bool            initialized;
    bool            stop;
} lc_l256_pool_t;

typedef struct {
    lc_l256_pool_t *pool;
    int             tid;
} lc_l256_arg_t;

static lc_l256_pool_t g_l256_pool;
static lc_l256_arg_t  g_l256_args[LC_L256_MAX_THREADS];
static pthread_t      g_l256_tids[LC_L256_MAX_THREADS];
static lc_l256_acc_t  g_l256_acc[LC_L256_MAX_THREADS];

static void lc_l256_init_lb_identity_g(lc_log_block8_t *lb, int n_blk) {
    memset(lb, 0, (size_t)n_blk * sizeof(lc_log_block8_t));
}

static void lc_l256_exec_slice(const lc_l256_pool_t *p, int tid, int i0, int i1) {
    lc_chart_block8_t LC_ALIGN64 ch[LC_L256_MAX_BLOCKS];
    lc_log_block8_t   LC_ALIGN64 lb[LC_L256_MAX_BLOCKS];

    int dim   = p->dim;
    int n_blk = dim >> 3;
    lc_l256_init_lb_identity_g(lb, n_blk);

    if (p->job == LC_L256_JOB_FORWARD) {
        for (int i = i0; i < i1; i++) {
            lc_c16_t *sr = &p->up_s[(size_t)i * (size_t)dim];
            lc_c16_t *vr = &p->up_v[(size_t)i * (size_t)dim];
            for (int L = 0; L < p->depth; L++) {
                lc_l256_step_sample(sr, vr, dim,
                                    &p->w[(size_t)L * (size_t)dim],
                                    &p->b[(size_t)L * (size_t)dim],
                                    p->modes[L], p->shifts[L], false, ch, lb);
            }
        }
    } else if (p->job == LC_L256_JOB_INVERSE) {
        for (int i = i0; i < i1; i++) {
            lc_c16_t *sr = &p->up_s[(size_t)i * (size_t)dim];
            lc_c16_t *vr = &p->up_v[(size_t)i * (size_t)dim];
            for (int L = p->depth - 1; L >= 0; L--) {
                lc_l256_step_sample(sr, vr, dim,
                                    &p->w[(size_t)L * (size_t)dim],
                                    &p->b[(size_t)L * (size_t)dim],
                                    p->modes[L], p->shifts[L], true, ch, lb);
            }
        }
    } else if (p->job == LC_L256_JOB_REV_ACC) {
        int L = p->cur_layer;
        const lc_logc16_t *w_row = &p->w[(size_t)L * (size_t)dim];
        const lc_c16_t    *b_row = &p->b[(size_t)L * (size_t)dim];
        uint8_t mode = p->modes[L];
        int     sh   = p->shifts[L];

        lc_l256_acc_t *acc = &g_l256_acc[tid];
        memset(acc->sum_gre, 0, (size_t)dim * sizeof(int64_t));
        memset(acc->sum_gim, 0, (size_t)dim * sizeof(int64_t));
        memset(acc->sum_u2,  0, (size_t)dim * sizeof(int64_t));
        memset(acc->sum_dre, 0, (size_t)dim * sizeof(int64_t));
        memset(acc->sum_dim, 0, (size_t)dim * sizeof(int64_t));

        int32_t d_re[LC_L256_MAX_DIM], d_im[LC_L256_MAX_DIM];

        for (int i = i0; i < i1; i++) {
            size_t off   = (size_t)i * (size_t)dim;
            lc_c16_t *us = &p->up_s[off], *uv = &p->up_v[off];
            lc_c16_t *ds = &p->dn_s[off], *dv = &p->dn_v[off];

            /* 1. 層 L 出口での局所残差 d_L = up_s_{L+1} - dn_s_{L+1} を確保 */
            for (int d = 0; d < dim; d++) {
                d_re[d] = (int32_t)us[d].re - (int32_t)ds[d].re;
                d_im[d] = (int32_t)us[d].im - (int32_t)ds[d].im;
            }

            /* 2. ターゲットストリームを逆走 (dn_s, dn_v -> 層 L ターゲット) */
            lc_l256_step_sample(ds, dv, dim, w_row, b_row, mode, sh, true, ch, lb);

            /* 3. 順流ストリームを逆走 (ここで lb[bi].s_rho/s_theta に特徴 zf が確定) */
            lc_l256_step_sample(us, uv, dim, w_row, b_row, mode, sh, true, ch, lb);

            /* 4. 同じ lb の重みを (0,0) にして Dual-Chart 変換 -> u = log_to_cart(zf) を NEON 算出 */
            lc_l256_load_weights_and_dual(NULL, dim, lb);

            /* 5. X-Spider 複素相関を int64_t 厳密積算 (結合律により 1T == MT 完全一致) */
            for (int bi = 0; bi < n_blk; bi++) {
                for (int k = 0; k < 8; k++) {
                    int     d  = bi * 8 + k;
                    int64_t dr = (int64_t)d_re[d];
                    int64_t di = (int64_t)d_im[d];
                    int64_t ur = (int64_t)lb[bi].p_re[k];
                    int64_t ui = (int64_t)lb[bi].p_im[k];
                    acc->sum_gre[d] += dr * ur + di * ui;
                    acc->sum_gim[d] += di * ur - dr * ui;
                    acc->sum_u2[d]  += ur * ur + ui * ui;
                    acc->sum_dre[d] += dr;
                    acc->sum_dim[d] += di;
                }
            }
        }
    }
}

static void *lc_l256_worker(void *arg) {
    lc_l256_arg_t  *wa   = (lc_l256_arg_t *)arg;
    lc_l256_pool_t *pool = wa->pool;
    int             tid  = wa->tid;
    uint64_t        seen = 0;

    for (;;) {
        pthread_mutex_lock(&pool->mtx);
        while (!pool->stop && pool->epoch == seen) {
            pthread_cond_wait(&pool->cv_start, &pool->mtx);
        }
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mtx);
            break;
        }
        seen = pool->epoch;
        int n_samp = pool->n_samples;
        int n_thr  = pool->num_threads;
        pthread_mutex_unlock(&pool->mtx);

        if (tid < n_thr) {
            int i0 = (n_samp * tid) / n_thr;
            int i1 = (n_samp * (tid + 1)) / n_thr;
            if (i1 > i0) {
                lc_l256_exec_slice(pool, tid, i0, i1);
            } else if (pool->job == LC_L256_JOB_REV_ACC) {
                memset(&g_l256_acc[tid], 0, sizeof(lc_l256_acc_t));
            }
        }

        pthread_mutex_lock(&pool->mtx);
        pool->done_workers++;
        if (pool->done_workers == pool->max_workers) {
            pthread_cond_signal(&pool->cv_done);
        }
        pthread_mutex_unlock(&pool->mtx);
    }
    return NULL;
}

static void lc_l256_dispatch(lc_l256_job_t job, int cur_layer, int num_threads) {
    if (num_threads <= 1) {
        g_l256_pool.job         = job;
        g_l256_pool.cur_layer   = cur_layer;
        g_l256_pool.num_threads = 1;
        lc_l256_exec_slice(&g_l256_pool, 0, 0, g_l256_pool.n_samples);
        return;
    }
    pthread_mutex_lock(&g_l256_pool.mtx);
    g_l256_pool.job          = job;
    g_l256_pool.cur_layer    = cur_layer;
    g_l256_pool.num_threads  = num_threads;
    g_l256_pool.done_workers = 0;
    g_l256_pool.epoch++;
    pthread_cond_broadcast(&g_l256_pool.cv_start);
    while (g_l256_pool.done_workers < g_l256_pool.max_workers) {
        pthread_cond_wait(&g_l256_pool.cv_done, &g_l256_pool.mtx);
    }
    pthread_mutex_unlock(&g_l256_pool.mtx);
}

/* ============================================================================
 * 3. 公開 C ABI (liblifting_core_v2.dylib エクスポート関数群)
 * ============================================================================ */
LC_API int lc_v2_l256_mt_init(int max_workers) {
    if (g_l256_pool.initialized) return g_l256_pool.max_workers;
    if (max_workers <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        max_workers = (n > 0) ? (int)n : 4;
    }
    if (max_workers > LC_L256_MAX_THREADS) max_workers = LC_L256_MAX_THREADS;

    memset(&g_l256_pool, 0, sizeof(g_l256_pool));
    if (pthread_mutex_init(&g_l256_pool.mtx, NULL) != 0) return -1;
    if (pthread_cond_init(&g_l256_pool.cv_start, NULL) != 0) {
        pthread_mutex_destroy(&g_l256_pool.mtx);
        return -2;
    }
    if (pthread_cond_init(&g_l256_pool.cv_done, NULL) != 0) {
        pthread_cond_destroy(&g_l256_pool.cv_start);
        pthread_mutex_destroy(&g_l256_pool.mtx);
        return -3;
    }
    g_l256_pool.max_workers = max_workers;
    for (int i = 0; i < max_workers; i++) {
        g_l256_args[i].pool = &g_l256_pool;
        g_l256_args[i].tid  = i;
        if (pthread_create(&g_l256_tids[i], NULL, lc_l256_worker, &g_l256_args[i]) != 0) {
            pthread_mutex_lock(&g_l256_pool.mtx);
            g_l256_pool.stop = true;
            g_l256_pool.epoch++;
            pthread_cond_broadcast(&g_l256_pool.cv_start);
            pthread_mutex_unlock(&g_l256_pool.mtx);
            for (int j = 0; j < i; j++) pthread_join(g_l256_tids[j], NULL);
            pthread_cond_destroy(&g_l256_pool.cv_done);
            pthread_cond_destroy(&g_l256_pool.cv_start);
            pthread_mutex_destroy(&g_l256_pool.mtx);
            return -4;
        }
    }
    g_l256_pool.initialized = true;
    return max_workers;
}

LC_API void lc_v2_l256_mt_shutdown(void) {
    if (!g_l256_pool.initialized) return;
    pthread_mutex_lock(&g_l256_pool.mtx);
    g_l256_pool.stop = true;
    g_l256_pool.epoch++;
    pthread_cond_broadcast(&g_l256_pool.cv_start);
    pthread_mutex_unlock(&g_l256_pool.mtx);
    for (int i = 0; i < g_l256_pool.max_workers; i++) pthread_join(g_l256_tids[i], NULL);
    pthread_cond_destroy(&g_l256_pool.cv_done);
    pthread_cond_destroy(&g_l256_pool.cv_start);
    pthread_mutex_destroy(&g_l256_pool.mtx);
    g_l256_pool.initialized = false;
}

static int lc_l256_validate(int n_samples, int dim, int depth, int *num_threads) {
    if (n_samples <= 0 || dim <= 0 || (dim & 7) != 0 || dim > LC_L256_MAX_DIM || depth <= 0) {
        return -1;
    }
    if (*num_threads > 1) {
        if (!g_l256_pool.initialized && lc_v2_l256_mt_init(0) < 0) return -2;
        if (*num_threads > g_l256_pool.max_workers) *num_threads = g_l256_pool.max_workers;
        if (*num_threads > n_samples) *num_threads = n_samples;
    }
    return 0;
}

LC_API int lc_v2_l256_forward(lc_c16_t *s, lc_c16_t *v, int n_samples, int dim, int depth,
                              const lc_logc16_t *w, const lc_c16_t *b,
                              const uint8_t *modes, const int32_t *shifts, int num_threads) {
    int rc = lc_l256_validate(n_samples, dim, depth, &num_threads);
    if (rc != 0) return rc;
    g_l256_pool.up_s      = s;
    g_l256_pool.up_v      = v;
    g_l256_pool.w         = w;
    g_l256_pool.b         = b;
    g_l256_pool.modes     = modes;
    g_l256_pool.shifts    = shifts;
    g_l256_pool.n_samples = n_samples;
    g_l256_pool.dim       = dim;
    g_l256_pool.depth     = depth;
    lc_l256_dispatch(LC_L256_JOB_FORWARD, 0, num_threads);
    return 0;
}

LC_API int lc_v2_l256_inverse(lc_c16_t *s, lc_c16_t *v, int n_samples, int dim, int depth,
                              const lc_logc16_t *w, const lc_c16_t *b,
                              const uint8_t *modes, const int32_t *shifts, int num_threads) {
    int rc = lc_l256_validate(n_samples, dim, depth, &num_threads);
    if (rc != 0) return rc;
    g_l256_pool.up_s      = s;
    g_l256_pool.up_v      = v;
    g_l256_pool.w         = w;
    g_l256_pool.b         = b;
    g_l256_pool.modes     = modes;
    g_l256_pool.shifts    = shifts;
    g_l256_pool.n_samples = n_samples;
    g_l256_pool.dim       = dim;
    g_l256_pool.depth     = depth;
    lc_l256_dispatch(LC_L256_JOB_INVERSE, 0, num_threads);
    return 0;
}

LC_API int lc_v2_l256_train_epoch(const lc_c16_t *in_s,  const lc_c16_t *in_v,
                                  const lc_c16_t *tgt_s, const lc_c16_t *tgt_v,
                                  lc_c16_t *work_up_s,   lc_c16_t *work_up_v,
                                  lc_c16_t *work_dn_s,   lc_c16_t *work_dn_v,
                                  int n_samples, int dim, int depth,
                                  lc_logc16_t *w, lc_c16_t *b,
                                  int32_t *err_fb_re, int32_t *err_fb_im,
                                  const uint8_t *modes, const int32_t *shifts,
                                  uint8_t lr_shift, int16_t max_rho, int learn_bias,
                                  int num_threads, double *out_rmse) {
    int rc = lc_l256_validate(n_samples, dim, depth, &num_threads);
    if (rc != 0) return rc;

    size_t total_elems = (size_t)n_samples * (size_t)dim;
    size_t state_bytes = total_elems * sizeof(lc_c16_t);
    memcpy(work_up_s, in_s,  state_bytes);
    memcpy(work_up_v, in_v,  state_bytes);
    memcpy(work_dn_s, tgt_s, state_bytes);
    memcpy(work_dn_v, tgt_v, state_bytes);

    g_l256_pool.up_s      = work_up_s;
    g_l256_pool.up_v      = work_up_v;
    g_l256_pool.dn_s      = work_dn_s;
    g_l256_pool.dn_v      = work_dn_v;
    g_l256_pool.w         = w;
    g_l256_pool.b         = b;
    g_l256_pool.modes     = modes;
    g_l256_pool.shifts    = shifts;
    g_l256_pool.n_samples = n_samples;
    g_l256_pool.dim       = dim;
    g_l256_pool.depth     = depth;

    /* 1. 順流 Δ^depth (全 depth 層の中間活性化を一切保持しない: O(1) メモリ) */
    lc_l256_dispatch(LC_L256_JOB_FORWARD, 0, num_threads);

    if (out_rmse != NULL) {
        double se = 0.0;
        for (size_t idx = 0; idx < total_elems; idx++) {
            double e0 = (double)((int32_t)work_up_s[idx].re - (int32_t)work_dn_s[idx].re);
            double e1 = (double)((int32_t)work_up_s[idx].im - (int32_t)work_dn_s[idx].im);
            double e2 = (double)((int32_t)work_up_v[idx].re - (int32_t)work_dn_v[idx].re);
            double e3 = (double)((int32_t)work_up_v[idx].im - (int32_t)work_dn_v[idx].im);
            se += e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
        }
        *out_rmse = lc_l256_sqrt(se / (double)(4.0 * (double)total_elems));
    }

    /* 2. 複線逆走 μ_{depth-1} -> μ_0: 各層で 0-bit 逆算 + 局所 2-射リダクション + 重み更新 */
    int n_red = (num_threads <= 1) ? 1 : num_threads;
    for (int L = depth - 1; L >= 0; L--) {
        lc_l256_dispatch(LC_L256_JOB_REV_ACC, L, num_threads);

        size_t loff = (size_t)L * (size_t)dim;
        for (int d = 0; d < dim; d++) {
            int64_t sgre = 0, sgim = 0, su2 = 0, sdre = 0, sdim = 0;
            for (int t = 0; t < n_red; t++) {
                sgre += g_l256_acc[t].sum_gre[d];
                sgim += g_l256_acc[t].sum_gim[d];
                su2  += g_l256_acc[t].sum_u2[d];
                sdre += g_l256_acc[t].sum_dre[d];
                sdim += g_l256_acc[t].sum_dim[d];
            }

            int64_t denom = su2 + (int64_t)n_samples * 262144LL;
            int32_t dw_re = lc_sym_rshift64((sgre * (int64_t)LC_Q14_ONE) / denom, (int32_t)lr_shift);
            int32_t dw_im = lc_sym_rshift64((sgim * (int64_t)LC_Q14_ONE) / denom, (int32_t)lr_shift);

            lc_c16_t wc   = lc_log_to_cart(w[loff + (size_t)d].rho, w[loff + (size_t)d].theta);
            int32_t  t_re = lc_l256_clamp32((int32_t)wc.re + err_fb_re[loff + (size_t)d] + dw_re, -16384, 16384);
            int32_t  t_im = lc_l256_clamp32((int32_t)wc.im + err_fb_im[loff + (size_t)d] + dw_im, -16384, 16384);

            lc_logc16_t w_new = lc_cart_to_log((int16_t)t_re, (int16_t)t_im);
            if (w_new.rho != LC_LOG_ZERO && w_new.rho > max_rho) {
                w_new.rho = max_rho;
            }
            lc_c16_t wc_q = lc_log_to_cart(w_new.rho, w_new.theta);

            w[loff + (size_t)d]         = w_new;
            err_fb_re[loff + (size_t)d] = lc_l256_clamp32(t_re - (int32_t)wc_q.re, -256, 256);
            err_fb_im[loff + (size_t)d] = lc_l256_clamp32(t_im - (int32_t)wc_q.im, -256, 256);

            if (learn_bias != 0) {
                int32_t db_re = lc_sym_rshift64(sdre / (int64_t)n_samples, (int32_t)lr_shift);
                int32_t db_im = lc_sym_rshift64(sdim / (int64_t)n_samples, (int32_t)lr_shift);
                b[loff + (size_t)d].re = (int16_t)lc_l256_clamp32((int32_t)b[loff + (size_t)d].re + db_re, -2048, 2048);
                b[loff + (size_t)d].im = (int16_t)lc_l256_clamp32((int32_t)b[loff + (size_t)d].im + db_im, -2048, 2048);
            }
        }
    }
    return 0;
}