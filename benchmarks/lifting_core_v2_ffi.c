/* lifting_core_v2_ffi.c — C ABI Shim for liblifting_core_v2.dylib
 *   Exports: Log-Polar FHRR, Pure-Integer Dual-Chart (Cart<->Log), 64-Layer
 *   Closed-Loop Ladder, Higher-Category H1 3-Morphism Rewrite, Associator,
 *   Dynamic Currying, 1-UIP CDCL & 128B-Aligned Batched Work-Stealing Pool */
#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#define LIFTING_CORE_V2_IMPLEMENTATION
#include "lifting_core_v2.h"

#if defined(__GNUC__) || defined(__clang__)
#define LC_API      __attribute__((visibility("default")))
#define LC_ALIGN128 __attribute__((aligned(128)))
#else
#define LC_API
#define LC_ALIGN128
#endif

#define LC_FFI_MAX_NETS    32
#define LC_FFI_MAX_THREADS 64

static lc_currying_net_t g_nets[LC_FFI_MAX_NETS];
static bool              g_net_used[LC_FFI_MAX_NETS];
static lc_cdcl_graph_t   g_cdcl;
static uint8_t           g_cur_tt = 0;

static bool lc_v2_tt_eval(bool a, bool b, bool c) {
    int idx = (a ? 1 : 0) | (b ? 2 : 0) | (c ? 4 : 0);
    return ((g_cur_tt >> idx) & 1u) != 0u;
}

/* ============================================================================
 * 128-byte 整列・全 AoSoA-8 カーネル共通バッチ／Guided ワークスティーリングプール
 * ============================================================================ */
typedef enum {
    LC_JOB_Q14_STEP    = 0,
    LC_JOB_LOG_BIND    = 1,
    LC_JOB_LOG_DUAL    = 2,
    LC_JOB_CART_TO_LOG = 3,
    LC_JOB_HOLO_2MORPH = 4
} lc_mt_job_t;

typedef struct {
    void        *buf;
    int          n_blocks;
    int          total_chunks;
    int          num_threads;
    int          batch_chunks;  /* >0: 固定バッチ (推奨 32), 0: Guided 動的縮小 */
    lc_mt_job_t  job;
    uint8_t      lr_shift;
    bool         apply_twist;

    LC_ALIGN128 int      next_chunk;
    LC_ALIGN128 uint64_t holo_l1_acc;

    LC_ALIGN128 pthread_mutex_t mtx;
    pthread_cond_t  cv_start;
    pthread_cond_t  cv_done;
    uint64_t        epoch;
    int             max_workers;
    int             done_workers;
    bool            initialized;
    bool            stop;
} lc_v2_mt_pool_t;

typedef struct {
    lc_v2_mt_pool_t *pool;
    int              tid;
} lc_v2_mt_arg_t;

static lc_v2_mt_pool_t g_pool;
static lc_v2_mt_arg_t  g_args[LC_FFI_MAX_THREADS];
static pthread_t       g_tids[LC_FFI_MAX_THREADS];

static inline uint64_t lc_v2_exec_slice(lc_mt_job_t job, void *buf, int b_start, int count,
                                        uint8_t lr_shift, bool apply_twist) {
    switch (job) {
        case LC_JOB_Q14_STEP:
            lc_aosoa8_step_l1_blocked(&((lc_block8_t *)buf)[b_start], count, lr_shift);
            return 0u;
        case LC_JOB_LOG_BIND:
            lc_log_aosoa8_bind_only(&((lc_log_block8_t *)buf)[b_start], count);
            return 0u;
        case LC_JOB_LOG_DUAL:
            lc_log_aosoa8_dual_step(&((lc_log_block8_t *)buf)[b_start], count);
            return 0u;
        case LC_JOB_CART_TO_LOG:
            lc_chart_aosoa8_cart_to_log(&((lc_chart_block8_t *)buf)[b_start], count);
            return 0u;
        case LC_JOB_HOLO_2MORPH:
            return lc_holo_aosoa8_eval_2morph(&((lc_holo_block8_t *)buf)[b_start], count, apply_twist);
    }
    return 0u;
}

static void *lc_v2_mt_worker(void *arg) {
    lc_v2_mt_arg_t  *wa   = (lc_v2_mt_arg_t *)arg;
    lc_v2_mt_pool_t *pool = wa->pool;
    int              tid  = wa->tid;
    uint64_t         seen = 0;

    for (;;) {
        pthread_mutex_lock(&pool->mtx);
        while (!pool->stop && pool->epoch == seen) {
            pthread_cond_wait(&pool->cv_start, &pool->mtx);
        }
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mtx);
            break;
        }
        seen             = pool->epoch;
        void       *buf  = pool->buf;
        int         nblk = pool->n_blocks;
        int         totc = pool->total_chunks;
        int         nthr = pool->num_threads;
        int         bsz  = pool->batch_chunks;
        lc_mt_job_t job  = pool->job;
        uint8_t     lr   = pool->lr_shift;
        bool        atw  = pool->apply_twist;
        pthread_mutex_unlock(&pool->mtx);

        uint64_t local_l1 = 0u;
        if (buf != NULL && tid < nthr) {
            if (bsz > 0) {
                for (;;) {
                    int sc = __atomic_fetch_add(&pool->next_chunk, bsz, __ATOMIC_RELAXED);
                    if (sc >= totc) break;
                    int ec = sc + bsz;
                    if (ec > totc) ec = totc;
                    int bs = sc * LC_L1_BLOCK_GROUPS;
                    int be = ec * LC_L1_BLOCK_GROUPS;
                    if (be > nblk) be = nblk;
                    if (be > bs) {
                        local_l1 += lc_v2_exec_slice(job, buf, bs, be - bs, lr, atw);
                    }
                }
            } else {
                int cur = __atomic_load_n(&pool->next_chunk, __ATOMIC_RELAXED);
                for (;;) {
                    int rem = totc - cur;
                    if (rem <= 0) break;
                    int step = rem / (nthr * 2);
                    if (step < 4)   step = 4;
                    if (step > rem) step = rem;
                    if (__atomic_compare_exchange_n(&pool->next_chunk, &cur, cur + step,
                                                    true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                        int bs = cur * LC_L1_BLOCK_GROUPS;
                        int be = (cur + step) * LC_L1_BLOCK_GROUPS;
                        if (be > nblk) be = nblk;
                        if (be > bs) {
                            local_l1 += lc_v2_exec_slice(job, buf, bs, be - bs, lr, atw);
                        }
                        cur = __atomic_load_n(&pool->next_chunk, __ATOMIC_RELAXED);
                    }
                }
            }
        }

        if (job == LC_JOB_HOLO_2MORPH && local_l1 != 0u) {
            __atomic_fetch_add(&pool->holo_l1_acc, local_l1, __ATOMIC_RELAXED);
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

/* --- 初期化 & ABI / 環境情報 --- */
LC_API int    lc_v2_ffi_init(void)               { return lc_v2_init() ? 0 : -1; }
LC_API size_t lc_v2_ffi_sizeof_cell(void)        { return sizeof(LiftingCell); }
LC_API size_t lc_v2_ffi_sizeof_cell_log(void)    { return sizeof(LiftingCellLog); }
LC_API size_t lc_v2_ffi_sizeof_block8(void)      { return sizeof(lc_block8_t); }
LC_API size_t lc_v2_ffi_sizeof_log_block8(void)  { return sizeof(lc_log_block8_t); }
LC_API size_t lc_v2_ffi_sizeof_chart_block8(void){ return sizeof(lc_chart_block8_t); }
LC_API size_t lc_v2_ffi_sizeof_holo_block8(void) { return sizeof(lc_holo_block8_t); }
LC_API int    lc_v2_ffi_has_neon(void)           { return LC_HAS_ARM_NEON; }

LC_API int lc_v2_ffi_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    int  c = (n > 0) ? (int)n : 4;
    return (c > LC_FFI_MAX_THREADS) ? LC_FFI_MAX_THREADS : c;
}

/* --- マルチコアプール制御 --- */
LC_API int lc_v2_ffi_mt_init(int max_workers) {
    if (g_pool.initialized) return g_pool.max_workers;
    if (max_workers <= 0) max_workers = lc_v2_ffi_cpu_count();
    if (max_workers > LC_FFI_MAX_THREADS) max_workers = LC_FFI_MAX_THREADS;

    memset(&g_pool, 0, sizeof(g_pool));
    if (pthread_mutex_init(&g_pool.mtx, NULL) != 0) return -1;
    if (pthread_cond_init(&g_pool.cv_start, NULL) != 0) {
        pthread_mutex_destroy(&g_pool.mtx);
        return -2;
    }
    if (pthread_cond_init(&g_pool.cv_done, NULL) != 0) {
        pthread_cond_destroy(&g_pool.cv_start);
        pthread_mutex_destroy(&g_pool.mtx);
        return -3;
    }

    g_pool.max_workers = max_workers;
    for (int i = 0; i < max_workers; i++) {
        g_args[i].pool = &g_pool;
        g_args[i].tid  = i;
        if (pthread_create(&g_tids[i], NULL, lc_v2_mt_worker, &g_args[i]) != 0) {
            pthread_mutex_lock(&g_pool.mtx);
            g_pool.stop = true;
            g_pool.epoch++;
            pthread_cond_broadcast(&g_pool.cv_start);
            pthread_mutex_unlock(&g_pool.mtx);
            for (int j = 0; j < i; j++) pthread_join(g_tids[j], NULL);
            pthread_cond_destroy(&g_pool.cv_done);
            pthread_cond_destroy(&g_pool.cv_start);
            pthread_mutex_destroy(&g_pool.mtx);
            return -4;
        }
    }
    g_pool.initialized = true;
    return max_workers;
}

LC_API void lc_v2_ffi_mt_shutdown(void) {
    if (!g_pool.initialized) return;
    pthread_mutex_lock(&g_pool.mtx);
    g_pool.stop = true;
    g_pool.epoch++;
    pthread_cond_broadcast(&g_pool.cv_start);
    pthread_mutex_unlock(&g_pool.mtx);

    for (int i = 0; i < g_pool.max_workers; i++) pthread_join(g_tids[i], NULL);
    pthread_cond_destroy(&g_pool.cv_done);
    pthread_cond_destroy(&g_pool.cv_start);
    pthread_mutex_destroy(&g_pool.mtx);
    g_pool.initialized = false;
}

static int lc_v2_dispatch_mt(lc_mt_job_t job, void *buf, int n_blocks, int iters,
                             uint8_t lr_shift, bool apply_twist,
                             int num_threads, int batch_chunks, uint64_t *out_l1) {
    if (buf == NULL || n_blocks <= 0 || iters <= 0) return -1;
    if (num_threads <= 1) {
        uint64_t last_l1 = 0u;
        for (int it = 0; it < iters; it++) {
            last_l1 = lc_v2_exec_slice(job, buf, 0, n_blocks, lr_shift, apply_twist);
        }
        if (out_l1 != NULL) *out_l1 = last_l1;
        return 0;
    }
    if (!g_pool.initialized) {
        if (lc_v2_ffi_mt_init(0) < 0) return -2;
    }
    if (num_threads > g_pool.max_workers) num_threads = g_pool.max_workers;
    if (batch_chunks < 0) batch_chunks = 32;

    int total_chunks = (n_blocks + LC_L1_BLOCK_GROUPS - 1) / LC_L1_BLOCK_GROUPS;
    uint64_t last_l1 = 0u;

    for (int it = 0; it < iters; it++) {
        pthread_mutex_lock(&g_pool.mtx);
        g_pool.buf          = buf;
        g_pool.n_blocks     = n_blocks;
        g_pool.total_chunks = total_chunks;
        g_pool.num_threads  = num_threads;
        g_pool.batch_chunks = batch_chunks;
        g_pool.job          = job;
        g_pool.lr_shift     = lr_shift;
        g_pool.apply_twist  = apply_twist;
        g_pool.done_workers = 0;
        __atomic_store_n(&g_pool.next_chunk,  0,  __ATOMIC_RELEASE);
        __atomic_store_n(&g_pool.holo_l1_acc, 0u, __ATOMIC_RELEASE);
        g_pool.epoch++;
        pthread_cond_broadcast(&g_pool.cv_start);
        while (g_pool.done_workers < g_pool.max_workers) {
            pthread_cond_wait(&g_pool.cv_done, &g_pool.mtx);
        }
        last_l1 = __atomic_load_n(&g_pool.holo_l1_acc, __ATOMIC_ACQUIRE);
        pthread_mutex_unlock(&g_pool.mtx);
    }
    if (out_l1 != NULL) *out_l1 = last_l1;
    return 0;
}

/* --- Layer 1: 純整数双対チャート (Cart<->Log) & 64 層閉ループ梯子 --- */
LC_API void lc_v2_ffi_dual_ladder_forward(lc_c16_t *s, lc_c16_t *v, int n_pairs,
                                          const lc_logc16_t *weights, int depth) {
    lc_dual_ladder_forward(s, v, n_pairs, weights, depth);
}

LC_API void lc_v2_ffi_dual_ladder_inverse(lc_c16_t *s, lc_c16_t *v, int n_pairs,
                                          const lc_logc16_t *weights, int depth) {
    lc_dual_ladder_inverse(s, v, n_pairs, weights, depth);
}

LC_API int lc_v2_ffi_chart_cart_to_log_mt(lc_chart_block8_t *blocks, int n_blocks,
                                          int iters, int num_threads, int batch_chunks) {
    return lc_v2_dispatch_mt(LC_JOB_CART_TO_LOG, blocks, n_blocks, iters,
                             0, false, num_threads, batch_chunks, NULL);
}

/* --- Layer 2: 高次圏 2-射ホロノミー 3-射書き換え・Associator・一括検出 --- */
LC_API int lc_v2_ffi_diamond_solve_h1(int16_t a_rho,  uint16_t a_theta,
                                      int16_t f1_rho, uint16_t f1_theta,
                                      int16_t f2_rho, uint16_t f2_theta,
                                      int16_t g2_rho, uint16_t g2_theta,
                                      int16_t tw_rho, uint16_t tw_theta,
                                      int *out_spawned, int *out_steps, int32_t *out_static_l1,
                                      int16_t *out_tw_rho, uint16_t *out_tw_theta) {
    lc_diamond_t dm;
    lc_diamond_init(&dm,
                    (lc_logc16_t){a_rho,  a_theta},
                    (lc_logc16_t){f1_rho, f1_theta},
                    (lc_logc16_t){f2_rho, f2_theta},
                    (lc_logc16_t){g2_rho, g2_theta});
    bool spawned = false;
    int  steps   = 0;
    int32_t sl1  = 0;
    bool ok = lc_diamond_solve_h1(&dm, (lc_logc16_t){tw_rho, tw_theta},
                                  &spawned, &steps, &sl1);
    if (out_spawned   != NULL) *out_spawned   = spawned ? 1 : 0;
    if (out_steps     != NULL) *out_steps     = steps;
    if (out_static_l1 != NULL) *out_static_l1 = sl1;
    if (out_tw_rho    != NULL) *out_tw_rho    = dm.p2_tail->w.rho;
    if (out_tw_theta  != NULL) *out_tw_theta  = dm.p2_tail->w.theta;
    return ok ? 1 : 0;
}

LC_API int lc_v2_ffi_assoc_resolve(int16_t a_rho, uint16_t a_theta,
                                   int16_t b_rho, uint16_t b_theta,
                                   int16_t c_rho, uint16_t c_theta,
                                   int16_t *out_rho, uint16_t *out_theta,
                                   int32_t *out_alpha_drho, int16_t *out_alpha_dtheta,
                                   int *out_alpha_bot, int *out_reassoc) {
    lc_logc16_t val;
    lc_2morph_t alpha;
    bool reassoc = false;
    bool ok = lc_assoc_resolve((lc_logc16_t){a_rho, a_theta},
                               (lc_logc16_t){b_rho, b_theta},
                               (lc_logc16_t){c_rho, c_theta},
                               &val, &alpha, &reassoc);
    if (out_rho          != NULL) *out_rho          = val.rho;
    if (out_theta        != NULL) *out_theta        = val.theta;
    if (out_alpha_drho   != NULL) *out_alpha_drho   = alpha.drho;
    if (out_alpha_dtheta != NULL) *out_alpha_dtheta = alpha.dtheta;
    if (out_alpha_bot    != NULL) *out_alpha_bot    = alpha.has_bot ? 1 : 0;
    if (out_reassoc      != NULL) *out_reassoc      = reassoc ? 1 : 0;
    return ok ? 1 : 0;
}

LC_API int lc_v2_ffi_holo_eval_mt(lc_holo_block8_t *blocks, int n_blocks,
                                  int iters, int apply_twist,
                                  int num_threads, int batch_chunks,
                                  uint64_t *out_phase_l1) {
    return lc_v2_dispatch_mt(LC_JOB_HOLO_2MORPH, blocks, n_blocks, iters,
                             0, apply_twist != 0, num_threads, batch_chunks, out_phase_l1);
}

/* --- Layer 6: Log-Polar FHRR Bind / Dual-Chart / Q1.14 3-pass マルチコア API --- */
LC_API int lc_v2_ffi_log_bind_mt(lc_log_block8_t *blocks, int n_blocks,
                                 int iters, int num_threads, int batch_chunks) {
    return lc_v2_dispatch_mt(LC_JOB_LOG_BIND, blocks, n_blocks, iters,
                             0, false, num_threads, batch_chunks, NULL);
}

LC_API int lc_v2_ffi_log_dual_mt(lc_log_block8_t *blocks, int n_blocks,
                                 int iters, int num_threads, int batch_chunks) {
    return lc_v2_dispatch_mt(LC_JOB_LOG_DUAL, blocks, n_blocks, iters,
                             0, false, num_threads, batch_chunks, NULL);
}

LC_API int lc_v2_ffi_q14_step_mt(lc_block8_t *blocks, int n_blocks,
                                 int iters, uint8_t lr_shift,
                                 int num_threads, int batch_chunks) {
    return lc_v2_dispatch_mt(LC_JOB_Q14_STEP, blocks, n_blocks, iters,
                             lr_shift, false, num_threads, batch_chunks, NULL);
}

/* 大規模ベンチ向け C 側高速決定論的フィラー (Log-Polar FHRR) */
LC_API void lc_v2_ffi_fill_log_blocks(lc_log_block8_t *blocks, int n_blocks, uint32_t seed) {
    uint32_t rng = seed;
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
        for (int k = 0; k < 8; k++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            uint32_t r0 = rng;
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            uint32_t r1 = rng;
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            uint32_t r2 = rng;

            int idx = bi * 8 + k;
            b->s_rho[k]   = ((idx % 17) == 0) ? LC_LOG_ZERO : (int16_t)((int32_t)(r0 & 0x7FFu) - 1024);
            b->g_rho[k]   = ((idx % 23) == 0) ? LC_LOG_ZERO : (int16_t)((int32_t)(r1 & 0x3FFu) - 512);
            b->w_rho[k]   = (int16_t)((int32_t)(r2 & 0x3FFu) - 512);
            b->s_theta[k] = (uint16_t)(r0 >> 16);
            b->g_theta[k] = (uint16_t)(r1 >> 16);
            b->w_theta[k] = (uint16_t)(r2 >> 16);
            b->v_re[k]    = (int16_t)((idx * 257 - 16384) & 0xFFFF);
            b->v_im[k]    = (int16_t)((idx * 131 +  8192) & 0xFFFF);
        }
    }
}

/* --- Layer 3 & 4 (v1 互換): 動的カリー化合成 & 1-UIP CDCL --- */
LC_API void lc_v2_ffi_nets_reset(void) {
    memset(g_net_used, 0, sizeof(g_net_used));
}

LC_API int lc_v2_ffi_net_synthesize_tt(uint8_t truth_table, int max_epochs,
                                       int *out_num_cells, int *out_spawned) {
    int slot = -1;
    for (int i = 0; i < LC_FFI_MAX_NETS; i++) if (!g_net_used[i]) { slot = i; break; }
    if (slot < 0) return -1;
    g_cur_tt = truth_table;
    lc_net_init(&g_nets[slot], 3);
    if (!lc_net_synthesize(&g_nets[slot], lc_v2_tt_eval, max_epochs)) return -2;
    g_net_used[slot] = true;
    if (out_num_cells != NULL) *out_num_cells = g_nets[slot].num_cells;
    if (out_spawned   != NULL) *out_spawned   = g_nets[slot].spawned_count;
    return slot;
}

LC_API void lc_v2_ffi_cdcl_init(void) { lc_cdcl_init(&g_cdcl); }
LC_API int  lc_v2_ffi_cdcl_add_var(int8_t obs_val) {
    return (g_cdcl.num_vars >= LC_CDCL_MAX_VARS) ? -1 : lc_cdcl_add_var(&g_cdcl, obs_val);
}
LC_API int  lc_v2_ffi_cdcl_add_gate(int net_id, int in0, int in1, int in2, int out_v) {
    if (net_id < 0 || net_id >= LC_FFI_MAX_NETS || !g_net_used[net_id]) return -1;
    if (g_cdcl.num_gates >= LC_CDCL_MAX_GATES) return -2;
    lc_cdcl_add_gate(&g_cdcl, &g_nets[net_id], in0, in1, in2, out_v);
    return 0;
}
LC_API int  lc_v2_ffi_cdcl_solve(int max_conflicts, uint8_t *out_bits, int out_cap,
                                 lc_cdcl_stats_t *out_stats) {
    bool sol[LC_CDCL_MAX_VARS];
    bool ok = lc_cdcl_solve(&g_cdcl, max_conflicts, sol, out_stats);
    if (ok && out_bits != NULL) {
        int n = (g_cdcl.num_vars < out_cap) ? g_cdcl.num_vars : out_cap;
        for (int i = 0; i < n; i++) out_bits[i] = sol[i] ? 1u : 0u;
    }
    return ok ? 1 : 0;
}