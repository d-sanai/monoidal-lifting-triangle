/* ============================================================================
 * lifting_core_v2.h — Zero-Heap C99 Single-Header Reversible Lifting Core v2
 *   [Layer 0] 64-byte LiftingCell (Q1.14) & LiftingCellLog (Log-Polar FHRR)
 *   [Layer 1] Pure-Integer Dual-Chart (Log <-> Cart, 7.5 KB L1D LUT, no -lm)
 *             & Arbitrary-Depth Closed-Loop Reversible Lifting Ladder (μ∘Δ=id)
 *   [Layer 2] Higher-Category 2-Morphism (H1 Holonomy), 3-Morphism Autonomous
 *             TwistedSection Rewrite, Associator Recovery & Frobenius ZX-Spider
 *   [Layer 3] Dynamic Currying Synthesis & Ref-Counted Shared DAG Pool
 *   [Layer 4] Minimal Sufficient Reason + 1-UIP Curried CDCL Abduction Engine
 *   [Layer 5] Continuous-Phase Soft-Decision ECC (Hamming / QC-LDPC, no -lm)
 *   [Layer 6] ARM64 NEON AoSoA-8 Kernels (Q1.14 / FHRR / Dual-Chart / Holonomy)
 *
 * Usage (in ONE .c file):
 *   #define LIFTING_CORE_V2_IMPLEMENTATION
 *   #include "lifting_core_v2.h"
 * ============================================================================ */
#ifndef LIFTING_CORE_V2_H
#define LIFTING_CORE_V2_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define LC_HAS_ARM_NEON 1
#else
#define LC_HAS_ARM_NEON 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LC_ALIGN64  __attribute__((aligned(64)))
#define LC_NOINLINE __attribute__((noinline))
#else
#define LC_ALIGN64
#define LC_NOINLINE
#endif

/* ----------------------------------------------------------------------------
 * Layer 0: 基本定数・直交/対数極座標型・64-byte 均一セル 2 種・2-射
 * ---------------------------------------------------------------------------- */
#define LC_Q14_ONE    16384
#define LC_Q14_HALF    8192
#define LC_Q14_NOT   (-32768)

#define LC_LOG_SCALE    256          /* Q7.8: +256 = x2.0, 0 = x1.0, -256 = x0.5 */
#define LC_LOG_ZERO  (-32768)        /* INT16_MIN: 対数空間の吸収元 ⊥ */

#define LC_TRIG_BITS    10
#define LC_TRIG_SIZE    (1 << LC_TRIG_BITS) /* 1024 点 */
#define LC_TRIG_SHIFT   (16 - LC_TRIG_BITS) /* 6 */
#define LC_LOG2_BITS    9
#define LC_LOG2_SIZE    (1 << LC_LOG2_BITS) /* 512 点 */
#define LC_ATAN_BITS    10
#define LC_ATAN_SIZE    (1 << LC_ATAN_BITS) /* 1024 点 */

#define LC_LLR_SCALE    256
#define LC_LLR_MAX      2048

#define LC_FLAG_ALIVE         (1u << 0)
#define LC_FLAG_ACTIVE        (1u << 1)
#define LC_FLAG_CONFLICT_TERM (1u << 2)
#define LC_FLAG_CURRIED_TWIST (1u << 3)
#define LC_FLAG_LOCKED        (1u << 4)

typedef struct {
    int16_t re;
    int16_t im;
} lc_c16_t;

typedef struct {
    int16_t  rho;    /* Q7.8 log2(|z| / 2^14), INT16_MIN = ⊥ */
    uint16_t theta;  /* Z / 2^16 Z 円環位相 */
} lc_logc16_t;

typedef struct {
    int32_t drho;    /* 対数強度ズレ (rho1 - rho2) */
    int16_t dtheta;  /* 符号付き循環位相差 ∈ [-32768, 32767] */
    bool    has_bot; /* 片方のみ ⊥ */
} lc_2morph_t;

typedef enum {
    LC_MODE_EXACT = 0,
    LC_MODE_LEARN = 1,
    LC_MODE_EVAL  = 2
} lc_exec_mode_t;

/* 64-byte 直交 Q1.14 LiftingCell (v1 互換) */
typedef struct LiftingCell {
    struct LiftingCell *in_s;
    struct LiftingCell *in_g;
    struct LiftingCell *in_v;

    lc_c16_t w;
    lc_c16_t s_eff;
    lc_c16_t p;
    lc_c16_t r;
    lc_c16_t td_r;
    lc_c16_t restored;

    int32_t adj_s;
    int32_t meta_energy;

    uint16_t ref_count;
    uint16_t var_id;
    uint8_t  idle_counter;
    uint8_t  var_mask;
    uint8_t  order;
    uint8_t  flags;
} LiftingCell;

/* 64-byte 複素対数極座標 FHRR LiftingCellLog (v2 拡張・無肥大) */
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

typedef char lc_assert_c16_4b[(sizeof(lc_c16_t) == 4) ? 1 : -1];
typedef char lc_assert_logc16_4b[(sizeof(lc_logc16_t) == 4) ? 1 : -1];
#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFULL
typedef char lc_assert_cell_64b[(sizeof(LiftingCell) == 64) ? 1 : -1];
typedef char lc_assert_cell_log_64b[(sizeof(LiftingCellLog) == 64) ? 1 : -1];
#endif

/* 基本インライン演算 (直交 & Log-Polar) */
static inline int32_t lc_iabs32(int32_t x) { return (x >= 0) ? x : -x; }

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

static inline int16_t lc_quantize_q14(int64_t acc) {
    int64_t rounded = (acc >= 0) ? ((acc + LC_Q14_HALF) >> 14)
                                 : -((-acc + LC_Q14_HALF) >> 14);
    if (rounded >  32767) return  32767;
    if (rounded < -32768) return -32768;
    return (int16_t)rounded;
}

static inline int16_t lc_quantize_q14_s32(int32_t acc) {
    int32_t mag = (acc >= 0) ? acc : -acc;
    int32_t q   = (mag + LC_Q14_HALF) >> 14;
    if (acc < 0) q = -q;
    if (q >  32767) return  32767;
    if (q < -32768) return -32768;
    return (int16_t)q;
}

static inline lc_c16_t lc_c16_mul_q14(lc_c16_t a, lc_c16_t b) {
    int64_t re_acc = (int64_t)a.re * b.re - (int64_t)a.im * b.im;
    int64_t im_acc = (int64_t)a.re * b.im + (int64_t)a.im * b.re;
    lc_c16_t out = { lc_quantize_q14(re_acc), lc_quantize_q14(im_acc) };
    return out;
}

static inline lc_c16_t lc_c16_sub_mod(lc_c16_t a, lc_c16_t b) {
    lc_c16_t out = {
        (int16_t)((uint16_t)a.re - (uint16_t)b.re),
        (int16_t)((uint16_t)a.im - (uint16_t)b.im)
    };
    return out;
}

static inline lc_c16_t lc_c16_add_mod(lc_c16_t a, lc_c16_t b) {
    lc_c16_t out = {
        (int16_t)((uint16_t)a.re + (uint16_t)b.re),
        (int16_t)((uint16_t)a.im + (uint16_t)b.im)
    };
    return out;
}

static inline int32_t lc_c16_l1(lc_c16_t a) {
    return lc_iabs32((int32_t)a.re) + lc_iabs32((int32_t)a.im);
}

static inline bool lc_c16_eq(lc_c16_t a, lc_c16_t b) {
    return (a.re == b.re) && (a.im == b.im);
}

static inline int32_t lc_sym_rshift64(int64_t val, int32_t shift) {
    return (val >= 0) ? (int32_t)(val >> shift) : -(int32_t)((-val) >> shift);
}

static inline uint8_t lc_popcount8(uint8_t x) {
    uint8_t c = 0;
    while (x != 0u) { c += (uint8_t)(x & 1u); x >>= 1; }
    return c;
}

static inline int16_t lc_sat_add_s16(int16_t a, int16_t b) {
    int32_t s = (int32_t)a + (int32_t)b;
    if (s >  32767) return  32767;
    if (s < -32768) return -32768;
    return (int16_t)s;
}

static inline int16_t lc_sat_sub_s16(int16_t a, int16_t b) {
    int32_t s = (int32_t)a - (int32_t)b;
    if (s >  32767) return  32767;
    if (s < -32768) return -32768;
    return (int16_t)s;
}

static inline int16_t lc_log_add_rho(int16_t a, int16_t b, bool *saturated) {
    if (a == LC_LOG_ZERO || b == LC_LOG_ZERO) return LC_LOG_ZERO;
    int32_t s = (int32_t)a + (int32_t)b;
    if (s > 32767) {
        if (saturated != NULL) *saturated = true;
        return 32767;
    }
    if (s <= LC_LOG_ZERO) {
        if (saturated != NULL) *saturated = true;
        return LC_LOG_ZERO;
    }
    return (int16_t)s;
}

static inline lc_logc16_t lc_fhrr_bind_sat(lc_logc16_t a, lc_logc16_t b, bool *saturated) {
    lc_logc16_t out = {
        lc_log_add_rho(a.rho, b.rho, saturated),
        (uint16_t)(a.theta + b.theta)
    };
    return out;
}

static inline lc_logc16_t lc_fhrr_bind(lc_logc16_t a, lc_logc16_t b) {
    return lc_fhrr_bind_sat(a, b, NULL);
}

static inline lc_logc16_t lc_fhrr_unbind(lc_logc16_t a, lc_logc16_t key) {
    int16_t inv_rho = (key.rho == LC_LOG_ZERO) ? LC_LOG_ZERO : (int16_t)(-key.rho);
    lc_logc16_t out = {
        lc_log_add_rho(a.rho, inv_rho, NULL),
        (uint16_t)(a.theta - key.theta)
    };
    return out;
}

/* 2-射 (経路合流差 α: z2 => z1) */
static inline lc_2morph_t lc_eval_2morph(lc_logc16_t z1, lc_logc16_t z2) {
    lc_2morph_t m = { 0, 0, false };
    if (z1.rho == LC_LOG_ZERO && z2.rho == LC_LOG_ZERO) return m;
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

static inline int32_t lc_2morph_phase_norm(lc_2morph_t m) {
    return (m.dtheta == -32768) ? 32768 : lc_iabs32((int32_t)m.dtheta);
}

static inline int32_t lc_2morph_l1(lc_2morph_t m) {
    return lc_iabs32(m.drho) + lc_2morph_phase_norm(m) + (m.has_bot ? 65536 : 0);
}

static inline bool lc_2morph_is_flat(lc_2morph_t m) {
    return !m.has_bot && m.drho == 0 && m.dtheta == 0;
}

static inline void lc_cell_reset(LiftingCell *c) { memset(c, 0, sizeof(*c)); }

static inline void lc_cell_bottom_up(LiftingCell *c) {
    lc_c16_t g = (c->in_g != NULL) ? c->in_g->s_eff : (lc_c16_t){LC_Q14_ONE, 0};
    c->s_eff = lc_c16_mul_q14(g, c->in_s->s_eff);
    if (lc_c16_l1(c->s_eff) > 0) c->flags |= LC_FLAG_ACTIVE;
    else                         c->flags &= (uint8_t)(~LC_FLAG_ACTIVE);
    c->p = lc_c16_mul_q14(c->w, c->s_eff);
    c->r = lc_c16_sub_mod(c->in_v->r, c->p);
}

static inline void lc_cell_top_down(LiftingCell *c, lc_exec_mode_t mode, bool is_top) {
    if (is_top) c->td_r = (mode == LC_MODE_EVAL) ? (lc_c16_t){0, 0} : c->r;
    c->restored = lc_c16_add_mod(c->td_r, c->p);
    if (c->in_v != NULL) c->in_v->td_r = c->restored;
}

static inline void lc_update_weight_mod(lc_c16_t *w, lc_c16_t err, lc_c16_t s, uint8_t lr_shift) {
    if ((err.re == 0 && err.im == 0) || (s.re == 0 && s.im == 0)) return;
    int64_t grad_re = (int64_t)err.re * s.re + (int64_t)err.im * s.im;
    int64_t grad_im = (int64_t)err.im * s.re - (int64_t)err.re * s.im;
    int32_t shift   = 14 + (int32_t)lr_shift;
    int32_t step_re = lc_sym_rshift64(grad_re, shift);
    int32_t step_im = lc_sym_rshift64(grad_im, shift);
    if (step_re == 0 && grad_re != 0) step_re = (grad_re > 0) ? 1 : -1;
    if (step_im == 0 && grad_im != 0) step_im = (grad_im > 0) ? 1 : -1;
    w->re = (int16_t)((uint16_t)w->re + (uint16_t)step_re);
    w->im = (int16_t)((uint16_t)w->im + (uint16_t)step_im);
}

/* 64-byte LiftingCellLog 1-射評価 & ゲージ緩和ステップ */
static inline int32_t lc_half_step_floor(int32_t d) {
    if (d == 0) return 0;
    return (d > 0) ? ((d + 1) >> 1) : -((-d + 1) >> 1);
}

static inline void lc_cell_log_eval(LiftingCellLog *c) {
    lc_logc16_t s = c->in_s->s_eff;
    if (c->in_g != NULL && c->in_g->s_eff.rho == LC_LOG_ZERO) {
        c->s_eff = s;
        return;
    }
    c->s_eff = lc_fhrr_bind(s, c->w);
}

static inline void lc_cell_log_gauge_step(LiftingCellLog *c, int32_t drho, int32_t dtheta) {
    if ((c->flags & LC_FLAG_LOCKED) != 0u) return;
    int32_t sr = lc_half_step_floor(drho);
    int32_t st = lc_half_step_floor(dtheta);
    int32_t nr = (int32_t)c->w.rho + sr;
    if (nr >  32767) nr =  32767;
    if (nr < -32767) nr = -32767;
    c->w.rho   = (int16_t)nr;
    c->w.theta = (uint16_t)(c->w.theta + (uint16_t)(int16_t)st);
}

/* ----------------------------------------------------------------------------
 * Layer 1: 純整数双対チャート (Log <-> Cart, 7.5 KB L1D LUT) & 閉ループ梯子
 * ---------------------------------------------------------------------------- */
bool        lc_dual_chart_init(void);
bool        lc_v2_init(void); /* Dual-Chart + ECC 全 LUT 統合初期化 */
lc_c16_t    lc_log_to_cart(int16_t rho, uint16_t theta);
lc_logc16_t lc_cart_to_log(int16_t re, int16_t im);
int16_t     lc_trig_cos_q14(uint16_t theta);
int16_t     lc_trig_sin_q14(uint16_t theta);

void lc_dual_ladder_forward(lc_c16_t *s, lc_c16_t *v, int n_pairs,
                            const lc_logc16_t *weights, int depth);
void lc_dual_ladder_inverse(lc_c16_t *s, lc_c16_t *v, int n_pairs,
                            const lc_logc16_t *weights, int depth);

/* ----------------------------------------------------------------------------
 * Layer 2: 高次圏 2-射ダイアモンド・3-射自律書き換え・Associator・ZX-Spider
 * ---------------------------------------------------------------------------- */
typedef struct {
    LiftingCellLog  pool[16];
    int             used;
    LiftingCellLog  port_a;
    LiftingCellLog  port_ctx;
    LiftingCellLog *f1;
    LiftingCellLog *f2;
    LiftingCellLog *g1;
    LiftingCellLog *g2;
    LiftingCellLog *p2_tail;
} lc_diamond_t;

void        lc_diamond_init(lc_diamond_t *d, lc_logc16_t base_a,
                            lc_logc16_t w_f1, lc_logc16_t w_f2, lc_logc16_t w_g2_init);
lc_2morph_t lc_diamond_eval_ctx(lc_diamond_t *d, bool ctx_on, lc_logc16_t env_twist);
bool        lc_diamond_solve_h1(lc_diamond_t *d, lc_logc16_t env_twist,
                                bool *out_spawned_3morph, int *out_steps,
                                int32_t *out_static_l1);

/* Associator 2-射検出 & 飽和/偽⊥回避のための自律結合組み換え */
bool lc_assoc_resolve(lc_logc16_t a, lc_logc16_t b, lc_logc16_t c,
                      lc_logc16_t *out_val, lc_2morph_t *out_alpha,
                      bool *out_reassociated);

/* Frobenius 双対 (ZX-Spider) プリミティブ */
static inline void lc_zx_z_copy(lc_logc16_t in, lc_logc16_t phase,
                                lc_logc16_t *out0, lc_logc16_t *out1) {
    lc_logc16_t y = lc_fhrr_bind(in, phase);
    *out0 = y;
    *out1 = y;
}

static inline uint16_t lc_zx_x_parity(uint16_t a, uint16_t b, uint16_t phase) {
    return (uint16_t)(a + b + phase);
}

static inline lc_c16_t lc_zx_x_bundle(lc_c16_t a, lc_c16_t b) {
    return lc_c16_add_mod(a, b);
}

/* ----------------------------------------------------------------------------
 * Layer 3: 動的カリー化 3 入力ゲート自己合成 & 参照カウント付き共有 DAG プール
 * ---------------------------------------------------------------------------- */
#define LC_NET_MAX_VARS  3
#define LC_NET_MAX_CELLS 16

typedef bool (*lc_bool_fn3_t)(bool, bool, bool);

typedef struct {
    LiftingCell in_ports[LC_NET_MAX_VARS];
    LiftingCell in_target;
    LiftingCell cells[LC_NET_MAX_CELLS];
    int         num_cells;
    int         spawned_count;
    uint8_t     num_vars;
    uint8_t     lr_shift;
} lc_currying_net_t;

void     lc_net_init(lc_currying_net_t *net, uint8_t num_vars);
bool     lc_net_synthesize(lc_currying_net_t *net, lc_bool_fn3_t fn, int max_epochs);
bool     lc_net_verify_exact(lc_currying_net_t *net, lc_bool_fn3_t fn);
lc_c16_t lc_net_forward_cont(lc_currying_net_t *net, const int16_t u_in[3], lc_c16_t target);
void     lc_net_backward_adj(lc_currying_net_t *net, lc_c16_t top_r);
int32_t  lc_net_eval_discrete_err(lc_currying_net_t *net, const bool b_in[3], bool b_out);

#ifndef LC_DAG_POOL_CAP
#define LC_DAG_POOL_CAP 256
#endif
#define LC_DAG_MAX_LITS 8
#define LC_DAG_GLUE_ORD 3u

typedef struct {
    uint16_t var_id;
    uint8_t  pol_bit;
} lc_lit_t;

typedef struct {
    LiftingCell pool[LC_DAG_POOL_CAP];
    LiftingCell zero_target;
    bool        slot_used[LC_DAG_POOL_CAP];
    int         alive_cells;
    int         active_terms;
    int         spawned_cells;
    int         shared_hits;
    int         pruned_cells;
    int         reused_slots;
} lc_dag_pool_t;

void         lc_dag_init(lc_dag_pool_t *dp);
LiftingCell *lc_dag_add_term(lc_dag_pool_t *dp, const lc_lit_t *lits, int n_lits);
bool         lc_dag_prune_leaf(lc_dag_pool_t *dp, bool allow_evict_short);
int          lc_dag_extract_lits(const LiftingCell *term, lc_lit_t out_lits[LC_DAG_MAX_LITS]);

/* ----------------------------------------------------------------------------
 * Layer 4: 最小十分原因 + 中間変数 1-UIP カリー化 CDCL アブダクションエンジン
 * ---------------------------------------------------------------------------- */
#ifndef LC_CDCL_MAX_VARS
#define LC_CDCL_MAX_VARS  512
#endif
#ifndef LC_CDCL_MAX_GATES
#define LC_CDCL_MAX_GATES 512
#endif
#define LC_CDCL_MAX_INC   24
#define LC_CDCL_MAX_DEPTH 64
#define LC_CDCL_MAX_REAS  64

typedef struct {
    int16_t  u;
    int32_t  conf;
    int16_t  level;
    int16_t  trail_pos;
    bool     is_decision;
    int8_t   saved_phase;
    int16_t  bond_deg;
    int32_t  activity;
    int32_t  force;
    uint16_t reason_vars[LC_CDCL_MAX_REAS];
    uint8_t  reason_len;
    uint16_t inc_gates[LC_CDCL_MAX_INC];
    uint8_t  inc_cnt;
} lc_phase_var_t;

typedef struct {
    lc_currying_net_t *net;
    int                in_var[3];
    int                out_var;
} lc_lifting_gate_t;

typedef struct {
    lc_phase_var_t    vars[LC_CDCL_MAX_VARS];
    lc_lifting_gate_t gates[LC_CDCL_MAX_GATES];
    int               num_vars;
    int               num_gates;

    uint16_t          trail[LC_CDCL_MAX_VARS];
    int               trail_len;
    int               trail_lim[LC_CDCL_MAX_DEPTH + 1];
    int               cur_level;

    uint16_t          gate_q[LC_CDCL_MAX_GATES];
    bool              in_gate_q[LC_CDCL_MAX_GATES];
    int               gq_head;
    int               gq_tail;

    int               dec_var[LC_CDCL_MAX_DEPTH + 1];
    int32_t           act_inc;
    lc_dag_pool_t     cache;
} lc_cdcl_graph_t;

typedef struct {
    int branches;
    int conflicts;
    int backjumps;
    int cache_props;
    int cache_spawned;
    int cache_pruned;
} lc_cdcl_stats_t;

void lc_cdcl_init(lc_cdcl_graph_t *g);
int  lc_cdcl_add_var(lc_cdcl_graph_t *g, int8_t obs_val);
void lc_cdcl_add_gate(lc_cdcl_graph_t *g, lc_currying_net_t *net,
                      int in0, int in1, int in2, int out_v);
bool lc_cdcl_solve(lc_cdcl_graph_t *g, int max_conflicts,
                   bool out_bits[LC_CDCL_MAX_VARS], lc_cdcl_stats_t *st);

/* ----------------------------------------------------------------------------
 * Layer 5: 連続位相軟判定 ECC (Hamming / QC-LDPC, -lm 不要)
 * ---------------------------------------------------------------------------- */
#ifndef LC_ECC_MAX_N
#define LC_ECC_MAX_N   256
#endif
#ifndef LC_ECC_MAX_M
#define LC_ECC_MAX_M   128
#endif
#define LC_ECC_MAX_DEG 8
#define LC_ECC_MAX_E   1024

typedef struct {
    int vars[LC_ECC_MAX_DEG];
    int edges[LC_ECC_MAX_DEG];
    int deg;
} lc_ecc_check_t;

typedef struct {
    int edges[LC_ECC_MAX_DEG];
    int deg;
} lc_ecc_vadj_t;

typedef struct {
    int     var_id;
    int     chk_id;
    int32_t msg_v2c;
    int32_t msg_c2v;
} lc_ecc_edge_t;

typedef struct {
    lc_currying_net_t *xor3_net;
    int                n_vars;
    int                n_checks;
    int                n_edges;
    lc_ecc_check_t     checks[LC_ECC_MAX_M];
    lc_ecc_vadj_t      vadj[LC_ECC_MAX_N];
    lc_ecc_edge_t      edges[LC_ECC_MAX_E];
    LiftingCell        p_in[LC_ECC_MAX_DEG];
    LiftingCell        p_pref[LC_ECC_MAX_DEG];
    LiftingCell        p_suff[LC_ECC_MAX_DEG];
    LiftingCell        p_ext[LC_ECC_MAX_DEG];
    LiftingCell        p_zero;
} lc_ecc_graph_t;

void lc_ecc_init_luts(void);
void lc_ecc_init(lc_ecc_graph_t *g, lc_currying_net_t *xor3_net, int n_vars);
void lc_ecc_add_check(lc_ecc_graph_t *g, const int *vars, int deg);
int  lc_ecc_syndrome_errors(lc_ecc_graph_t *g, const uint8_t *bits);
int  lc_ecc_decode_erasure(lc_ecc_graph_t *g, const int8_t *obs, uint8_t *out_bits);
int  lc_ecc_decode_soft(lc_ecc_graph_t *g, const int32_t *ch_llr,
                        int max_iters, bool damping, uint8_t *out_bits);

/* ----------------------------------------------------------------------------
 * Layer 6: L1 ブロック分割 ARM64 NEON AoSoA-8 並列カーネル群 (0-bit 一致保証)
 * ---------------------------------------------------------------------------- */
#define LC_L1_BLOCK_CELLS  512
#define LC_L1_BLOCK_GROUPS (LC_L1_BLOCK_CELLS / 8)

/* 6-A: Q1.14 3-pass AoSoA-8 (384 B = 64B x 6) */
typedef struct LC_ALIGN64 {
    int16_t s_re[8], s_im[8];
    int16_t g_re[8], g_im[8];
    int16_t v_re[8], v_im[8];
    int16_t w_re[8], w_im[8];

    int16_t s_eff_re[8], s_eff_im[8];
    int16_t p_re[8],     p_im[8];
    int16_t r_re[8],     r_im[8];
    int16_t restored_re[8], restored_im[8];

    int32_t adj_in[8];
    int32_t adj_s[8];
    int32_t adj_g[8];
    int32_t gate_mask[8];
} lc_block8_t;

/* 6-B: Log-Polar FHRR & Dual-Chart AoSoA-8 (320 B = 64B x 5) */
typedef struct LC_ALIGN64 {
    int16_t  s_rho[8];
    uint16_t s_theta[8];
    int16_t  g_rho[8];
    uint16_t g_theta[8];
    int16_t  w_rho[8];
    uint16_t w_theta[8];

    int16_t  v_re[8];
    int16_t  v_im[8];

    int16_t  se_rho[8];
    uint16_t se_theta[8];
    int16_t  p_rho[8];
    uint16_t p_theta[8];

    int16_t  p_re[8];
    int16_t  p_im[8];
    int16_t  r_re[8];
    int16_t  r_im[8];
    int16_t  res_re[8];
    int16_t  res_im[8];

    int16_t  unb_rho[8];
    uint16_t unb_theta[8];
} lc_log_block8_t;

/* 6-C: Cart->Log 純整数逆写像 AoSoA-8 (128 B = 64B x 2) */
typedef struct LC_ALIGN64 {
    int16_t  re[8];
    int16_t  im[8];
    int16_t  rho[8];
    uint16_t theta[8];
    int16_t  rt_re[8];
    int16_t  rt_im[8];
    int16_t  pad[16];
} lc_chart_block8_t;

/* 6-D: 2-射ホロノミー一括検出 AoSoA-8 (128 B = 64B x 2) */
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

typedef char lc_assert_block8_384b[(sizeof(lc_block8_t) == 384) ? 1 : -1];
typedef char lc_assert_log_block8_320b[(sizeof(lc_log_block8_t) == 320) ? 1 : -1];
typedef char lc_assert_chart_block8_128b[(sizeof(lc_chart_block8_t) == 128) ? 1 : -1];
typedef char lc_assert_holo_block8_128b[(sizeof(lc_holo_block8_t) == 128) ? 1 : -1];

/* Q1.14 3-pass カーネル */
void lc_aosoa8_forward(lc_block8_t *blocks, int n_blocks);
void lc_aosoa8_backward(lc_block8_t *blocks, int n_blocks);
void lc_aosoa8_update(lc_block8_t *blocks, int n_blocks, uint8_t lr_shift);
void lc_aosoa8_step_l1_blocked(lc_block8_t *blocks, int n_blocks, uint8_t lr_shift);

/* Log-Polar FHRR / Dual-Chart / Cart->Log / Holonomy 2-Morphism カーネル (NEON + スカラー参照) */
void     lc_log_aosoa8_bind_only_scalar(lc_log_block8_t *blocks, int n_blocks);
void     lc_log_aosoa8_dual_step_scalar(lc_log_block8_t *blocks, int n_blocks);
void     lc_log_aosoa8_bind_only(lc_log_block8_t *blocks, int n_blocks);
void     lc_log_aosoa8_dual_step(lc_log_block8_t *blocks, int n_blocks);

void     lc_chart_aosoa8_cart_to_log_scalar(lc_chart_block8_t *blocks, int n_blocks);
void     lc_chart_aosoa8_cart_to_log(lc_chart_block8_t *blocks, int n_blocks);

uint64_t lc_holo_aosoa8_eval_2morph_scalar(lc_holo_block8_t *blocks, int n_blocks, bool apply_twist);
uint64_t lc_holo_aosoa8_eval_2morph(lc_holo_block8_t *blocks, int n_blocks, bool apply_twist);

/* ============================================================================
 * IMPLEMENTATION SECTION
 * ============================================================================ */
#ifdef LIFTING_CORE_V2_IMPLEMENTATION

/* --- Layer 1: Dual-Chart LUTs & Closed-Loop Ladder Implementation --- */
static int16_t  LC_ALIGN64 lc_lut_cos[LC_TRIG_SIZE];
static int16_t  LC_ALIGN64 lc_lut_sin[LC_TRIG_SIZE];
static uint16_t LC_ALIGN64 lc_lut_exp2_frac[256];
static uint16_t LC_ALIGN64 lc_lut_half_log2[LC_LOG2_SIZE + 1];
static uint16_t LC_ALIGN64 lc_lut_atan[LC_ATAN_SIZE + 1];

static double lc_math_sqrt_pos(double x) {
    if (x <= 0.0) return 0.0;
    double y = (x >= 1.0) ? x : 1.0;
    for (int i = 0; i < 20; i++) y = 0.5 * (y + x / y);
    return y;
}

static double lc_math_sin(double x) {
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

static double lc_math_cos(double x) {
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

static double lc_math_exp2_frac(double f) {
    static const double LN2 = 0.693147180559945309417;
    double x = f * LN2, term = 1.0, sum = 1.0;
    for (int k = 1; k <= 12; k++) {
        term *= x / (double)k;
        sum  += term;
    }
    return sum;
}

static double lc_math_log2_1p(double f) {
    static const double INV_LN2 = 1.44269504088896340736;
    double u  = f / (2.0 + f);
    double u2 = u * u, p = u, sum = 0.0;
    for (int k = 1; k <= 17; k += 2) {
        sum += p / (double)k;
        p   *= u2;
    }
    return 2.0 * sum * INV_LN2;
}

static double lc_math_atan_unit(double t) {
    double u = t / (1.0 + lc_math_sqrt_pos(1.0 + t * t));
    u = u / (1.0 + lc_math_sqrt_pos(1.0 + u * u));
    double u2 = u * u, p = u, sum = 0.0;
    for (int k = 0; k < 9; k++) {
        double term = p / (double)(2 * k + 1);
        sum += ((k & 1) == 0) ? term : -term;
        p   *= u2;
    }
    return 4.0 * sum;
}

bool lc_dual_chart_init(void) {
    static const double TWO_PI = 6.28318530717958647692;
    static const double PI_4   = 0.78539816339744830962;

    double e_log  = lc_math_log2_1p(1.0) - 1.0;
    double e_atan = lc_math_atan_unit(1.0) - PI_4;
    if (e_log < -1e-9 || e_log > 1e-9 || e_atan < -1e-9 || e_atan > 1e-9) return false;

    for (int i = 0; i < LC_TRIG_SIZE; i++) {
        double ang = TWO_PI * (double)i / (double)LC_TRIG_SIZE;
        double c   = lc_math_cos(ang) * (double)LC_Q14_ONE;
        double s   = lc_math_sin(ang) * (double)LC_Q14_ONE;
        lc_lut_cos[i] = (int16_t)((c >= 0.0) ? (c + 0.5) : (c - 0.5));
        lc_lut_sin[i] = (int16_t)((s >= 0.0) ? (s + 0.5) : (s - 0.5));
    }
    for (int f = 0; f < 256; f++) {
        double v = lc_math_exp2_frac((double)f / 256.0) * (double)LC_Q14_ONE;
        lc_lut_exp2_frac[f] = (uint16_t)(v + 0.5);
    }
    for (int i = 0; i <= LC_LOG2_SIZE; i++) {
        double v = 128.0 * lc_math_log2_1p((double)i / (double)LC_LOG2_SIZE);
        lc_lut_half_log2[i] = (uint16_t)(v + 0.5);
    }
    for (int i = 0; i <= LC_ATAN_SIZE; i++) {
        double ang = lc_math_atan_unit((double)i / (double)LC_ATAN_SIZE);
        double q   = ang * (65536.0 / TWO_PI);
        lc_lut_atan[i] = (uint16_t)(q + 0.5);
    }
    return true;
}

bool lc_v2_init(void) {
    if (!lc_dual_chart_init()) return false;
    lc_ecc_init_luts();
    return true;
}

int16_t lc_trig_cos_q14(uint16_t theta) {
    uint32_t idx = (((uint32_t)theta + (1u << (LC_TRIG_SHIFT - 1))) >> LC_TRIG_SHIFT)
                   & (LC_TRIG_SIZE - 1u);
    return lc_lut_cos[idx];
}

int16_t lc_trig_sin_q14(uint16_t theta) {
    uint32_t idx = (((uint32_t)theta + (1u << (LC_TRIG_SHIFT - 1))) >> LC_TRIG_SHIFT)
                   & (LC_TRIG_SIZE - 1u);
    return lc_lut_sin[idx];
}

lc_c16_t lc_log_to_cart(int16_t rho, uint16_t theta) {
    int32_t e = (int32_t)(rho >> 8);
    if (e < -16) e = -16;
    if (e >   2) e =   2;

    int32_t base = (int32_t)lc_lut_exp2_frac[(uint8_t)(rho & 0xFF)];
    int32_t mag  = (e >= 0) ? (base << e) : (base >> (-e));
    if (mag > 32767) mag = 32767;

    uint32_t idx = (((uint32_t)theta + (1u << (LC_TRIG_SHIFT - 1))) >> LC_TRIG_SHIFT)
                   & (LC_TRIG_SIZE - 1u);
    lc_c16_t out = {
        lc_quantize_q14_s32(mag * (int32_t)lc_lut_cos[idx]),
        lc_quantize_q14_s32(mag * (int32_t)lc_lut_sin[idx])
    };
    return out;
}

lc_logc16_t lc_cart_to_log(int16_t re, int16_t im) {
    if (re == 0 && im == 0) {
        lc_logc16_t z = { LC_LOG_ZERO, 0u };
        return z;
    }
    uint32_t ax   = (re >= 0) ? (uint32_t)re : (uint32_t)(-(int32_t)re);
    uint32_t ay   = (im >= 0) ? (uint32_t)im : (uint32_t)(-(int32_t)im);
    uint32_t r2   = ax * ax + ay * ay;
    uint32_t lz   = lc_clz32(r2);
    int32_t  k    = 31 - (int32_t)lz;
    uint32_t norm = r2 << lz;
    uint32_t lidx = ((norm & 0x7FFFFFFFu) + (1u << 21)) >> 22;
    int16_t  rho  = (int16_t)(128 * (k - 28) + (int32_t)lc_lut_half_log2[lidx]);

    uint32_t hi    = (ax >= ay) ? ax : ay;
    uint32_t lo    = (ax >= ay) ? ay : ax;
    uint32_t aidx  = ((lo << LC_ATAN_BITS) + (hi >> 1)) / hi;
    uint16_t phi   = lc_lut_atan[aidx];
    uint16_t oct   = (ay > ax) ? (uint16_t)(16384u - phi) : phi;
    uint16_t half  = (re < 0)  ? (uint16_t)(32768u - oct) : oct;
    uint16_t theta = (im < 0)  ? (uint16_t)(0u - half)    : half;

    lc_logc16_t out = { rho, theta };
    return out;
}

void lc_dual_ladder_forward(lc_c16_t *s, lc_c16_t *v, int n_pairs,
                            const lc_logc16_t *weights, int depth) {
    for (int L = 0; L < depth; L++) {
        for (int i = 0; i < n_pairs; i++) {
            lc_logc16_t sl = lc_cart_to_log(s[i].re, s[i].im);
            lc_logc16_t pl = lc_fhrr_bind(weights[L], sl);
            lc_c16_t    pc = lc_log_to_cart(pl.rho, pl.theta);
            lc_c16_t    vn = lc_c16_sub_mod(v[i], pc);
            v[i] = s[i];
            s[i] = vn;
        }
    }
}

void lc_dual_ladder_inverse(lc_c16_t *s, lc_c16_t *v, int n_pairs,
                            const lc_logc16_t *weights, int depth) {
    for (int L = depth - 1; L >= 0; L--) {
        for (int i = 0; i < n_pairs; i++) {
            lc_c16_t    s_prev = v[i];
            lc_c16_t    vn     = s[i];
            lc_logc16_t sl     = lc_cart_to_log(s_prev.re, s_prev.im);
            lc_logc16_t pl     = lc_fhrr_bind(weights[L], sl);
            lc_c16_t    pc     = lc_log_to_cart(pl.rho, pl.theta);
            s[i] = s_prev;
            v[i] = lc_c16_add_mod(vn, pc);
        }
    }
}

/* --- Layer 2: Higher-Category Diamond, 3-Morphism Rewrite & Associator --- */
static LiftingCellLog *lc_diamond_alloc(lc_diamond_t *d) {
    if (d->used >= 16) return NULL;
    LiftingCellLog *c = &d->pool[d->used++];
    memset(c, 0, sizeof(*c));
    c->flags = LC_FLAG_ALIVE;
    return c;
}

void lc_diamond_init(lc_diamond_t *d, lc_logc16_t base_a,
                     lc_logc16_t w_f1, lc_logc16_t w_f2, lc_logc16_t w_g2_init) {
    memset(d, 0, sizeof(*d));
    d->port_a.s_eff = base_a;
    d->f1 = lc_diamond_alloc(d); d->f2 = lc_diamond_alloc(d);
    d->g1 = lc_diamond_alloc(d); d->g2 = lc_diamond_alloc(d);

    d->f1->in_s = &d->port_a; d->f1->w = w_f1;      d->f1->flags |= LC_FLAG_LOCKED;
    d->f2->in_s = d->f1;      d->f2->w = w_f2;      d->f2->flags |= LC_FLAG_LOCKED;
    d->g1->in_s = &d->port_a; d->g1->w = w_f1;      d->g1->flags |= LC_FLAG_LOCKED;
    d->g2->in_s = d->g1;      d->g2->w = w_g2_init;
    d->p2_tail  = d->g2;
}

lc_2morph_t lc_diamond_eval_ctx(lc_diamond_t *d, bool ctx_on, lc_logc16_t env_twist) {
    d->port_ctx.s_eff = ctx_on ? (lc_logc16_t){ 0, 0 } : (lc_logc16_t){ LC_LOG_ZERO, 0 };
    lc_cell_log_eval(d->f1);
    lc_cell_log_eval(d->f2);
    lc_logc16_t p1 = ctx_on ? lc_fhrr_bind(d->f2->s_eff, env_twist) : d->f2->s_eff;

    lc_cell_log_eval(d->g1);
    lc_cell_log_eval(d->g2);
    if (d->p2_tail != d->g2) lc_cell_log_eval(d->p2_tail);
    return lc_eval_2morph(p1, d->p2_tail->s_eff);
}

bool lc_diamond_solve_h1(lc_diamond_t *d, lc_logc16_t env_twist,
                         bool *out_spawned_3morph, int *out_steps,
                         int32_t *out_static_l1) {
    if (out_spawned_3morph != NULL) *out_spawned_3morph = false;
    if (out_steps != NULL) *out_steps = 0;

    /* Stage 1: 静的ゲージ緩和を試行 */
    for (int step = 0; step <= 24; step++) {
        lc_2morph_t m0 = lc_diamond_eval_ctx(d, false, env_twist);
        lc_2morph_t m1 = lc_diamond_eval_ctx(d, true,  env_twist);
        if (lc_2morph_is_flat(m0) && lc_2morph_is_flat(m1)) {
            if (out_steps != NULL) *out_steps = step;
            if (out_static_l1 != NULL) *out_static_l1 = 0;
            return true;
        }
        lc_cell_log_gauge_step(d->g2, (m0.drho + m1.drho) / 2,
                                      ((int32_t)m0.dtheta + (int32_t)m1.dtheta) / 2);
    }

    lc_2morph_t m0_st = lc_diamond_eval_ctx(d, false, env_twist);
    lc_2morph_t m1_st = lc_diamond_eval_ctx(d, true,  env_twist);
    if (out_static_l1 != NULL) *out_static_l1 = lc_2morph_l1(m0_st) + lc_2morph_l1(m1_st);

    int32_t  inv_dr = m1_st.drho - m0_st.drho;
    uint16_t inv_dt = (uint16_t)((uint16_t)m1_st.dtheta - (uint16_t)m0_st.dtheta);
    if (inv_dr == 0 && inv_dt == 0u) return false;

    /* Stage 2: 非自明 H1 障害 -> 3-射発火 (文脈ゲート付き TwistedSection セル自律生成) */
    LiftingCellLog *tw = lc_diamond_alloc(d);
    if (tw == NULL) return false;
    tw->in_s  = d->g2;
    tw->in_g  = &d->port_ctx;
    tw->w     = (lc_logc16_t){ 0, 0 };
    tw->order = 2;
    tw->flags |= LC_FLAG_CURRIED_TWIST;
    d->p2_tail = tw;
    if (out_spawned_3morph != NULL) *out_spawned_3morph = true;

    for (int step = 0; step <= 24; step++) {
        lc_2morph_t m0 = lc_diamond_eval_ctx(d, false, env_twist);
        lc_2morph_t m1 = lc_diamond_eval_ctx(d, true,  env_twist);
        if (lc_2morph_is_flat(m0) && lc_2morph_is_flat(m1)) {
            if (out_steps != NULL) *out_steps = step;
            return true;
        }
        lc_cell_log_gauge_step(d->g2, m0.drho, (int32_t)m0.dtheta);
        lc_2morph_t m1_rem = lc_diamond_eval_ctx(d, true, env_twist);
        lc_cell_log_gauge_step(tw, m1_rem.drho, (int32_t)m1_rem.dtheta);
    }
    return false;
}

bool lc_assoc_resolve(lc_logc16_t a, lc_logc16_t b, lc_logc16_t c,
                      lc_logc16_t *out_val, lc_2morph_t *out_alpha,
                      bool *out_reassociated) {
    bool sat_L = false, sat_R = false;
    lc_logc16_t res_L = lc_fhrr_bind_sat(lc_fhrr_bind_sat(a, b, &sat_L), c, &sat_L);
    lc_logc16_t res_R = lc_fhrr_bind_sat(a, lc_fhrr_bind_sat(b, c, &sat_R), &sat_R);
    lc_2morph_t alpha = lc_eval_2morph(res_R, res_L);
    if (out_alpha != NULL) *out_alpha = alpha;

    if (!lc_2morph_is_flat(alpha) && sat_L && !sat_R) {
        if (out_val != NULL) *out_val = res_R;
        if (out_reassociated != NULL) *out_reassociated = true;
        return true;
    }
    if (!lc_2morph_is_flat(alpha) && sat_R && !sat_L) {
        if (out_val != NULL) *out_val = res_L;
        if (out_reassociated != NULL) *out_reassociated = true;
        return true;
    }
    if (out_val != NULL) *out_val = res_L;
    if (out_reassociated != NULL) *out_reassociated = false;
    return !sat_L;
}

/* --- Layer 3: Dynamic Currying Synthesis & Shared DAG Implementation --- */
void lc_net_init(lc_currying_net_t *net, uint8_t num_vars) {
    net->num_vars      = num_vars;
    net->num_cells     = 0;
    net->spawned_count = 0;
    net->lr_shift      = 2;
    lc_cell_reset(&net->in_target);
    for (int i = 0; i < LC_NET_MAX_VARS; i++) {
        lc_cell_reset(&net->in_ports[i]);
        net->in_ports[i].var_mask = (uint8_t)(1u << i);
        net->in_ports[i].order    = 1;
    }
    for (int i = 0; i < LC_NET_MAX_CELLS; i++) lc_cell_reset(&net->cells[i]);
    for (uint8_t i = 0; i < num_vars; i++) {
        LiftingCell *c = &net->cells[i];
        c->in_s     = &net->in_ports[i];
        c->in_g     = NULL;
        c->in_v     = (i == 0) ? &net->in_target : &net->cells[i - 1];
        c->var_mask = (uint8_t)(1u << i);
        c->order    = 1;
        c->flags    = LC_FLAG_ALIVE;
        net->num_cells++;
    }
}

static bool lc_net_has_mask(const lc_currying_net_t *net, uint8_t mask) {
    for (int i = 0; i < net->num_cells; i++) {
        if (net->cells[i].var_mask == mask) return true;
    }
    return false;
}

lc_c16_t lc_net_forward_cont(lc_currying_net_t *net, const int16_t u_in[3], lc_c16_t target) {
    for (uint8_t i = 0; i < net->num_vars; i++) {
        lc_c16_t v = { u_in[i], 0 };
        net->in_ports[i].s_eff = v;
        net->in_ports[i].r     = v;
        net->in_ports[i].adj_s = 0;
    }
    net->in_target.s_eff = target;
    net->in_target.r     = target;
    net->in_target.adj_s = 0;
    for (int i = 0; i < net->num_cells; i++) {
        net->cells[i].adj_s = 0;
        lc_cell_bottom_up(&net->cells[i]);
    }
    return net->cells[net->num_cells - 1].r;
}

void lc_net_backward_adj(lc_currying_net_t *net, lc_c16_t top_r) {
    if (top_r.re == 0 && top_r.im == 0) return;
    for (int i = net->num_cells - 1; i >= 0; i--) {
        LiftingCell *c = &net->cells[i];
        int32_t eff_re = top_r.re;
        int32_t eff_im = (top_r.im == LC_Q14_NOT)
                         ? ((c->s_eff.re > LC_Q14_HALF) ? 32767 : -32768)
                         : top_r.im;
        int64_t f64   = (int64_t)c->w.re * eff_re + (int64_t)c->w.im * eff_im;
        int32_t total = c->adj_s + lc_sym_rshift64(f64, 14);
        if (total == 0) continue;
        if (c->in_g == NULL) {
            c->in_s->adj_s += total;
        } else {
            int32_t gv = c->in_g->s_eff.re, sv = c->in_s->s_eff.re;
            if (total > 0) {
                if (gv < 2048) gv = 2048;
                if (sv < 2048) sv = 2048;
            }
            c->in_s->adj_s += lc_sym_rshift64((int64_t)total * gv, 14);
            c->in_g->adj_s += lc_sym_rshift64((int64_t)total * sv, 14);
        }
    }
}

static lc_c16_t lc_net_step_disc(lc_currying_net_t *net, const bool b_in[3],
                                 lc_c16_t tgt, lc_exec_mode_t mode) {
    uint8_t amask = 0;
    int16_t u[3]  = {0, 0, 0};
    for (uint8_t i = 0; i < net->num_vars; i++) {
        u[i] = b_in[i] ? LC_Q14_ONE : 0;
        if (b_in[i]) amask |= (uint8_t)(1u << i);
    }
    (void)lc_net_forward_cont(net, u, tgt);
    for (int i = net->num_cells - 1; i >= 0; i--) {
        lc_cell_top_down(&net->cells[i], mode, i == net->num_cells - 1);
    }
    if (mode == LC_MODE_LEARN) {
        for (int i = 0; i < net->num_cells; i++) {
            LiftingCell *c = &net->cells[i];
            if ((c->flags & LC_FLAG_ACTIVE) != 0u && c->var_mask == amask) {
                lc_update_weight_mod(&c->w, c->r, c->s_eff, net->lr_shift);
            }
        }
    }
    return net->cells[0].restored;
}

int32_t lc_net_eval_discrete_err(lc_currying_net_t *net, const bool b_in[3], bool b_out) {
    int16_t u[3] = {
        b_in[0] ? LC_Q14_ONE : 0,
        b_in[1] ? LC_Q14_ONE : 0,
        b_in[2] ? LC_Q14_ONE : 0
    };
    lc_c16_t tgt = b_out ? (lc_c16_t){0, LC_Q14_NOT} : (lc_c16_t){0, 0};
    return lc_c16_l1(lc_net_forward_cont(net, u, tgt));
}

bool lc_net_verify_exact(lc_currying_net_t *net, lc_bool_fn3_t fn) {
    for (int m = 0; m < 8; m++) {
        bool b[3] = { (m & 1) != 0, (m & 2) != 0, (m & 4) != 0 };
        lc_c16_t exp = fn(b[0], b[1], b[2]) ? (lc_c16_t){0, LC_Q14_NOT} : (lc_c16_t){0, 0};
        lc_c16_t ev  = lc_net_step_disc(net, b, exp, LC_MODE_EVAL);
        lc_c16_t rv  = lc_net_step_disc(net, b, exp, LC_MODE_EXACT);
        if (!lc_c16_eq(ev, exp) || !lc_c16_eq(rv, exp) ||
            lc_c16_l1(net->cells[net->num_cells - 1].r) != 0) {
            return false;
        }
    }
    return true;
}

bool lc_net_synthesize(lc_currying_net_t *net, lc_bool_fn3_t fn, int max_epochs) {
    for (int ep = 0; ep < max_epochs; ep++) {
        int64_t known_e = 0, unk_e = 0;
        int64_t pscore[LC_NET_MAX_CELLS][LC_NET_MAX_CELLS];
        memset(pscore, 0, sizeof(pscore));

        for (int m = 0; m < 8; m++) {
            bool b[3] = { (m & 1) != 0, (m & 2) != 0, (m & 4) != 0 };
            lc_c16_t tgt = fn(b[0], b[1], b[2]) ? (lc_c16_t){0, LC_Q14_NOT} : (lc_c16_t){0, 0};
            (void)lc_net_step_disc(net, b, tgt, LC_MODE_LEARN);
            int32_t en = lc_c16_l1(net->cells[net->num_cells - 1].r);
            if (en == 0) continue;

            if (lc_net_has_mask(net, (uint8_t)m)) {
                known_e += en;
            } else {
                unk_e += en;
                for (int u = 0; u < net->num_cells; u++) {
                    if ((net->cells[u].flags & LC_FLAG_ACTIVE) == 0u) continue;
                    for (int v = u + 1; v < net->num_cells; v++) {
                        if ((net->cells[v].flags & LC_FLAG_ACTIVE) == 0u) continue;
                        if ((net->cells[u].var_mask & net->cells[v].var_mask) != 0u) continue;
                        uint8_t um = (uint8_t)(net->cells[u].var_mask | net->cells[v].var_mask);
                        if (!lc_net_has_mask(net, um)) pscore[u][v] += en;
                    }
                }
            }
        }
        if (known_e == 0 && unk_e == 0) break;
        if (known_e == 0 && unk_e > 0 && net->num_cells < LC_NET_MAX_CELLS) {
            int bu = -1, bv = -1;
            int64_t bs = 0;
            uint8_t bb = 0;
            for (int u = 0; u < net->num_cells; u++) {
                for (int v = u + 1; v < net->num_cells; v++) {
                    if (pscore[u][v] <= 0) continue;
                    uint8_t ub = lc_popcount8((uint8_t)(net->cells[u].var_mask | net->cells[v].var_mask));
                    if (pscore[u][v] > bs || (pscore[u][v] == bs && ub > bb)) {
                        bs = pscore[u][v]; bb = ub; bu = u; bv = v;
                    }
                }
            }
            if (bu >= 0) {
                LiftingCell *nc = &net->cells[net->num_cells];
                lc_cell_reset(nc);
                nc->in_s     = &net->cells[bu];
                nc->in_g     = &net->cells[bv];
                nc->in_v     = &net->cells[net->num_cells - 1];
                nc->var_mask = (uint8_t)(net->cells[bu].var_mask | net->cells[bv].var_mask);
                nc->order    = lc_popcount8(nc->var_mask);
                nc->flags    = LC_FLAG_ALIVE;
                net->num_cells++;
                net->spawned_count++;
            }
        }
    }
    return lc_net_verify_exact(net, fn);
}

void lc_dag_init(lc_dag_pool_t *dp) {
    memset(dp, 0, sizeof(*dp));
    dp->zero_target.flags = LC_FLAG_ALIVE;
}

bool lc_dag_prune_leaf(lc_dag_pool_t *dp, bool allow_evict_short) {
    int best = -1, bscore = -1000000;
    for (int i = 0; i < LC_DAG_POOL_CAP; i++) {
        LiftingCell *c = &dp->pool[i];
        if ((c->flags & LC_FLAG_ALIVE) == 0u || c->ref_count != 0u) continue;
        bool is_glue = ((c->flags & LC_FLAG_CONFLICT_TERM) != 0u && c->order <= LC_DAG_GLUE_ORD);
        if (is_glue && !allow_evict_short) continue;
        int sc = (int)c->idle_counter * 16 + (int)c->order * 128 - c->meta_energy * 4;
        if (c->order <= LC_DAG_GLUE_ORD) sc -= 4096;
        if (best < 0 || sc > bscore) { bscore = sc; best = i; }
    }
    if (best < 0) return false;

    LiftingCell *q[LC_DAG_POOL_CAP];
    int qh = 0, qt = 0;
    q[qt++] = &dp->pool[best];
    while (qh < qt) {
        LiftingCell *c = q[qh++];
        if ((c->flags & LC_FLAG_ALIVE) == 0u || c->ref_count != 0u) continue;
        LiftingCell *ps = c->in_s, *pg = c->in_g;
        if ((c->flags & LC_FLAG_CONFLICT_TERM) != 0u) dp->active_terms--;
        lc_cell_reset(c);
        dp->alive_cells--;
        dp->pruned_cells++;
        if (ps != NULL && ps->ref_count > 0u) {
            ps->ref_count--;
            if (ps->ref_count == 0u && (ps->flags & LC_FLAG_CONFLICT_TERM) == 0u) q[qt++] = ps;
        }
        if (pg != NULL && pg->ref_count > 0u) {
            pg->ref_count--;
            if (pg->ref_count == 0u && (pg->flags & LC_FLAG_CONFLICT_TERM) == 0u) q[qt++] = pg;
        }
    }
    return true;
}

static LiftingCell *lc_dag_alloc(lc_dag_pool_t *dp) {
    for (int i = 0; i < LC_DAG_POOL_CAP; i++) {
        if ((dp->pool[i].flags & LC_FLAG_ALIVE) == 0u) {
            if (dp->slot_used[i]) dp->reused_slots++;
            dp->slot_used[i] = true;
            lc_cell_reset(&dp->pool[i]);
            dp->pool[i].flags = LC_FLAG_ALIVE;
            dp->alive_cells++;
            dp->spawned_cells++;
            return &dp->pool[i];
        }
    }
    return NULL;
}

static LiftingCell *lc_dag_get_lit(lc_dag_pool_t *dp, uint16_t vid, uint8_t pol) {
    for (int i = 0; i < LC_DAG_POOL_CAP; i++) {
        LiftingCell *c = &dp->pool[i];
        if ((c->flags & LC_FLAG_ALIVE) != 0u && c->order == 1 &&
            c->var_id == vid && c->var_mask == pol) {
            c->idle_counter = 0;
            dp->shared_hits++;
            return c;
        }
    }
    LiftingCell *nc = lc_dag_alloc(dp);
    if (nc == NULL) return NULL;
    nc->in_v        = &dp->zero_target;
    nc->var_id      = vid;
    nc->var_mask    = pol;
    nc->order       = 1;
    nc->meta_energy = 16;
    return nc;
}

static LiftingCell *lc_dag_get_pair(lc_dag_pool_t *dp, LiftingCell *lhs, LiftingCell *rhs) {
    for (int i = 0; i < LC_DAG_POOL_CAP; i++) {
        LiftingCell *c = &dp->pool[i];
        if ((c->flags & LC_FLAG_ALIVE) != 0u && c->order > 1 &&
            c->in_s == lhs && c->in_g == rhs) {
            c->idle_counter = 0;
            dp->shared_hits++;
            return c;
        }
    }
    LiftingCell *nc = lc_dag_alloc(dp);
    if (nc == NULL) return NULL;
    nc->in_s        = lhs;
    nc->in_g        = rhs;
    nc->in_v        = &dp->zero_target;
    nc->order       = (uint8_t)(lhs->order + 1u);
    nc->meta_energy = 16;
    lhs->ref_count++;
    rhs->ref_count++;
    return nc;
}

LiftingCell *lc_dag_add_term(lc_dag_pool_t *dp, const lc_lit_t *in_lits, int n_lits) {
    if (n_lits < 2 || n_lits > LC_DAG_MAX_LITS) return NULL;
    lc_lit_t lits[LC_DAG_MAX_LITS];
    for (int i = 0; i < n_lits; i++) lits[i] = in_lits[i];
    for (int i = 0; i < n_lits; i++) {
        for (int j = i + 1; j < n_lits; j++) {
            if (lits[j].var_id < lits[i].var_id) {
                lc_lit_t tmp = lits[i]; lits[i] = lits[j]; lits[j] = tmp;
            }
        }
    }
    int  needed = n_lits * 2;
    bool is_sh  = ((uint8_t)n_lits <= LC_DAG_GLUE_ORD);
    while (LC_DAG_POOL_CAP - dp->alive_cells < needed) {
        if (!lc_dag_prune_leaf(dp, false)) break;
    }
    if (LC_DAG_POOL_CAP - dp->alive_cells < needed) {
        if (!is_sh) return NULL;
        while (LC_DAG_POOL_CAP - dp->alive_cells < needed) {
            if (!lc_dag_prune_leaf(dp, true)) break;
        }
    }
    if (LC_DAG_POOL_CAP - dp->alive_cells < needed) return NULL;

    LiftingCell *cur = lc_dag_get_lit(dp, lits[0].var_id, lits[0].pol_bit);
    if (cur == NULL) return NULL;
    for (int i = 1; i < n_lits; i++) {
        LiftingCell *lit = lc_dag_get_lit(dp, lits[i].var_id, lits[i].pol_bit);
        if (lit == NULL) return NULL;
        cur = lc_dag_get_pair(dp, cur, lit);
        if (cur == NULL) return NULL;
    }
    if ((cur->flags & LC_FLAG_CONFLICT_TERM) == 0u) {
        cur->flags      |= LC_FLAG_CONFLICT_TERM;
        cur->w           = (lc_c16_t){0, LC_Q14_NOT};
        cur->meta_energy = is_sh ? 64 : 24;
        dp->active_terms++;
    } else {
        cur->idle_counter = 0;
        cur->meta_energy += 16;
    }
    return cur;
}

int lc_dag_extract_lits(const LiftingCell *term, lc_lit_t out_lits[LC_DAG_MAX_LITS]) {
    int n = 0;
    const LiftingCell *cur = term;
    while (cur != NULL && n < LC_DAG_MAX_LITS) {
        if (cur->order == 1) {
            out_lits[n].var_id  = cur->var_id;
            out_lits[n].pol_bit = cur->var_mask;
            n++;
            break;
        }
        if (cur->in_g != NULL) {
            out_lits[n].var_id  = cur->in_g->var_id;
            out_lits[n].pol_bit = cur->in_g->var_mask;
            n++;
        }
        cur = cur->in_s;
    }
    return n;
}

/* --- Layer 4: Minimal Sufficient Reason + 1-UIP CDCL Implementation --- */
void lc_cdcl_init(lc_cdcl_graph_t *g) {
    memset(g, 0, sizeof(*g));
    g->act_inc = 16;
    lc_dag_init(&g->cache);
}

int lc_cdcl_add_var(lc_cdcl_graph_t *g, int8_t obs_val) {
    int id = g->num_vars++;
    lc_phase_var_t *v = &g->vars[id];
    v->saved_phase = -1;
    v->trail_pos   = -1;
    if (obs_val < 0) {
        v->u     = LC_Q14_HALF;
        v->conf  = 0;
        v->level = -1;
    } else {
        v->u     = (obs_val != 0) ? LC_Q14_ONE : 0;
        v->conf  = LC_Q14_ONE;
        v->level = 0;
    }
    return id;
}

static void lc_cdcl_link(lc_cdcl_graph_t *g, int vid, int gi) {
    lc_phase_var_t *v = &g->vars[vid];
    if (v->level == 0) return;
    for (uint8_t i = 0; i < v->inc_cnt; i++) {
        if (v->inc_gates[i] == (uint16_t)gi) return;
    }
    if (v->inc_cnt < LC_CDCL_MAX_INC) v->inc_gates[v->inc_cnt++] = (uint16_t)gi;
}

void lc_cdcl_add_gate(lc_cdcl_graph_t *g, lc_currying_net_t *net,
                      int in0, int in1, int in2, int out_v) {
    int gi = g->num_gates++;
    lc_lifting_gate_t *gt = &g->gates[gi];
    gt->net       = net;
    gt->in_var[0] = in0;
    gt->in_var[1] = in1;
    gt->in_var[2] = in2;
    gt->out_var   = out_v;
    lc_cdcl_link(g, in0, gi);
    lc_cdcl_link(g, in1, gi);
    lc_cdcl_link(g, in2, gi);
    lc_cdcl_link(g, out_v, gi);

    for (int c = 0; c < net->num_cells; c++) {
        if (net->cells[c].in_g == NULL) continue;
        uint8_t mask = net->cells[c].var_mask;
        for (uint8_t k = 0; k < net->num_vars; k++) {
            if ((mask & (1u << k)) != 0u) g->vars[gt->in_var[k]].bond_deg++;
        }
    }
}

static inline void lc_gq_push(lc_cdcl_graph_t *g, int gi) {
    if (g->in_gate_q[gi]) return;
    g->in_gate_q[gi] = true;
    g->gate_q[g->gq_tail] = (uint16_t)gi;
    g->gq_tail = (g->gq_tail + 1) % LC_CDCL_MAX_GATES;
}

static void lc_cdcl_assign(lc_cdcl_graph_t *g, int vid, bool bit, int level, bool is_dec,
                           const uint16_t *reas, int rlen) {
    lc_phase_var_t *pv = &g->vars[vid];
    pv->u           = bit ? LC_Q14_ONE : 0;
    pv->conf        = (level == 0) ? LC_Q14_ONE : (is_dec ? LC_Q14_HALF : (LC_Q14_ONE - 64));
    pv->level       = (int16_t)level;
    pv->is_decision = is_dec;
    pv->trail_pos   = (int16_t)g->trail_len;
    pv->reason_len  = (uint8_t)rlen;
    for (int i = 0; i < rlen; i++) pv->reason_vars[i] = reas[i];
    g->trail[g->trail_len++] = (uint16_t)vid;
    for (uint8_t i = 0; i < pv->inc_cnt; i++) lc_gq_push(g, (int)pv->inc_gates[i]);
}

static void lc_cdcl_undo(lc_cdcl_graph_t *g, int target_lv) {
    int lim = g->trail_lim[target_lv + 1];
    int wp  = lim;
    while (g->gq_head != g->gq_tail) {
        uint16_t gi = g->gate_q[g->gq_head];
        g->gq_head = (g->gq_head + 1) % LC_CDCL_MAX_GATES;
        g->in_gate_q[gi] = false;
    }
    for (int i = lim; i < g->trail_len; i++) {
        int vid = (int)g->trail[i];
        lc_phase_var_t *pv = &g->vars[vid];
        if (pv->level <= target_lv) {
            g->trail[wp] = (uint16_t)vid;
            pv->trail_pos = (int16_t)wp;
            wp++;
        } else {
            pv->saved_phase = (pv->u >= LC_Q14_HALF) ? 1 : 0;
            pv->u           = LC_Q14_HALF;
            pv->conf        = 0;
            pv->level       = -1;
            pv->trail_pos   = -1;
            pv->is_decision = false;
            pv->reason_len  = 0;
        }
    }
    g->trail_len = wp;
    g->cur_level = target_lv;
    for (int i = lim; i < wp; i++) {
        const lc_phase_var_t *pv = &g->vars[g->trail[i]];
        for (uint8_t k = 0; k < pv->inc_cnt; k++) lc_gq_push(g, (int)pv->inc_gates[k]);
    }
}

static bool lc_gate_can_complete(const lc_cdcl_graph_t *g, const lc_lifting_gate_t *gt,
                                 int fvar, bool fval, const int cand[4], const bool keep[4], int nc) {
    int  term[4] = { gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var };
    int  fvars[4], nf = 0;
    bool fis[4], fvl[4];
    for (int k = 0; k < 4; k++) {
        int vid = term[k];
        if (vid == fvar) { fis[k] = true; fvl[k] = fval; continue; }
        if (g->vars[vid].conf > 0 && g->vars[vid].level == 0) {
            fis[k] = true; fvl[k] = (g->vars[vid].u >= LC_Q14_HALF); continue;
        }
        bool kept = false;
        for (int c = 0; c < nc; c++) if (cand[c] == vid && keep[c]) { kept = true; break; }
        if (kept) {
            fis[k] = true; fvl[k] = (g->vars[vid].u >= LC_Q14_HALF);
        } else {
            fis[k] = false; fvl[k] = false;
            bool dup = false;
            for (int f = 0; f < nf; f++) if (fvars[f] == vid) dup = true;
            if (!dup) fvars[nf++] = vid;
        }
    }
    for (int sub = 0; sub < (1 << nf); sub++) {
        bool bv[4];
        for (int k = 0; k < 4; k++) {
            if (fis[k]) bv[k] = fvl[k];
            else {
                int pos = 0;
                while (pos < nf && fvars[pos] != term[k]) pos++;
                bv[k] = ((sub >> pos) & 1) != 0;
            }
        }
        bool bi[3] = { bv[0], bv[1], bv[2] };
        if (lc_net_eval_discrete_err(gt->net, bi, bv[3]) == 0) return true;
    }
    return false;
}

static int lc_extract_min_reason(const lc_cdcl_graph_t *g, const lc_lifting_gate_t *gt,
                                 int dvar, bool dval, uint16_t out_r[4]) {
    int term[4] = { gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var };
    int cand[4], nc = 0;
    for (int k = 0; k < 4; k++) {
        int vid = term[k];
        if (vid == dvar || g->vars[vid].conf == 0 || g->vars[vid].level <= 0) continue;
        bool dup = false;
        for (int j = 0; j < nc; j++) if (cand[j] == vid) dup = true;
        if (!dup) cand[nc++] = vid;
    }
    for (int i = 0; i < nc; i++) {
        for (int j = i + 1; j < nc; j++) {
            if (g->vars[cand[j]].level > g->vars[cand[i]].level) {
                int tmp = cand[i]; cand[i] = cand[j]; cand[j] = tmp;
            }
        }
    }
    bool keep[4] = { true, true, true, true };
    for (int i = 0; i < nc; i++) {
        keep[i] = false;
        if (lc_gate_can_complete(g, gt, dvar, !dval, cand, keep, nc)) keep[i] = true;
    }
    int n = 0;
    for (int i = 0; i < nc; i++) if (keep[i]) out_r[n++] = (uint16_t)cand[i];
    return n;
}

static int lc_cdcl_prop_cache(lc_cdcl_graph_t *g, uint16_t cvars[LC_CDCL_MAX_REAS],
                              int *clen, lc_cdcl_stats_t *st) {
    if (g->cache.active_terms == 0) return 0;
    int prog = 0;
    for (int i = 0; i < LC_DAG_POOL_CAP; i++) {
        LiftingCell *term = &g->cache.pool[i];
        if ((term->flags & (LC_FLAG_ALIVE | LC_FLAG_CONFLICT_TERM)) !=
            (LC_FLAG_ALIVE | LC_FLAG_CONFLICT_TERM)) continue;
        if (term->idle_counter < 255u) term->idle_counter++;

        lc_lit_t lits[LC_DAG_MAX_LITS];
        int n = lc_dag_extract_lits(term, lits);
        int uidx = -1, ucnt = 0;
        bool sat = false;
        for (int k = 0; k < n; k++) {
            const lc_phase_var_t *pv = &g->vars[lits[k].var_id];
            if (pv->conf == 0) { ucnt++; uidx = k; if (ucnt > 1) break; }
            else if (((pv->u >= LC_Q14_HALF) ? 1u : 0u) != lits[k].pol_bit) { sat = true; break; }
        }
        if (sat || ucnt > 1) continue;
        if (ucnt == 0) {
            int cn = 0;
            for (int k = 0; k < n; k++) {
                if (g->vars[lits[k].var_id].level > 0) cvars[cn++] = lits[k].var_id;
            }
            *clen = cn;
            term->idle_counter = 0;
            term->meta_energy += 8;
            return -1;
        } else {
            int vid = (int)lits[uidx].var_id;
            bool fbit = (lits[uidx].pol_bit == 0u);
            uint16_t reas[LC_DAG_MAX_LITS];
            int rl = 0, mlv = 0;
            for (int k = 0; k < n; k++) {
                if (k == uidx) continue;
                int lv = g->vars[lits[k].var_id].level;
                if (lv > 0) { reas[rl++] = lits[k].var_id; if (lv > mlv) mlv = lv; }
            }
            lc_cdcl_assign(g, vid, fbit, mlv, false, reas, rl);
            term->idle_counter = 0;
            term->meta_energy += 6;
            if (st != NULL) st->cache_props++;
            prog = 1;
        }
    }
    return prog;
}

static int lc_cdcl_propagate(lc_cdcl_graph_t *g, uint16_t cvars[LC_CDCL_MAX_REAS],
                             int *clen, lc_cdcl_stats_t *st) {
    for (;;) {
        while (g->gq_head != g->gq_tail) {
            int gi = (int)g->gate_q[g->gq_head];
            g->gq_head = (g->gq_head + 1) % LC_CDCL_MAX_GATES;
            g->in_gate_q[gi] = false;

            lc_lifting_gate_t *gt = &g->gates[gi];
            int term[4] = { gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var };
            int uvars[4], nu = 0, kcnt = 0;
            for (int k = 0; k < 4; k++) {
                int vid = term[k];
                if (g->vars[vid].conf > 0) { kcnt++; continue; }
                bool dup = false;
                for (int j = 0; j < nu; j++) if (uvars[j] == vid) dup = true;
                if (!dup) uvars[nu++] = vid;
            }
            if (kcnt == 0 && nu == 4) continue;
            if (nu == 0) {
                bool bi[3] = {
                    g->vars[term[0]].u >= LC_Q14_HALF,
                    g->vars[term[1]].u >= LC_Q14_HALF,
                    g->vars[term[2]].u >= LC_Q14_HALF
                };
                if (lc_net_eval_discrete_err(gt->net, bi, g->vars[term[3]].u >= LC_Q14_HALF) != 0) {
                    uint16_t rt[4];
                    int rn = lc_extract_min_reason(g, gt, -1, false, rt);
                    for (int i = 0; i < rn; i++) cvars[i] = rt[i];
                    *clen = rn;
                    return -1;
                }
                continue;
            }
            int valid = 0;
            bool s0[4] = {0}, s1[4] = {0};
            for (int sub = 0; sub < (1 << nu); sub++) {
                bool bv[4];
                for (int k = 0; k < 4; k++) {
                    int vid = term[k];
                    if (g->vars[vid].conf > 0) bv[k] = (g->vars[vid].u >= LC_Q14_HALF);
                    else {
                        int pos = 0;
                        while (pos < nu && uvars[pos] != vid) pos++;
                        bv[k] = ((sub >> pos) & 1) != 0;
                    }
                }
                bool bi[3] = { bv[0], bv[1], bv[2] };
                if (lc_net_eval_discrete_err(gt->net, bi, bv[3]) == 0) {
                    valid++;
                    for (int j = 0; j < nu; j++) {
                        if (((sub >> j) & 1) != 0) s1[j] = true; else s0[j] = true;
                    }
                }
            }
            if (valid == 0) {
                uint16_t rt[4];
                int rn = lc_extract_min_reason(g, gt, -1, false, rt);
                for (int i = 0; i < rn; i++) cvars[i] = rt[i];
                *clen = rn;
                return -1;
            }
            for (int j = 0; j < nu; j++) {
                if (s0[j] != s1[j]) {
                    uint16_t rt[4];
                    int rn = lc_extract_min_reason(g, gt, uvars[j], s1[j], rt);
                    int lv = 0;
                    for (int i = 0; i < rn; i++) if (g->vars[rt[i]].level > lv) lv = g->vars[rt[i]].level;
                    lc_cdcl_assign(g, uvars[j], s1[j], lv, false, rt, rn);
                }
            }
        }
        int cr = lc_cdcl_prop_cache(g, cvars, clen, st);
        if (cr < 0) return -1;
        if (cr > 0) continue;
        break;
    }
    for (int v = 0; v < g->num_vars; v++) if (g->vars[v].conf == 0) return 0;
    return 1;
}

static bool lc_cdcl_analyze_1uip(lc_cdcl_graph_t *g, const uint16_t cvars[LC_CDCL_MAX_REAS],
                                 int clen, lc_lit_t out_cl[LC_CDCL_MAX_REAS],
                                 int *out_len, int *out_bt) {
    if (clen == 0) return false;
    int tlv = 0;
    for (int i = 0; i < clen; i++) if (g->vars[cvars[i]].level > tlv) tlv = g->vars[cvars[i]].level;
    if (tlv == 0) return false;

    bool seen[LC_CDCL_MAX_VARS];
    memset(seen, 0, (size_t)g->num_vars * sizeof(bool));
    uint16_t lvars[LC_CDCL_MAX_REAS];
    int nl = 0, at_cnt = 0;

    for (int i = 0; i < clen; i++) {
        int v = (int)cvars[i], lv = g->vars[v].level;
        if (lv <= 0 || seen[v]) continue;
        seen[v] = true;
        g->vars[v].activity += g->act_inc;
        if (lv == tlv) at_cnt++;
        else if (nl < LC_CDCL_MAX_REAS - 1) lvars[nl++] = (uint16_t)v;
    }

    int tidx = g->trail_len - 1;
    while (at_cnt > 1) {
        while (tidx >= 0 && !(seen[g->trail[tidx]] && g->vars[g->trail[tidx]].level == tlv)) tidx--;
        if (tidx < 0) break;
        int p = (int)g->trail[tidx];
        const lc_phase_var_t *pv = &g->vars[p];
        if (pv->is_decision || pv->reason_len == 0) break;
        tidx--;
        seen[p] = false;
        at_cnt--;
        for (uint8_t i = 0; i < pv->reason_len; i++) {
            int q = (int)pv->reason_vars[i], lv = g->vars[q].level;
            if (lv <= 0 || seen[q]) continue;
            seen[q] = true;
            g->vars[q].activity += g->act_inc;
            if (lv == tlv) at_cnt++;
            else if (nl < LC_CDCL_MAX_REAS - 1) lvars[nl++] = (uint16_t)q;
        }
    }

    int uip = -1;
    while (tidx >= 0) {
        int p = (int)g->trail[tidx--];
        if (seen[p] && g->vars[p].level == tlv) { uip = p; break; }
    }
    if (uip < 0) return false;

    out_cl[0].var_id  = (uint16_t)uip;
    out_cl[0].pol_bit = (g->vars[uip].u >= LC_Q14_HALF) ? 1u : 0u;
    int bt = 0;
    for (int i = 0; i < nl; i++) {
        int q = (int)lvars[i];
        out_cl[1 + i].var_id  = (uint16_t)q;
        out_cl[1 + i].pol_bit = (g->vars[q].u >= LC_Q14_HALF) ? 1u : 0u;
        if (g->vars[q].level > bt) bt = g->vars[q].level;
    }
    *out_len = 1 + nl;
    *out_bt  = bt;
    return true;
}

static int lc_cdcl_select_pivot(const lc_cdcl_graph_t *g) {
    int t2[LC_CDCL_MAX_VARS], fr[LC_CDCL_MAX_VARS];
    memset(t2, 0, (size_t)g->num_vars * sizeof(int));
    memset(fr, 0, (size_t)g->num_vars * sizeof(int));
    for (int gi = 0; gi < g->num_gates; gi++) {
        const lc_lifting_gate_t *gt = &g->gates[gi];
        int term[4] = { gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var };
        int u[4], nu = 0, kc = 0;
        for (int k = 0; k < 4; k++) {
            if (g->vars[term[k]].conf > 0) kc++;
            else {
                bool dup = false;
                for (int j = 0; j < nu; j++) if (u[j] == term[k]) dup = true;
                if (!dup) u[nu++] = term[k];
            }
        }
        if (nu == 0 || kc == 0) continue;
        for (int j = 0; j < nu; j++) { fr[u[j]] += kc; if (nu == 2) t2[u[j]]++; }
    }
    int max_b = 0;
    for (int v = 0; v < g->num_vars; v++) {
        if (g->vars[v].conf == 0 && g->vars[v].bond_deg > max_b) max_b = g->vars[v].bond_deg;
    }
    int hub_min = (max_b * 3) / 4;
    int bv = -1;
    int64_t bs = -1;
    for (int v = 0; v < g->num_vars; v++) {
        if (g->vars[v].conf > 0 || g->vars[v].bond_deg < hub_min) continue;
        int64_t sc = (int64_t)t2[v] * 64 + (int64_t)g->vars[v].bond_deg * 4 +
                     (int64_t)fr[v] + (int64_t)g->vars[v].activity * 4;
        if (sc > bs) { bs = sc; bv = v; }
    }
    return bv;
}

bool lc_cdcl_solve(lc_cdcl_graph_t *g, int max_conflicts,
                   bool out_bits[LC_CDCL_MAX_VARS], lc_cdcl_stats_t *st) {
    if (st != NULL) memset(st, 0, sizeof(*st));
    for (int gi = 0; gi < g->num_gates; gi++) lc_gq_push(g, gi);

    for (;;) {
        uint16_t cvars[LC_CDCL_MAX_REAS];
        int clen = 0;
        int res  = lc_cdcl_propagate(g, cvars, &clen, st);
        if (res == 1) {
            for (int v = 0; v < g->num_vars; v++) out_bits[v] = (g->vars[v].u >= LC_Q14_HALF);
            if (st != NULL) {
                st->cache_spawned = g->cache.spawned_cells;
                st->cache_pruned  = g->cache.pruned_cells;
            }
            return true;
        }
        if (res == 0) {
            int piv = lc_cdcl_select_pivot(g);
            if (piv < 0 || g->cur_level >= LC_CDCL_MAX_DEPTH) return false;
            int8_t fbit = (g->vars[piv].saved_phase >= 0) ? g->vars[piv].saved_phase : 1;
            g->cur_level++;
            g->trail_lim[g->cur_level] = g->trail_len;
            g->dec_var[g->cur_level]   = piv;
            lc_cdcl_assign(g, piv, fbit == 1, g->cur_level, true, NULL, 0);
            if (st != NULL) st->branches++;
            continue;
        }
        if (st != NULL) st->conflicts++;
        if (st != NULL && st->conflicts > max_conflicts) return false;

        lc_lit_t learned[LC_CDCL_MAX_REAS];
        int llen = 0, bt = 0;
        if (!lc_cdcl_analyze_1uip(g, cvars, clen, learned, &llen, &bt)) {
            if (st != NULL) {
                st->cache_spawned = g->cache.spawned_cells;
                st->cache_pruned  = g->cache.pruned_cells;
            }
            return false;
        }
        if (llen >= 2 && llen <= LC_DAG_MAX_LITS) {
            (void)lc_dag_add_term(&g->cache, learned, llen);
        }
        if (bt < g->cur_level - 1 && st != NULL) st->backjumps++;
        lc_cdcl_undo(g, bt);

        uint16_t ur[LC_CDCL_MAX_REAS];
        for (int i = 0; i < llen - 1; i++) ur[i] = learned[1 + i].var_id;
        lc_cdcl_assign(g, (int)learned[0].var_id, learned[0].pol_bit == 0u, bt, false, ur, llen - 1);
    }
}

/* --- Layer 5: Continuous-Phase Soft-Decision ECC Implementation --- */
static int16_t lc_lut_tanh[LC_LLR_MAX + 1];
static int16_t lc_lut_atanh[LC_Q14_ONE + 1];

static double lc_exp_pos(double x) {
    double xs = x * 0.125, sum = 1.0, term = 1.0;
    for (int k = 1; k <= 14; k++) { term *= xs / (double)k; sum += term; }
    sum *= sum; sum *= sum; sum *= sum;
    return sum;
}

void lc_ecc_init_luts(void) {
    for (int L = 0; L <= LC_LLR_MAX; L++) {
        double e2x = lc_exp_pos((double)L / (double)LC_LLR_SCALE);
        int32_t q  = (int32_t)(((e2x - 1.0) / (e2x + 1.0)) * (double)LC_Q14_ONE + 0.5);
        lc_lut_tanh[L] = (int16_t)((q >= LC_Q14_ONE) ? (LC_Q14_ONE - 1) : q);
    }
    lc_lut_atanh[0] = 0;
    for (int q = 1; q <= LC_Q14_ONE; q++) {
        if (q >= lc_lut_tanh[LC_LLR_MAX]) { lc_lut_atanh[q] = LC_LLR_MAX; continue; }
        double tgt = (double)q / (double)LC_Q14_ONE;
        double lo = 0.0, hi = (double)LC_LLR_MAX / (double)LC_LLR_SCALE;
        for (int it = 0; it < 36; it++) {
            double mid = 0.5 * (lo + hi), e = lc_exp_pos(mid);
            if ((e - 1.0) / (e + 1.0) < tgt) lo = mid; else hi = mid;
        }
        int32_t L = (int32_t)(0.5 * (lo + hi) * (double)LC_LLR_SCALE + 0.5);
        lc_lut_atanh[q] = (int16_t)((L > LC_LLR_MAX) ? LC_LLR_MAX : L);
    }
}

static inline int32_t lc_clamp_llr(int32_t v) {
    return (v < -LC_LLR_MAX) ? -LC_LLR_MAX : ((v > LC_LLR_MAX) ? LC_LLR_MAX : v);
}

static inline int16_t lc_llr_to_q14(int32_t llr) {
    if (llr >= 0) return lc_lut_tanh[(llr > LC_LLR_MAX) ? LC_LLR_MAX : llr];
    int32_t m = -llr;
    return (int16_t)(-lc_lut_tanh[(m > LC_LLR_MAX) ? LC_LLR_MAX : m]);
}

static inline int32_t lc_q14_to_llr(int16_t mu) {
    if (mu >= 0) return (int32_t)lc_lut_atanh[(mu > LC_Q14_ONE) ? LC_Q14_ONE : mu];
    int32_t m = -(int32_t)mu;
    return -(int32_t)lc_lut_atanh[(m > LC_Q14_ONE) ? LC_Q14_ONE : m];
}

void lc_ecc_init(lc_ecc_graph_t *g, lc_currying_net_t *xor3_net, int n_vars) {
    memset(g, 0, sizeof(*g));
    g->xor3_net = xor3_net;
    g->n_vars   = n_vars;
    for (int i = 0; i < LC_ECC_MAX_DEG; i++) {
        g->p_pref[i].in_v = &g->p_zero;
        g->p_suff[i].in_v = &g->p_zero;
        g->p_ext[i].in_v  = &g->p_zero;
    }
}

void lc_ecc_add_check(lc_ecc_graph_t *g, const int *vars, int deg) {
    int cid = g->n_checks++;
    lc_ecc_check_t *c = &g->checks[cid];
    c->deg = deg;
    for (int k = 0; k < deg; k++) {
        int vid = vars[k], eid = g->n_edges++;
        c->vars[k]  = vid;
        c->edges[k] = eid;
        g->edges[eid].var_id = vid;
        g->edges[eid].chk_id = cid;
        g->vadj[vid].edges[g->vadj[vid].deg++] = eid;
    }
}

int lc_ecc_syndrome_errors(lc_ecc_graph_t *g, const uint8_t *bits) {
    int err = 0;
    for (int cid = 0; cid < g->n_checks; cid++) {
        const lc_ecc_check_t *c = &g->checks[cid];
        if (c->deg == 4) {
            bool bi[3] = { bits[c->vars[0]] != 0, bits[c->vars[1]] != 0, bits[c->vars[2]] != 0 };
            if (lc_net_eval_discrete_err(g->xor3_net, bi, bits[c->vars[3]] != 0) != 0) err++;
        } else {
            bool acc = false;
            for (int k = 0; k < c->deg; k++) acc ^= (bits[c->vars[k]] != 0);
            if (acc) err++;
        }
    }
    return err;
}

int lc_ecc_decode_erasure(lc_ecc_graph_t *g, const int8_t *obs, uint8_t *out_bits) {
    int8_t st[LC_ECC_MAX_N];
    int unk = 0;
    for (int v = 0; v < g->n_vars; v++) { st[v] = obs[v]; if (st[v] < 0) unk++; }
    bool prog = true;
    while (prog && unk > 0) {
        prog = false;
        for (int cid = 0; cid < g->n_checks; cid++) {
            const lc_ecc_check_t *c = &g->checks[cid];
            int up = -1, uc = 0;
            uint8_t par = 0;
            for (int k = 0; k < c->deg; k++) {
                if (st[c->vars[k]] < 0) { uc++; up = c->vars[k]; }
                else par ^= (uint8_t)(st[c->vars[k]] & 1);
            }
            if (uc == 1) { st[up] = (int8_t)par; unk--; prog = true; }
        }
    }
    for (int v = 0; v < g->n_vars; v++) out_bits[v] = (st[v] > 0) ? 1u : 0u;
    return unk;
}

int lc_ecc_decode_soft(lc_ecc_graph_t *g, const int32_t *ch_llr,
                       int max_iters, bool damping, uint8_t *out_bits) {
    for (int v = 0; v < g->n_vars; v++) out_bits[v] = (ch_llr[v] < 0) ? 1u : 0u;
    if (lc_ecc_syndrome_errors(g, out_bits) == 0) return 0;

    for (int e = 0; e < g->n_edges; e++) {
        g->edges[e].msg_v2c = lc_clamp_llr(ch_llr[g->edges[e].var_id]);
        g->edges[e].msg_c2v = 0;
    }
    for (int it = 1; it <= max_iters; it++) {
        for (int cid = 0; cid < g->n_checks; cid++) {
            const lc_ecc_check_t *chk = &g->checks[cid];
            int d = chk->deg;
            for (int k = 0; k < d; k++) {
                g->p_in[k].s_eff = (lc_c16_t){ lc_llr_to_q14(g->edges[chk->edges[k]].msg_v2c), 0 };
            }
            g->p_pref[0].s_eff = (lc_c16_t){ LC_Q14_ONE, 0 };
            for (int k = 1; k < d; k++) {
                g->p_pref[k].in_s = &g->p_pref[k - 1];
                g->p_pref[k].in_g = &g->p_in[k - 1];
                lc_cell_bottom_up(&g->p_pref[k]);
            }
            g->p_suff[d - 1].s_eff = (lc_c16_t){ LC_Q14_ONE, 0 };
            for (int k = d - 2; k >= 0; k--) {
                g->p_suff[k].in_s = &g->p_suff[k + 1];
                g->p_suff[k].in_g = &g->p_in[k + 1];
                lc_cell_bottom_up(&g->p_suff[k]);
            }
            for (int k = 0; k < d; k++) {
                g->p_ext[k].in_s = &g->p_pref[k];
                g->p_ext[k].in_g = &g->p_suff[k];
                lc_cell_bottom_up(&g->p_ext[k]);
                int32_t nl = lc_q14_to_llr(g->p_ext[k].s_eff.re);
                int eid    = chk->edges[k];
                if (damping && it > 1) nl = (nl * 3 + g->edges[eid].msg_c2v) >> 2;
                g->edges[eid].msg_c2v = lc_clamp_llr(nl);
            }
        }
        for (int v = 0; v < g->n_vars; v++) {
            const lc_ecc_vadj_t *va = &g->vadj[v];
            int32_t tot = ch_llr[v];
            for (int k = 0; k < va->deg; k++) tot += g->edges[va->edges[k]].msg_c2v;
            out_bits[v] = (tot < 0) ? 1u : 0u;
            for (int k = 0; k < va->deg; k++) {
                int eid = va->edges[k];
                g->edges[eid].msg_v2c = lc_clamp_llr(tot - g->edges[eid].msg_c2v);
            }
        }
        if (lc_ecc_syndrome_errors(g, out_bits) == 0) return it;
    }
    return -1;
}

/* --- Layer 6: Scalar Reference Kernels (常に提供・パリティ検証用) --- */
LC_NOINLINE void lc_log_aosoa8_bind_only_scalar(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            lc_logc16_t se = lc_fhrr_bind((lc_logc16_t){b->g_rho[k], b->g_theta[k]},
                                          (lc_logc16_t){b->s_rho[k], b->s_theta[k]});
            lc_logc16_t pl = lc_fhrr_bind((lc_logc16_t){b->w_rho[k], b->w_theta[k]}, se);
            b->se_rho[k] = se.rho; b->se_theta[k] = se.theta;
            b->p_rho[k]  = pl.rho; b->p_theta[k]  = pl.theta;
        }
    }
}

LC_NOINLINE void lc_log_aosoa8_dual_step_scalar(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            lc_logc16_t w   = { b->w_rho[k], b->w_theta[k] };
            lc_logc16_t se  = lc_fhrr_bind((lc_logc16_t){b->g_rho[k], b->g_theta[k]},
                                           (lc_logc16_t){b->s_rho[k], b->s_theta[k]});
            lc_logc16_t pl  = lc_fhrr_bind(w, se);
            lc_logc16_t unb = lc_fhrr_unbind(pl, w);
            lc_c16_t    pc  = lc_log_to_cart(pl.rho, pl.theta);
            lc_c16_t    r   = lc_c16_sub_mod((lc_c16_t){b->v_re[k], b->v_im[k]}, pc);
            lc_c16_t    res = lc_c16_add_mod(r, pc);

            b->se_rho[k]  = se.rho;  b->se_theta[k]  = se.theta;
            b->p_rho[k]   = pl.rho;  b->p_theta[k]   = pl.theta;
            b->unb_rho[k] = unb.rho; b->unb_theta[k] = unb.theta;
            b->p_re[k]    = pc.re;   b->p_im[k]      = pc.im;
            b->r_re[k]    = r.re;    b->r_im[k]      = r.im;
            b->res_re[k]  = res.re;  b->res_im[k]    = res.im;
        }
    }
}

LC_NOINLINE void lc_chart_aosoa8_cart_to_log_scalar(lc_chart_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_chart_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            lc_logc16_t z = lc_cart_to_log(b->re[k], b->im[k]);
            b->rho[k]   = z.rho;
            b->theta[k] = z.theta;
        }
    }
}

LC_NOINLINE uint64_t lc_holo_aosoa8_eval_2morph_scalar(lc_holo_block8_t *blocks, int n_blocks,
                                                       bool apply_twist) {
    uint64_t phase_l1 = 0;
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_holo_block8_t *b = &blocks[bi];
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (int k = 0; k < 8; k++) {
            int16_t  r2 = apply_twist ? lc_sat_add_s16(b->z2_rho[k], b->tw_rho[k]) : b->z2_rho[k];
            uint16_t t2 = apply_twist ? (uint16_t)(b->z2_theta[k] + b->tw_theta[k]) : b->z2_theta[k];
            int16_t  dr = lc_sat_sub_s16(b->z1_rho[k], r2);
            int16_t  dt = (int16_t)((uint16_t)(b->z1_theta[k] - t2));
            b->out_drho[k]   = dr;
            b->out_dtheta[k] = dt;
            phase_l1 += (uint64_t)((dt == -32768) ? 32768 : ((dt >= 0) ? dt : -dt));
        }
    }
    return phase_l1;
}

/* --- Layer 6: ARM64 NEON Optimized Implementations --- */
#if LC_HAS_ARM_NEON

static inline int16x8_t lc_sub_mod_x8(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vsubq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b)));
}
static inline int16x8_t lc_add_mod_x8(int16x8_t a, int16x8_t b) {
    return vreinterpretq_s16_u16(vaddq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b)));
}
static inline int32x4_t lc_quant_q14_x4(int32x4_t acc) {
    uint32x4_t neg = vcltq_s32(acc, vdupq_n_s32(0));
    int32x4_t  mag = vshrq_n_s32(vqaddq_s32(vqabsq_s32(acc), vdupq_n_s32(LC_Q14_HALF)), 14);
    return vbslq_s32(neg, vnegq_s32(mag), mag);
}
static inline void lc_mul_q14_x8(int16x8_t ar, int16x8_t ai, int16x8_t br, int16x8_t bi,
                                 int16x8_t *or_, int16x8_t *oi) {
    int32x4_t rlo = vmlsl_s16(vmull_s16(vget_low_s16(ar), vget_low_s16(br)),
                              vget_low_s16(ai), vget_low_s16(bi));
    int32x4_t rhi = vmlsl_high_s16(vmull_high_s16(ar, br), ai, bi);
    int32x4_t ilo = vqaddq_s32(vmull_s16(vget_low_s16(ar), vget_low_s16(bi)),
                               vmull_s16(vget_low_s16(ai), vget_low_s16(br)));
    int32x4_t ihi = vqaddq_s32(vmull_high_s16(ar, bi), vmull_high_s16(ai, br));
    *or_ = vqmovn_high_s32(vqmovn_s32(lc_quant_q14_x4(rlo)), lc_quant_q14_x4(rhi));
    *oi  = vqmovn_high_s32(vqmovn_s32(lc_quant_q14_x4(ilo)), lc_quant_q14_x4(ihi));
}
static inline int32x4_t lc_add_sr14_x4(int32x4_t p0, int32x4_t p1) {
    int32x4_t  sum = vaddq_s32(p0, p1);
    uint32x4_t ovf = vceqq_s32(sum, vdupq_n_s32(INT32_MIN));
    uint32x4_t neg = vbicq_u32(vcltq_s32(sum, vdupq_n_s32(0)), ovf);
    int32x4_t  mag = vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(vabsq_s32(sum)), 14));
    return vbslq_s32(neg, vnegq_s32(mag), mag);
}
static inline int32x4_t lc_qr_mul14_x4(uint32x4_t q, uint32x4_t r, int32x4_t tot, int32x4_t fac) {
    uint32x4_t neg = vcltq_s32(veorq_s32(tot, fac), vdupq_n_s32(0));
    uint32x4_t G   = vreinterpretq_u32_s32(vabsq_s32(fac));
    int32x4_t  mag = vreinterpretq_s32_u32(vmlaq_u32(vshrq_n_u32(vmulq_u32(r, G), 14), q, G));
    return vbslq_s32(neg, vnegq_s32(mag), mag);
}
static inline int16x4_t lc_gsum_x4(int32x4_t p0, int32x4_t p1, int32x4_t nsh) {
    int32x4_t  sum = vaddq_s32(p0, p1);
    uint32x4_t ovf = vceqq_s32(sum, vdupq_n_s32(INT32_MIN));
    uint32x4_t neg = vbicq_u32(vcltq_s32(sum, vdupq_n_s32(0)), ovf);
    uint32x4_t au  = vreinterpretq_u32_s32(vabsq_s32(sum));
    int32x4_t  mag = vreinterpretq_s32_u32(vmaxq_u32(vshlq_u32(au, nsh), vminq_u32(au, vdupq_n_u32(1))));
    return vmovn_s32(vbslq_s32(neg, vnegq_s32(mag), mag));
}
static inline int16x4_t lc_gdiff_x4(int32x4_t d, int32x4_t nsh) {
    uint32x4_t neg = vcltq_s32(d, vdupq_n_s32(0));
    uint32x4_t au  = vreinterpretq_u32_s32(vabsq_s32(d));
    int32x4_t  mag = vreinterpretq_s32_u32(vmaxq_u32(vshlq_u32(au, nsh), vminq_u32(au, vdupq_n_u32(1))));
    return vmovn_s32(vbslq_s32(neg, vnegq_s32(mag), mag));
}
static inline int16x8_t lc_log_add_rho_x8(int16x8_t a, int16x8_t b) {
    const int16x8_t z = vdupq_n_s16(LC_LOG_ZERO);
    uint16x8_t is_z = vorrq_u16(vceqq_s16(a, z), vceqq_s16(b, z));
    return vbslq_s16(is_z, z, vqaddq_s16(a, b));
}
static inline void lc_log_to_cart_x8_neon(const int16_t rho[8], const uint16_t theta[8],
                                          int16x8_t *out_re, int16x8_t *out_im) {
    int16_t tmp_base[8], tmp_cos[8], tmp_sin[8];
    for (int k = 0; k < 8; k++) {
        uint32_t idx = (((uint32_t)theta[k] + (1u << (LC_TRIG_SHIFT - 1))) >> LC_TRIG_SHIFT)
                       & (LC_TRIG_SIZE - 1u);
        tmp_base[k]  = (int16_t)lc_lut_exp2_frac[(uint8_t)(rho[k] & 0xFF)];
        tmp_cos[k]   = lc_lut_cos[idx];
        tmp_sin[k]   = lc_lut_sin[idx];
    }
    int16x8_t  v_e16  = vshrq_n_s16(vld1q_s16(rho), 8);
    uint16x8_t v_base = vreinterpretq_u16_s16(vld1q_s16(tmp_base));
    int16x8_t  v_cos  = vld1q_s16(tmp_cos);
    int16x8_t  v_sin  = vld1q_s16(tmp_sin);

    const int32x4_t min_e = vdupq_n_s32(-16), max_e = vdupq_n_s32(2), max15 = vdupq_n_s32(32767);
    int32x4_t e_lo = vmaxq_s32(vminq_s32(vmovl_s16(vget_low_s16(v_e16)), max_e), min_e);
    int32x4_t e_hi = vmaxq_s32(vminq_s32(vmovl_high_s16(v_e16), max_e), min_e);
    int32x4_t b_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_base)));
    int32x4_t b_hi = vreinterpretq_s32_u32(vmovl_high_u16(v_base));
    int32x4_t m_lo = vminq_s32(vshlq_s32(b_lo, e_lo), max15);
    int32x4_t m_hi = vminq_s32(vshlq_s32(b_hi, e_hi), max15);

    int32x4_t c_lo = vmovl_s16(vget_low_s16(v_cos)), c_hi = vmovl_high_s16(v_cos);
    int32x4_t s_lo = vmovl_s16(vget_low_s16(v_sin)), s_hi = vmovl_high_s16(v_sin);
    *out_re = vqmovn_high_s32(vqmovn_s32(lc_quant_q14_x4(vmulq_s32(m_lo, c_lo))),
                              lc_quant_q14_x4(vmulq_s32(m_hi, c_hi)));
    *out_im = vqmovn_high_s32(vqmovn_s32(lc_quant_q14_x4(vmulq_s32(m_lo, s_lo))),
                              lc_quant_q14_x4(vmulq_s32(m_hi, s_hi)));
}

LC_NOINLINE void lc_aosoa8_forward(lc_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_block8_t *b = &blocks[bi];
        int16x8_t sr = vld1q_s16(b->s_re), si = vld1q_s16(b->s_im);
        int16x8_t gr = vld1q_s16(b->g_re), gi = vld1q_s16(b->g_im);
        int16x8_t vr = vld1q_s16(b->v_re), vi = vld1q_s16(b->v_im);
        int16x8_t wr = vld1q_s16(b->w_re), wi = vld1q_s16(b->w_im);
        int16x8_t ser, sei, pr, pi;
        lc_mul_q14_x8(gr, gi, sr, si, &ser, &sei);
        lc_mul_q14_x8(wr, wi, ser, sei, &pr, &pi);
        int16x8_t rr = lc_sub_mod_x8(vr, pr), ri = lc_sub_mod_x8(vi, pi);
        vst1q_s16(b->s_eff_re, ser); vst1q_s16(b->s_eff_im, sei);
        vst1q_s16(b->p_re, pr);      vst1q_s16(b->p_im, pi);
        vst1q_s16(b->r_re, rr);      vst1q_s16(b->r_im, ri);
        vst1q_s16(b->restored_re, lc_add_mod_x8(rr, pr));
        vst1q_s16(b->restored_im, lc_add_mod_x8(ri, pi));
    }
}

LC_NOINLINE void lc_aosoa8_backward(lc_block8_t *blocks, int n_blocks) {
    const int16x8_t  not16 = vdupq_n_s16(LC_Q14_NOT), half16 = vdupq_n_s16(LC_Q14_HALF);
    const int16x8_t  max16 = vdupq_n_s16(32767);
    const int32x4_t  z32   = vdupq_n_s32(0), fl2048 = vdupq_n_s32(2048);
    const uint32x4_t m14   = vdupq_n_u32(0x3FFFu);
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_block8_t *b = &blocks[bi];
        int16x8_t sr = vld1q_s16(b->s_re), gr = vld1q_s16(b->g_re);
        int16x8_t wr = vld1q_s16(b->w_re), wi = vld1q_s16(b->w_im);
        int16x8_t rr = vld1q_s16(b->r_re), ri = vld1q_s16(b->r_im);
        int16x8_t se = vld1q_s16(b->s_eff_re);
        int16x8_t ei = vbslq_s16(vceqq_s16(ri, not16),
                                 vbslq_s16(vcgtq_s16(se, half16), max16, not16), ri);
        int32x4_t flo = lc_add_sr14_x4(vmull_s16(vget_low_s16(wr), vget_low_s16(rr)),
                                       vmull_s16(vget_low_s16(wi), vget_low_s16(ei)));
        int32x4_t fhi = lc_add_sr14_x4(vmull_high_s16(wr, rr), vmull_high_s16(wi, ei));
        int32x4_t tlo = vaddq_s32(vld1q_s32(&b->adj_in[0]), flo);
        int32x4_t thi = vaddq_s32(vld1q_s32(&b->adj_in[4]), fhi);
        int32x4_t glo = vmovl_s16(vget_low_s16(gr)), ghi = vmovl_high_s16(gr);
        int32x4_t slo = vmovl_s16(vget_low_s16(sr)), shi = vmovl_high_s16(sr);
        uint32x4_t plo = vcgtq_s32(tlo, z32), phi = vcgtq_s32(thi, z32);
        glo = vbslq_s32(plo, vmaxq_s32(glo, fl2048), glo);
        slo = vbslq_s32(plo, vmaxq_s32(slo, fl2048), slo);
        ghi = vbslq_s32(phi, vmaxq_s32(ghi, fl2048), ghi);
        shi = vbslq_s32(phi, vmaxq_s32(shi, fl2048), shi);
        uint32x4_t alo = vreinterpretq_u32_s32(vabsq_s32(tlo));
        uint32x4_t ahi = vreinterpretq_u32_s32(vabsq_s32(thi));
        uint32x4_t qlo = vshrq_n_u32(alo, 14), rlo = vandq_u32(alo, m14);
        uint32x4_t qhi = vshrq_n_u32(ahi, 14), rhi = vandq_u32(ahi, m14);
        vst1q_s32(&b->adj_s[0], lc_qr_mul14_x4(qlo, rlo, tlo, glo));
        vst1q_s32(&b->adj_s[4], lc_qr_mul14_x4(qhi, rhi, thi, ghi));
        vst1q_s32(&b->adj_g[0], vandq_s32(lc_qr_mul14_x4(qlo, rlo, tlo, slo), vld1q_s32(&b->gate_mask[0])));
        vst1q_s32(&b->adj_g[4], vandq_s32(lc_qr_mul14_x4(qhi, rhi, thi, shi), vld1q_s32(&b->gate_mask[4])));
    }
}

LC_NOINLINE void lc_aosoa8_update(lc_block8_t *blocks, int n_blocks, uint8_t lr_shift) {
    const int32x4_t nsh = vdupq_n_s32(-(14 + (int32_t)lr_shift));
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_block8_t *b = &blocks[bi];
        int16x8_t wr = vld1q_s16(b->w_re), wi = vld1q_s16(b->w_im);
        int16x8_t er = vld1q_s16(b->r_re), ei = vld1q_s16(b->r_im);
        int16x8_t sr = vld1q_s16(b->s_eff_re), si = vld1q_s16(b->s_eff_im);
        int16x4_t drlo = lc_gsum_x4(vmull_s16(vget_low_s16(er), vget_low_s16(sr)),
                                    vmull_s16(vget_low_s16(ei), vget_low_s16(si)), nsh);
        int16x4_t dilo = lc_gdiff_x4(vmlsl_s16(vmull_s16(vget_low_s16(ei), vget_low_s16(sr)),
                                               vget_low_s16(er), vget_low_s16(si)), nsh);
        int16x4_t drhi = lc_gsum_x4(vmull_high_s16(er, sr), vmull_high_s16(ei, si), nsh);
        int16x4_t dihi = lc_gdiff_x4(vmlsl_high_s16(vmull_high_s16(ei, sr), er, si), nsh);
        vst1q_s16(b->w_re, lc_add_mod_x8(wr, vcombine_s16(drlo, drhi)));
        vst1q_s16(b->w_im, lc_add_mod_x8(wi, vcombine_s16(dilo, dihi)));
    }
}

LC_NOINLINE void lc_log_aosoa8_bind_only(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
        int16x8_t  sr = vld1q_s16(b->s_rho),   gr = vld1q_s16(b->g_rho),   wr = vld1q_s16(b->w_rho);
        uint16x8_t st = vld1q_u16(b->s_theta), gt = vld1q_u16(b->g_theta), wt = vld1q_u16(b->w_theta);
        int16x8_t  ser = lc_log_add_rho_x8(gr, sr);
        uint16x8_t set = vaddq_u16(gt, st);
        int16x8_t  pr  = lc_log_add_rho_x8(wr, ser);
        uint16x8_t pt  = vaddq_u16(wt, set);
        vst1q_s16(b->se_rho, ser); vst1q_u16(b->se_theta, set);
        vst1q_s16(b->p_rho,  pr);  vst1q_u16(b->p_theta,  pt);
    }
}

LC_NOINLINE void lc_log_aosoa8_dual_step(lc_log_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_log_block8_t *b = &blocks[bi];
        int16x8_t  sr = vld1q_s16(b->s_rho),   gr = vld1q_s16(b->g_rho),   wr = vld1q_s16(b->w_rho);
        uint16x8_t st = vld1q_u16(b->s_theta), gt = vld1q_u16(b->g_theta), wt = vld1q_u16(b->w_theta);
        int16x8_t  ser  = lc_log_add_rho_x8(gr, sr);
        uint16x8_t set  = vaddq_u16(gt, st);
        int16x8_t  pr   = lc_log_add_rho_x8(wr, ser);
        uint16x8_t pt   = vaddq_u16(wt, set);
        int16x8_t  unbr = lc_log_add_rho_x8(pr, vnegq_s16(wr));
        uint16x8_t unbt = vsubq_u16(pt, wt);

        vst1q_s16(b->se_rho,  ser);  vst1q_u16(b->se_theta,  set);
        vst1q_s16(b->p_rho,   pr);   vst1q_u16(b->p_theta,   pt);
        vst1q_s16(b->unb_rho, unbr); vst1q_u16(b->unb_theta, unbt);

        int16x8_t p_re, p_im;
        lc_log_to_cart_x8_neon(b->p_rho, b->p_theta, &p_re, &p_im);
        int16x8_t v_re = vld1q_s16(b->v_re), v_im = vld1q_s16(b->v_im);
        int16x8_t r_re = lc_sub_mod_x8(v_re, p_re), r_im = lc_sub_mod_x8(v_im, p_im);
        vst1q_s16(b->p_re,   p_re);                  vst1q_s16(b->p_im,   p_im);
        vst1q_s16(b->r_re,   r_re);                  vst1q_s16(b->r_im,   r_im);
        vst1q_s16(b->res_re, lc_add_mod_x8(r_re, p_re));
        vst1q_s16(b->res_im, lc_add_mod_x8(r_im, p_im));
    }
}

LC_NOINLINE void lc_chart_aosoa8_cart_to_log(lc_chart_block8_t *blocks, int n_blocks) {
    const int16x8_t  zero16   = vdupq_n_s16(0), log_zero = vdupq_n_s16(LC_LOG_ZERO);
    const uint16x8_t zero_u16 = vdupq_n_u16(0), q1_16384 = vdupq_n_u16(16384u), q2_32768 = vdupq_n_u16(32768u);
    const uint32x4_t zero_u32 = vdupq_n_u32(0u), one_u32 = vdupq_n_u32(1u);
    const uint32x4_t mask31   = vdupq_n_u32(0x7FFFFFFFu), round21 = vdupq_n_u32(1u << 21);
    const int32x4_t  c3_32    = vdupq_n_s32(3);

    for (int bi = 0; bi < n_blocks; bi++) {
        lc_chart_block8_t *b = &blocks[bi];
        int16x8_t  v_re  = vld1q_s16(b->re), v_im = vld1q_s16(b->im);
        uint32x4_t ax_lo = vreinterpretq_u32_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_re))));
        uint32x4_t ax_hi = vreinterpretq_u32_s32(vabsq_s32(vmovl_high_s16(v_re)));
        uint32x4_t ay_lo = vreinterpretq_u32_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_im))));
        uint32x4_t ay_hi = vreinterpretq_u32_s32(vabsq_s32(vmovl_high_s16(v_im)));

        uint32x4_t r2_lo   = vmlaq_u32(vmulq_u32(ax_lo, ax_lo), ay_lo, ay_lo);
        uint32x4_t r2_hi   = vmlaq_u32(vmulq_u32(ax_hi, ax_hi), ay_hi, ay_hi);
        uint32x4_t lz_lo   = vclzq_u32(r2_lo), lz_hi = vclzq_u32(r2_hi);
        uint32x4_t norm_lo = vshlq_u32(r2_lo, vreinterpretq_s32_u32(lz_lo));
        uint32x4_t norm_hi = vshlq_u32(r2_hi, vreinterpretq_s32_u32(lz_hi));
        uint32x4_t lidx_lo = vshrq_n_u32(vaddq_u32(vandq_u32(norm_lo, mask31), round21), 22);
        uint32x4_t lidx_hi = vshrq_n_u32(vaddq_u32(vandq_u32(norm_hi, mask31), round21), 22);

        int32x4_t base_lo  = vshlq_n_s32(vsubq_s32(c3_32, vreinterpretq_s32_u32(lz_lo)), 7);
        int32x4_t base_hi  = vshlq_n_s32(vsubq_s32(c3_32, vreinterpretq_s32_u32(lz_hi)), 7);
        int16x8_t rho_base = vcombine_s16(vmovn_s32(base_lo), vmovn_s32(base_hi));

        uint32x4_t hi_lo = vmaxq_u32(vmaxq_u32(ax_lo, ay_lo), one_u32);
        uint32x4_t hi_hi = vmaxq_u32(vmaxq_u32(ax_hi, ay_hi), one_u32);
        uint32x4_t lo_lo = vminq_u32(ax_lo, ay_lo), lo_hi = vminq_u32(ax_hi, ay_hi);

        uint32_t lidx_a[8], hi_a[8], lo_a[8];
        vst1q_u32(&lidx_a[0], lidx_lo); vst1q_u32(&lidx_a[4], lidx_hi);
        vst1q_u32(&hi_a[0],   hi_lo);   vst1q_u32(&hi_a[4],   hi_hi);
        vst1q_u32(&lo_a[0],   lo_lo);   vst1q_u32(&lo_a[4],   lo_hi);

        uint16_t frac_a[8], phi_a[8];
        for (int k = 0; k < 8; k++) {
            frac_a[k] = lc_lut_half_log2[lidx_a[k]];
            uint32_t aidx = ((lo_a[k] << LC_ATAN_BITS) + (hi_a[k] >> 1)) / hi_a[k];
            phi_a[k]  = lc_lut_atan[aidx];
        }

        uint16x8_t is_zero  = vcombine_u16(vmovn_u32(vceqq_u32(r2_lo, zero_u32)),
                                           vmovn_u32(vceqq_u32(r2_hi, zero_u32)));
        int16x8_t  v_rho    = vbslq_s16(is_zero, log_zero,
                                        vaddq_s16(rho_base, vreinterpretq_s16_u16(vld1q_u16(frac_a))));
        uint16x8_t ay_gt_ax = vcombine_u16(vmovn_u32(vcgtq_u32(ay_lo, ax_lo)),
                                           vmovn_u32(vcgtq_u32(ay_hi, ax_hi)));
        uint16x8_t phi      = vld1q_u16(phi_a);
        uint16x8_t oct      = vbslq_u16(ay_gt_ax, vsubq_u16(q1_16384, phi), phi);
        uint16x8_t half     = vbslq_u16(vcltq_s16(v_re, zero16), vsubq_u16(q2_32768, oct), oct);
        uint16x8_t theta    = vbicq_u16(vbslq_u16(vcltq_s16(v_im, zero16), vsubq_u16(zero_u16, half), half),
                                        is_zero);
        vst1q_s16(b->rho, v_rho);
        vst1q_u16(b->theta, theta);
    }
}

LC_NOINLINE uint64_t lc_holo_aosoa8_eval_2morph(lc_holo_block8_t *blocks, int n_blocks,
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
        uint16x8_t abs_t = vreinterpretq_u16_s16(vabsq_s16(dt));
        acc = vpadalq_u32(acc, vpaddlq_u16(abs_t));
        vst1q_s16(b->out_drho,   dr);
        vst1q_s16(b->out_dtheta, dt);
    }
    return vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
}

#else /* 非 ARM 環境向け 0-bit 一致スカラーディスパッチ */

LC_NOINLINE void lc_aosoa8_forward(lc_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_block8_t *b = &blocks[bi];
        for (int k = 0; k < 8; k++) {
            lc_c16_t se  = lc_c16_mul_q14((lc_c16_t){b->g_re[k], b->g_im[k]},
                                          (lc_c16_t){b->s_re[k], b->s_im[k]});
            lc_c16_t p   = lc_c16_mul_q14((lc_c16_t){b->w_re[k], b->w_im[k]}, se);
            lc_c16_t r   = lc_c16_sub_mod((lc_c16_t){b->v_re[k], b->v_im[k]}, p);
            lc_c16_t res = lc_c16_add_mod(r, p);
            b->s_eff_re[k] = se.re; b->s_eff_im[k] = se.im;
            b->p_re[k] = p.re;      b->p_im[k] = p.im;
            b->r_re[k] = r.re;      b->r_im[k] = r.im;
            b->restored_re[k] = res.re; b->restored_im[k] = res.im;
        }
    }
}

LC_NOINLINE void lc_aosoa8_backward(lc_block8_t *blocks, int n_blocks) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_block8_t *b = &blocks[bi];
        for (int k = 0; k < 8; k++) {
            int32_t ei = (b->r_im[k] == LC_Q14_NOT)
                         ? ((b->s_eff_re[k] > LC_Q14_HALF) ? 32767 : -32768)
                         : b->r_im[k];
            int64_t f64 = (int64_t)b->w_re[k] * b->r_re[k] + (int64_t)b->w_im[k] * ei;
            int32_t tot = b->adj_in[k] + lc_sym_rshift64(f64, 14);
            int32_t gv = b->g_re[k], sv = b->s_re[k];
            if (tot > 0) { if (gv < 2048) gv = 2048; if (sv < 2048) sv = 2048; }
            b->adj_s[k] = lc_sym_rshift64((int64_t)tot * gv, 14);
            b->adj_g[k] = lc_sym_rshift64((int64_t)tot * sv, 14) & b->gate_mask[k];
        }
    }
}

LC_NOINLINE void lc_aosoa8_update(lc_block8_t *blocks, int n_blocks, uint8_t lr_shift) {
    for (int bi = 0; bi < n_blocks; bi++) {
        lc_block8_t *b = &blocks[bi];
        for (int k = 0; k < 8; k++) {
            lc_c16_t w = { b->w_re[k], b->w_im[k] };
            lc_update_weight_mod(&w, (lc_c16_t){b->r_re[k], b->r_im[k]},
                                 (lc_c16_t){b->s_eff_re[k], b->s_eff_im[k]}, lr_shift);
            b->w_re[k] = w.re; b->w_im[k] = w.im;
        }
    }
}

LC_NOINLINE void lc_log_aosoa8_bind_only(lc_log_block8_t *blocks, int n_blocks) {
    lc_log_aosoa8_bind_only_scalar(blocks, n_blocks);
}
LC_NOINLINE void lc_log_aosoa8_dual_step(lc_log_block8_t *blocks, int n_blocks) {
    lc_log_aosoa8_dual_step_scalar(blocks, n_blocks);
}
LC_NOINLINE void lc_chart_aosoa8_cart_to_log(lc_chart_block8_t *blocks, int n_blocks) {
    lc_chart_aosoa8_cart_to_log_scalar(blocks, n_blocks);
}
LC_NOINLINE uint64_t lc_holo_aosoa8_eval_2morph(lc_holo_block8_t *blocks, int n_blocks, bool apply_twist) {
    return lc_holo_aosoa8_eval_2morph_scalar(blocks, n_blocks, apply_twist);
}
#endif /* LC_HAS_ARM_NEON */

void lc_aosoa8_step_l1_blocked(lc_block8_t *blocks, int n_blocks, uint8_t lr_shift) {
    for (int off = 0; off < n_blocks; off += LC_L1_BLOCK_GROUPS) {
        int chunk = (n_blocks - off < LC_L1_BLOCK_GROUPS) ? (n_blocks - off) : LC_L1_BLOCK_GROUPS;
        lc_aosoa8_forward(&blocks[off], chunk);
        lc_aosoa8_backward(&blocks[off], chunk);
        lc_aosoa8_update(&blocks[off], chunk, lr_shift);
    }
}

#endif /* LIFTING_CORE_V2_IMPLEMENTATION */
#endif /* LIFTING_CORE_V2_H */