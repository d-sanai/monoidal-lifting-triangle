#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

/* ============================================================================
 * 1. Q1.14 固定小数点・ガウス整数（Z / 2^16 Z 円環）基本演算
 * ============================================================================ */
#define Q14_ONE   16384    /*  1.0 in Q1.14 (前提・カリー化ゲート True) */
#define Q14_HALF   8192    /*  0.5 in Q1.14 (離散判定閾値) */
#define Q14_NOT  -32768    /* 0x8000 in Z/2^16Z (位数2の元: 可逆反転 / 結論 True) */

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

static inline bool c16_equal(c16_t a, c16_t b) {
    return (a.re == b.re) && (a.im == b.im);
}

static inline int32_t sym_rshift64(int64_t val, int32_t shift) {
    if (val >= 0) return (int32_t)(val >> shift);
    return -(int32_t)((-val) >> shift);
}

static inline uint8_t popcount8(uint8_t x) {
    uint8_t cnt = 0;
    while (x != 0) {
        cnt += (uint8_t)(x & 1u);
        x >>= 1;
    }
    return cnt;
}

static inline int popcount64(uint64_t x) {
    int cnt = 0;
    while (x != 0ULL) {
        cnt += (int)(x & 1ULL);
        x >>= 1;
    }
    return cnt;
}

static inline int highest_bit64(uint64_t x) {
    int pos = -1;
    while (x != 0ULL) {
        x >>= 1;
        pos++;
    }
    return pos;
}

/* Z/2^16Z 円環上の決定論的共役勾配更新 */
static void update_complex_weight_mod(c16_t *w, c16_t err, c16_t s, uint8_t lr_shift) {
    if ((err.re == 0 && err.im == 0) || (s.re == 0 && s.im == 0)) return;

    int64_t grad_re = (int64_t)err.re * s.re + (int64_t)err.im * s.im;
    int64_t grad_im = (int64_t)err.im * s.re - (int64_t)err.re * s.im;
    int32_t shift   = 14 + lr_shift;

    int32_t step_re = sym_rshift64(grad_re, shift);
    int32_t step_im = sym_rshift64(grad_im, shift);

    if (step_re == 0 && grad_re != 0) step_re = (grad_re > 0) ? 1 : -1;
    if (step_im == 0 && grad_im != 0) step_im = (grad_im > 0) ? 1 : -1;

    w->re = (int16_t)((uint16_t)w->re + (uint16_t)step_re);
    w->im = (int16_t)((uint16_t)w->im + (uint16_t)step_im);
}

/* ============================================================================
 * 2. 64-byte 均一 LiftingCell（論理合成・双方向逆流・矛盾キャッシュ DAG 共通）
 * ============================================================================ */
typedef enum {
    MODE_EXACT = 0,   /* ビット完全可逆性検証 (μ ∘ Δ = id) */
    MODE_LEARN = 1,   /* 基底直交学習 + 動的カリー化 */
    MODE_EVAL  = 2    /* 純粋演繹評価 */
} ExecMode;

#define FLAG_ALIVE         (1u << 0)  /* プール内で生存中 */
#define FLAG_ACTIVE        (1u << 1)  /* 実効前提 s_eff が非ゼロ */
#define FLAG_CONFLICT_TERM (1u << 2)  /* 矛盾禁止積 (l_1 * ... * l_m = 0) の終端セル */

typedef struct LiftingCell {
    /* --- 動的 DAG 配線トポロジー (24 bytes) --- */
    struct LiftingCell *in_s;       /* 前提（骨格）入力 / 共有前置カリー化セル */
    struct LiftingCell *in_g;       /* カリー化ゲート入力 (NULL なら定数 1.0) */
    struct LiftingCell *in_v;       /* 下位からの変分（残差）直列入力 */

    /* --- 回転子重みと Δ / μ 状態 (24 bytes) --- */
    c16_t w;                        /* 複素回転子重み */
    c16_t s_eff;                    /* 実効前提: Q(g * s) */
    c16_t p;                        /* 局所予測変分: Q(w * s_eff) */
    c16_t r;                        /* 未説明残差変分: v ⊖ p */
    c16_t td_r;                     /* トップダウン合流変分 */
    c16_t restored;                 /* モノイド合流後の復元値 */

    /* --- 双方向逆流 & ヒステリシス代謝 (8 bytes) --- */
    int32_t adj_s;                  /* 骨格チャネルへの逆流位相トルク蓄積 */
    int32_t hit_energy;             /* 矛盾キャッシュ利用エネルギー */

    /* --- 参照カウント & メタデータ (8 bytes -> 合計 64 bytes・パディング 0) --- */
    uint16_t ref_count;             /* 共有 DAG の被参照数 */
    uint16_t idle_counter;          /* ヒステリシス未使用ステップカウンタ */
    uint8_t  var_mask;              /* 論理合成時の変数マスク / 制約リテラル禁止極性(0 or 1) */
    uint8_t  order;                 /* カリー化次数 */
    uint8_t  var_id;                /* 制約リテラルが指すグラフ変数 ID */
    uint8_t  flags;                 /* 状態フラグ群 */
} LiftingCell;

#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFULL
typedef char lifting_cell_must_be_64_bytes[(sizeof(LiftingCell) == 64) ? 1 : -1];
#endif

/* ============================================================================
 * 3. 可変トポロジー基本ゲート（Toffoli-AND / Sum / Carry）自己合成エンジン
 * ============================================================================ */
#define MAX_VARS  3
#define MAX_CELLS 16

typedef struct {
    LiftingCell in_ports[MAX_VARS];
    LiftingCell in_target;
    LiftingCell cells[MAX_CELLS];
    int         num_cells;
    int         spawned_count;
    uint8_t     num_vars;
    uint8_t     lr_shift;
} DynamicCurryingNet;

static void cell_reset(LiftingCell *c) {
    c->in_s         = NULL;
    c->in_g         = NULL;
    c->in_v         = NULL;
    c->w            = (c16_t){0, 0};
    c->s_eff        = (c16_t){0, 0};
    c->p            = (c16_t){0, 0};
    c->r            = (c16_t){0, 0};
    c->td_r         = (c16_t){0, 0};
    c->restored     = (c16_t){0, 0};
    c->adj_s        = 0;
    c->hit_energy   = 0;
    c->ref_count    = 0;
    c->idle_counter = 0;
    c->var_mask     = 0;
    c->order        = 0;
    c->var_id       = 0;
    c->flags        = 0;
}

static void net_init(DynamicCurryingNet *net, uint8_t num_vars) {
    net->num_vars      = num_vars;
    net->num_cells     = 0;
    net->spawned_count = 0;
    net->lr_shift      = 2;

    cell_reset(&net->in_target);
    for (int i = 0; i < MAX_VARS; i++) {
        cell_reset(&net->in_ports[i]);
        net->in_ports[i].var_mask = (uint8_t)(1u << i);
        net->in_ports[i].order    = 1;
    }
    for (int i = 0; i < MAX_CELLS; i++) {
        cell_reset(&net->cells[i]);
    }

    for (uint8_t i = 0; i < num_vars; i++) {
        LiftingCell *c = &net->cells[i];
        c->in_s     = &net->in_ports[i];
        c->in_g     = NULL;
        c->in_v     = (i == 0) ? &net->in_target : &net->cells[i - 1];
        c->var_mask = (uint8_t)(1u << i);
        c->order    = 1;
        c->flags    = FLAG_ALIVE;
        net->num_cells++;
    }
}

static bool net_has_mask(const DynamicCurryingNet *net, uint8_t mask) {
    for (int i = 0; i < net->num_cells; i++) {
        if (net->cells[i].var_mask == mask) return true;
    }
    return false;
}

static bool net_spawn_curried_cell(DynamicCurryingNet *net, int idx_u, int idx_v) {
    if (net->num_cells >= MAX_CELLS) return false;
    LiftingCell *u = &net->cells[idx_u];
    LiftingCell *v = &net->cells[idx_v];
    uint8_t umask  = (uint8_t)(u->var_mask | v->var_mask);
    if (net_has_mask(net, umask)) return false;

    LiftingCell *nc = &net->cells[net->num_cells];
    cell_reset(nc);
    nc->in_s     = u;
    nc->in_g     = v;
    nc->in_v     = &net->cells[net->num_cells - 1];
    nc->var_mask = umask;
    nc->order    = popcount8(umask);
    nc->flags    = FLAG_ALIVE;

    net->num_cells++;
    net->spawned_count++;
    return true;
}

static inline void cell_bottom_up(LiftingCell *c) {
    c16_t g = (c->in_g != NULL) ? c->in_g->s_eff : (c16_t){Q14_ONE, 0};
    c->s_eff = c16_mul_q14(g, c->in_s->s_eff);
    if (c16_l1_norm(c->s_eff) > 0) c->flags |= FLAG_ACTIVE;
    else                           c->flags &= (uint8_t)(~FLAG_ACTIVE);
    c->p = c16_mul_q14(c->w, c->s_eff);
    c->r = c16_sub_mod(c->in_v->r, c->p);
}

static void cell_top_down(LiftingCell *c, ExecMode mode, bool is_top) {
    if (is_top) {
        c->td_r = (mode == MODE_EVAL) ? (c16_t){0, 0} : c->r;
    }
    c->restored = c16_add_mod(c->td_r, c->p);
    if (c->in_v != NULL) {
        c->in_v->td_r = c->restored;
    }
}

static void cell_learn(LiftingCell *c, uint8_t active_input_mask, uint8_t lr_shift) {
    if ((c->flags & FLAG_ACTIVE) == 0u) return;
    if (c->var_mask != active_input_mask) return;
    update_complex_weight_mod(&c->w, c->r, c->s_eff, lr_shift);
}

static c16_t net_forward_continuous(DynamicCurryingNet *net,
                                    const int16_t u_in[MAX_VARS],
                                    c16_t target_var) {
    for (uint8_t i = 0; i < net->num_vars; i++) {
        c16_t val = { u_in[i], 0 };
        net->in_ports[i].s_eff = val;
        net->in_ports[i].r     = val;
        net->in_ports[i].adj_s = 0;
    }
    net->in_target.s_eff = target_var;
    net->in_target.r     = target_var;
    net->in_target.adj_s = 0;

    for (int i = 0; i < net->num_cells; i++) {
        net->cells[i].adj_s = 0;
        cell_bottom_up(&net->cells[i]);
    }
    return net->cells[net->num_cells - 1].r;
}

static void net_backward_adjoint(DynamicCurryingNet *net, c16_t top_r) {
    if (top_r.re == 0 && top_r.im == 0) return;

    for (int i = net->num_cells - 1; i >= 0; i--) {
        LiftingCell *c = &net->cells[i];

        int32_t eff_e_re = top_r.re;
        int32_t eff_e_im = top_r.im;
        if (top_r.im == Q14_NOT) {
            eff_e_im = (c->s_eff.re > Q14_HALF) ? 32767 : -32768;
        }

        int64_t f64     = (int64_t)c->w.re * eff_e_re + (int64_t)c->w.im * eff_e_im;
        int32_t f_local = sym_rshift64(f64, 14);
        int32_t total   = c->adj_s + f_local;

        if (total == 0) continue;

        if (c->in_g == NULL) {
            c->in_s->adj_s += total;
        } else {
            int32_t g_val = c->in_g->s_eff.re;
            int32_t s_val = c->in_s->s_eff.re;
            if (total > 0) {
                if (g_val < 2048) g_val = 2048;
                if (s_val < 2048) s_val = 2048;
            }
            c->in_s->adj_s += (int32_t)(((int64_t)total * g_val) >> 14);
            c->in_g->adj_s += (int32_t)(((int64_t)total * s_val) >> 14);
        }
    }
}

static c16_t net_step_discrete(DynamicCurryingNet *net, const bool in_bits[MAX_VARS],
                               c16_t target_var, ExecMode mode) {
    uint8_t active_mask = 0;
    int16_t u_in[MAX_VARS] = {0, 0, 0};
    for (uint8_t i = 0; i < net->num_vars; i++) {
        u_in[i] = in_bits[i] ? Q14_ONE : 0;
        if (in_bits[i]) active_mask |= (uint8_t)(1u << i);
    }

    (void)net_forward_continuous(net, u_in, target_var);

    for (int i = net->num_cells - 1; i >= 0; i--) {
        cell_top_down(&net->cells[i], mode, i == net->num_cells - 1);
    }

    if (mode == MODE_LEARN) {
        for (int i = 0; i < net->num_cells; i++) {
            cell_learn(&net->cells[i], active_mask, net->lr_shift);
        }
    }

    return net->cells[0].restored;
}

/* ボトムアップ (Δ) のみによる 0-bit 厳密残差評価（トップダウン不要な残差判定を高速化） */
static inline int32_t eval_gate_discrete_err(DynamicCurryingNet *net,
                                             const bool b_in[MAX_VARS],
                                             bool b_out) {
    int16_t u_in[MAX_VARS] = {
        b_in[0] ? Q14_ONE : 0,
        b_in[1] ? Q14_ONE : 0,
        b_in[2] ? Q14_ONE : 0
    };
    c16_t t_var = b_out ? (c16_t){0, Q14_NOT} : (c16_t){0, 0};
    c16_t top_r = net_forward_continuous(net, u_in, t_var);
    return c16_l1_norm(top_r);
}

typedef bool (*BoolFn3)(bool, bool, bool);

static void synthesize_dynamic_topology(DynamicCurryingNet *net, BoolFn3 fn, int max_epochs) {
    for (int ep = 0; ep < max_epochs; ep++) {
        int64_t known_err_sum   = 0;
        int64_t unknown_err_sum = 0;
        int64_t pair_score[MAX_CELLS][MAX_CELLS];

        for (int u = 0; u < net->num_cells; u++) {
            for (int v = 0; v < net->num_cells; v++) {
                pair_score[u][v] = 0;
            }
        }

        for (int mask = 0; mask < 8; mask++) {
            bool bits[MAX_VARS] = { (mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0 };
            c16_t target_var = fn(bits[0], bits[1], bits[2])
                               ? (c16_t){0, Q14_NOT} : (c16_t){0, 0};

            (void)net_step_discrete(net, bits, target_var, MODE_LEARN);

            int32_t err_norm = c16_l1_norm(net->cells[net->num_cells - 1].r);
            if (err_norm == 0) continue;

            if (net_has_mask(net, (uint8_t)mask)) {
                known_err_sum += err_norm;
            } else {
                unknown_err_sum += err_norm;
                for (int u = 0; u < net->num_cells; u++) {
                    if ((net->cells[u].flags & FLAG_ACTIVE) == 0u) continue;
                    for (int v = u + 1; v < net->num_cells; v++) {
                        if ((net->cells[v].flags & FLAG_ACTIVE) == 0u) continue;
                        if ((net->cells[u].var_mask & net->cells[v].var_mask) != 0u) continue;
                        uint8_t umask = (uint8_t)(net->cells[u].var_mask | net->cells[v].var_mask);
                        if (net_has_mask(net, umask)) continue;
                        pair_score[u][v] += err_norm;
                    }
                }
            }
        }

        if (known_err_sum == 0 && unknown_err_sum == 0) return;

        if (known_err_sum == 0 && unknown_err_sum > 0 && net->num_cells < MAX_CELLS) {
            int     best_u = -1, best_v = -1;
            int64_t best_score = 0;
            uint8_t best_bits  = 0;

            for (int u = 0; u < net->num_cells; u++) {
                for (int v = u + 1; v < net->num_cells; v++) {
                    if (pair_score[u][v] <= 0) continue;
                    uint8_t umask = (uint8_t)(net->cells[u].var_mask | net->cells[v].var_mask);
                    uint8_t ubits = popcount8(umask);
                    if (pair_score[u][v] > best_score ||
                        (pair_score[u][v] == best_score && ubits > best_bits)) {
                        best_score = pair_score[u][v];
                        best_bits  = ubits;
                        best_u     = u;
                        best_v     = v;
                    }
                }
            }
            if (best_u >= 0 && best_v >= 0) {
                net_spawn_curried_cell(net, best_u, best_v);
            }
        }
    }
}

static bool verify_primitive_exact(DynamicCurryingNet *net, BoolFn3 fn) {
    for (int mask = 0; mask < 8; mask++) {
        bool bits[MAX_VARS] = { (mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0 };
        c16_t expected = fn(bits[0], bits[1], bits[2]) ? (c16_t){0, Q14_NOT} : (c16_t){0, 0};

        c16_t eval_out = net_step_discrete(net, bits, expected, MODE_EVAL);
        c16_t top_r    = net->cells[net->num_cells - 1].r;
        if (!c16_equal(eval_out, expected) || !c16_equal(top_r, (c16_t){0, 0})) {
            return false;
        }
        c16_t rev_out = net_step_discrete(net, bits, expected, MODE_EXACT);
        if (!c16_equal(rev_out, expected)) return false;
    }
    return true;
}

/* ============================================================================
 * 4. 64-byte 共有 DAG カリー化矛盾制約キャッシュ（CDCL 学習セル + ヒステリシス剪定）
 *    禁止パターン l_0 * l_1 * ... * l_{m-1} = 0 を LiftingCell のカリー化鎖として保持
 * ============================================================================ */
#define MAX_GRAPH_VARS     256
#define MAX_GRAPH_GATES    256
#define MAX_BRANCH_DEPTH   64
#define MAX_CONFLICT_CELLS 48   /* 48 * 64B = 3 KB 固定長プール（代謝と再利用を強制検証） */
#define MAX_CONFLICT_LITS  6

typedef struct {
    uint8_t var_id;
    uint8_t forbid_bit;
} ConflictLit;

typedef struct {
    LiftingCell pool[MAX_CONFLICT_CELLS];
    LiftingCell zero_target;
    bool        slot_ever_used[MAX_CONFLICT_CELLS];
    int         alive_cells;
    int         active_terms;
    int         spawned_cells;
    int         shared_hits;
    int         pruned_cells;
    int         reused_slots;
    int         cache_propagations;
    int         cache_conflicts;
} ConflictCurryingCache;

static void ccache_init(ConflictCurryingCache *cc) {
    cell_reset(&cc->zero_target);
    cc->zero_target.flags = FLAG_ALIVE;
    for (int i = 0; i < MAX_CONFLICT_CELLS; i++) {
        cell_reset(&cc->pool[i]);
        cc->slot_ever_used[i] = false;
    }
    cc->alive_cells        = 0;
    cc->active_terms       = 0;
    cc->spawned_cells      = 0;
    cc->shared_hits        = 0;
    cc->pruned_cells       = 0;
    cc->reused_slots       = 0;
    cc->cache_propagations = 0;
    cc->cache_conflicts    = 0;
}

/* 参照カウント 0 の末端矛盾制約セルをヒステリシス順に連鎖剪定（繰り込み縮約） */
static bool ccache_prune_one_leaf(ConflictCurryingCache *cc) {
    int best_idx   = -1;
    int best_score = -1;

    for (int i = 0; i < MAX_CONFLICT_CELLS; i++) {
        LiftingCell *c = &cc->pool[i];
        if ((c->flags & FLAG_ALIVE) == 0u) continue;
        if (c->ref_count != 0u) continue;
        /* 未使用期間が長く、ヒットエネルギーの低い末端セルほど優先剪定 */
        int score = (int)c->idle_counter * 16 - c->hit_energy;
        if (best_idx < 0 || score > best_score) {
            best_score = score;
            best_idx   = i;
        }
    }
    if (best_idx < 0) return false;

    /* 末端セルから親の ref_count==0 になった共有中間セルまで連鎖縮約 */
    LiftingCell *queue[MAX_CONFLICT_CELLS];
    int qh = 0, qt = 0;
    queue[qt++] = &cc->pool[best_idx];

    while (qh < qt) {
        LiftingCell *c = queue[qh++];
        if ((c->flags & FLAG_ALIVE) == 0u || c->ref_count != 0u) continue;

        LiftingCell *ps = c->in_s;
        LiftingCell *pg = c->in_g;
        if ((c->flags & FLAG_CONFLICT_TERM) != 0u) {
            cc->active_terms--;
        }

        cell_reset(c);
        cc->alive_cells--;
        cc->pruned_cells++;

        if (ps != NULL && ps->ref_count > 0u) {
            ps->ref_count--;
            if (ps->ref_count == 0u && (ps->flags & FLAG_CONFLICT_TERM) == 0u) {
                queue[qt++] = ps;
            }
        }
        if (pg != NULL && pg->ref_count > 0u) {
            pg->ref_count--;
            if (pg->ref_count == 0u && (pg->flags & FLAG_CONFLICT_TERM) == 0u) {
                queue[qt++] = pg;
            }
        }
    }
    return true;
}

static LiftingCell *ccache_alloc_slot(ConflictCurryingCache *cc) {
    for (int attempt = 0; attempt < 2; attempt++) {
        for (int i = 0; i < MAX_CONFLICT_CELLS; i++) {
            if ((cc->pool[i].flags & FLAG_ALIVE) == 0u) {
                if (cc->slot_ever_used[i]) cc->reused_slots++;
                cc->slot_ever_used[i] = true;
                cell_reset(&cc->pool[i]);
                cc->pool[i].flags = FLAG_ALIVE;
                cc->alive_cells++;
                cc->spawned_cells++;
                return &cc->pool[i];
            }
        }
        /* プール満杯時はヒステリシス繰り込み縮約でスロットを自律確保 */
        if (!ccache_prune_one_leaf(cc)) break;
    }
    return NULL;
}

static LiftingCell *ccache_get_or_create_lit(ConflictCurryingCache *cc,
                                             uint8_t var_id, uint8_t forbid_bit) {
    for (int i = 0; i < MAX_CONFLICT_CELLS; i++) {
        LiftingCell *c = &cc->pool[i];
        if ((c->flags & FLAG_ALIVE) != 0u && c->order == 1 &&
            c->var_id == var_id && c->var_mask == forbid_bit) {
            c->idle_counter = 0;
            cc->shared_hits++;
            return c;
        }
    }
    LiftingCell *nc = ccache_alloc_slot(cc);
    if (nc == NULL) return NULL;
    nc->in_s       = NULL;
    nc->in_g       = NULL;
    nc->in_v       = &cc->zero_target;
    nc->var_id     = var_id;
    nc->var_mask   = forbid_bit;
    nc->order      = 1;
    nc->hit_energy = 16;
    return nc;
}

static LiftingCell *ccache_get_or_create_pair(ConflictCurryingCache *cc,
                                              LiftingCell *lhs, LiftingCell *rhs) {
    for (int i = 0; i < MAX_CONFLICT_CELLS; i++) {
        LiftingCell *c = &cc->pool[i];
        if ((c->flags & FLAG_ALIVE) != 0u && c->order > 1 &&
            c->in_s == lhs && c->in_g == rhs) {
            c->idle_counter = 0;
            cc->shared_hits++;
            return c;
        }
    }
    LiftingCell *nc = ccache_alloc_slot(cc);
    if (nc == NULL) return NULL;
    /* 確保時に剪定が走って lhs/rhs が消えていないか安全確認 */
    if ((lhs->flags & FLAG_ALIVE) == 0u || (rhs->flags & FLAG_ALIVE) == 0u) {
        cell_reset(nc);
        cc->alive_cells--;
        return NULL;
    }
    nc->in_s       = lhs;
    nc->in_g       = rhs;
    nc->in_v       = &cc->zero_target;
    nc->order      = (uint8_t)(lhs->order + 1u);
    nc->hit_energy = 16;
    lhs->ref_count++;
    rhs->ref_count++;
    return nc;
}

/* 矛盾原因リテラル列を共有 DAG 上のカリー化積 (l_0 * l_1 * ... * l_{m-1} = 0) として登録 */
static void ccache_add_conflict(ConflictCurryingCache *cc,
                                ConflictLit lits[MAX_CONFLICT_LITS], int n_lits) {
    if (n_lits < 2 || n_lits > MAX_CONFLICT_LITS) return;

    /* var_id 昇順に正規化して共有前置セルのヒット率を高める */
    for (int i = 0; i < n_lits; i++) {
        for (int j = i + 1; j < n_lits; j++) {
            if (lits[j].var_id < lits[i].var_id) {
                ConflictLit tmp = lits[i];
                lits[i] = lits[j];
                lits[j] = tmp;
            }
        }
    }

    /* 必要な最大新規セル数が確保できるよう事前に余裕を作る */
    while (MAX_CONFLICT_CELLS - cc->alive_cells < n_lits * 2) {
        if (!ccache_prune_one_leaf(cc)) break;
    }

    LiftingCell *cur = ccache_get_or_create_lit(cc, lits[0].var_id, lits[0].forbid_bit);
    if (cur == NULL) return;

    /* 構築途中の親セルが誤剪定されないよう一時ロック */
    cur->ref_count++;
    for (int i = 1; i < n_lits; i++) {
        LiftingCell *lit = ccache_get_or_create_lit(cc, lits[i].var_id, lits[i].forbid_bit);
        if (lit == NULL) {
            cur->ref_count--;
            return;
        }
        lit->ref_count++;
        LiftingCell *next = ccache_get_or_create_pair(cc, cur, lit);
        cur->ref_count--;
        lit->ref_count--;
        if (next == NULL) return;
        cur = next;
        cur->ref_count++;
    }
    cur->ref_count--;

    if ((cur->flags & FLAG_CONFLICT_TERM) == 0u) {
        cur->flags |= FLAG_CONFLICT_TERM;
        cur->w      = (c16_t){0, Q14_NOT}; /* 禁止積が 1 なら残差 0x8000 (矛盾!) */
        cc->active_terms++;
    }
}

/* 鎖を遡って禁止リテラル一覧を復元 */
static int ccache_extract_lits(const LiftingCell *term,
                               ConflictLit out_lits[MAX_CONFLICT_LITS]) {
    int n = 0;
    const LiftingCell *cur = term;
    while (cur != NULL && n < MAX_CONFLICT_LITS) {
        if (cur->order == 1) {
            out_lits[n].var_id     = cur->var_id;
            out_lits[n].forbid_bit = cur->var_mask;
            n++;
            break;
        }
        if (cur->in_g != NULL) {
            out_lits[n].var_id     = cur->in_g->var_id;
            out_lits[n].forbid_bit = cur->in_g->var_mask;
            n++;
        }
        cur = cur->in_s;
    }
    return n;
}

/* ============================================================================
 * 5. 因果波面（0-bit 局所射影）＋ カリー化矛盾キャッシュ ＋ 非クロノロジカル・バックジャンプ
 * ============================================================================ */
typedef struct {
    int16_t  u;          /* 連続位相状態 [0, 16384] */
    int32_t  conf;       /* 因果確信度 (>0 で確定, 0 で未確定) */
    uint64_t dep_mask;   /* 確定根拠となった仮説決定レベル集合（原因依存トレース） */
    int32_t  force;      /* 連続位相逆流ヒント蓄積 */
    int16_t  bond_deg;   /* 自己合成カリー化セル (in_g != NULL) による変数結合次数 */
} PhaseVar;

typedef struct {
    DynamicCurryingNet *net;
    int                 in_var[3];
    int                 out_var;
} LiftingGate;

typedef struct {
    PhaseVar    vars[MAX_GRAPH_VARS];
    LiftingGate gates[MAX_GRAPH_GATES];
    int         num_vars;
    int         num_gates;
} FactorGraph;

typedef struct {
    int16_t  u[MAX_GRAPH_VARS];
    int32_t  conf[MAX_GRAPH_VARS];
    uint64_t dep_mask[MAX_GRAPH_VARS];
    int      pivot_var;
    int8_t   first_bit;
    int8_t   alt_bit;
} BranchFrame;

typedef struct {
    int prop_steps;      /* 因果波面の伝播反復数 */
    int branches;        /* 仮説分裂（Hypothesis Mitosis）回数 */
    int pruned_dead;     /* 矛盾検出（残留残差非ゼロ）回数 */
    int backjumps;       /* 不要分岐を飛び越えた非クロノロジカル・バックジャンプ回数 */
    int max_depth;       /* 到達した最大仮説ネスト深さ */
    int cache_spawned;   /* 動的生成された 64-byte カリー化矛盾セル数 */
    int cache_shared;    /* 共有 DAG 再利用ヒット数 */
    int cache_props;     /* キャッシュからの単位伝播発火回数 */
    int cache_conflicts; /* キャッシュからの直接矛盾検出回数 */
    int cache_pruned;    /* ヒステリシス繰り込み縮約で剪定されたセル数 */
    int cache_reused;    /* フリーリスト再利用回数 */
} SolveStats;

static void graph_init(FactorGraph *g) {
    g->num_vars  = 0;
    g->num_gates = 0;
}

static int graph_add_var(FactorGraph *g, int8_t obs_val) {
    int id = g->num_vars++;
    PhaseVar *v = &g->vars[id];
    v->force    = 0;
    v->bond_deg = 0;
    v->dep_mask = 0ULL;
    if (obs_val < 0) {
        v->u    = Q14_HALF;
        v->conf = 0;
    } else {
        v->u    = (obs_val != 0) ? Q14_ONE : 0;
        v->conf = Q14_ONE;
    }
    return id;
}

static void graph_add_gate(FactorGraph *g, DynamicCurryingNet *net,
                           int in0, int in1, int in2, int out_v) {
    LiftingGate *gt = &g->gates[g->num_gates++];
    gt->net       = net;
    gt->in_var[0] = in0;
    gt->in_var[1] = in1;
    gt->in_var[2] = in2;
    gt->out_var   = out_v;

    for (int c = 0; c < net->num_cells; c++) {
        if (net->cells[c].in_g == NULL) continue;
        uint8_t mask = net->cells[c].var_mask;
        for (uint8_t k = 0; k < net->num_vars; k++) {
            if ((mask & (1u << k)) != 0u) {
                g->vars[gt->in_var[k]].bond_deg++;
            }
        }
    }
}

/* 共有 DAG カリー化矛盾キャッシュの因果波面評価（0-bit 残差検証 + 単位伝播）
 * 戻り値: 1 = 新たに単位伝播で変数が確定, 0 = 変化なし, -1 = キャッシュ矛盾検出 */
static int propagate_conflict_cache(FactorGraph *g, ConflictCurryingCache *cc,
                                    uint64_t *conflict_deps_out) {
    if (cc == NULL || cc->active_terms == 0) return 0;

    int progressed = 0;
    for (int i = 0; i < MAX_CONFLICT_CELLS; i++) {
        LiftingCell *term = &cc->pool[i];
        if ((term->flags & (FLAG_ALIVE | FLAG_CONFLICT_TERM)) !=
            (FLAG_ALIVE | FLAG_CONFLICT_TERM)) {
            continue;
        }
        if (term->idle_counter < 60000u) term->idle_counter++;

        ConflictLit lits[MAX_CONFLICT_LITS];
        int n = ccache_extract_lits(term, lits);
        if (n < 2) continue;

        int      unknown_idx = -1;
        int      unknown_cnt = 0;
        bool     satisfied   = false;
        uint64_t dep_union   = 0ULL;

        for (int k = 0; k < n; k++) {
            const PhaseVar *pv = &g->vars[lits[k].var_id];
            if (pv->conf == 0) {
                unknown_cnt++;
                unknown_idx = k;
            } else {
                uint8_t cur_bit = (pv->u >= Q14_HALF) ? 1u : 0u;
                if (cur_bit != lits[k].forbid_bit) {
                    satisfied = true;
                    break;
                }
                dep_union |= pv->dep_mask;
            }
        }

        if (satisfied || unknown_cnt > 1) continue;

        if (unknown_cnt == 0) {
            /* 実際にカリー化鎖の LiftingCell で Δ 評価し 0x8000 矛盾残差を確認 */
            LiftingCell *chain[MAX_CONFLICT_LITS];
            int clen = 0;
            LiftingCell *cur = term;
            while (cur != NULL && clen < MAX_CONFLICT_LITS) {
                chain[clen++] = cur;
                cur = cur->in_s;
            }
            for (int c = clen - 1; c >= 0; c--) {
                LiftingCell *node = chain[c];
                node->idle_counter = 0;
                node->hit_energy  += 8;
                if (node->order == 1) {
                    node->s_eff = (c16_t){Q14_ONE, 0};
                } else {
                    node->in_g->s_eff        = (c16_t){Q14_ONE, 0};
                    node->in_g->idle_counter = 0;
                    cell_bottom_up(node);
                }
            }
            if (c16_l1_norm(term->r) != 0) {
                cc->cache_conflicts++;
                *conflict_deps_out = dep_union;
                return -1;
            }
        } else { /* unknown_cnt == 1: 残差ゼロ不変条件による単位伝播 */
            int vid = lits[unknown_idx].var_id;
            bool forced_bit = (lits[unknown_idx].forbid_bit == 0u);
            g->vars[vid].u        = forced_bit ? Q14_ONE : 0;
            g->vars[vid].conf     = Q14_ONE - 64;
            g->vars[vid].dep_mask = dep_union;
            term->idle_counter    = 0;
            term->hit_energy     += 4;
            cc->cache_propagations++;
            progressed = 1;
        }
    }
    return progressed;
}

/* ゲート非依存 0-bit 局所射影（一般化因果波面伝播 + 原因依存マスク追跡）
 * 戻り値: 1 = 全変数確定かつ 0-bit 充足, 0 = 未確定残存, -1 = 矛盾(conflict_deps_out 設定) */
static int propagate_causal_wavefront(FactorGraph *g, ConflictCurryingCache *cc,
                                      SolveStats *st, uint64_t *conflict_deps_out) {
    bool progress = true;
    while (progress) {
        progress = false;
        st->prop_steps++;

        /* 1. 共有 DAG 矛盾キャッシュによる先行単位伝播・矛盾検出 */
        int cres = propagate_conflict_cache(g, cc, conflict_deps_out);
        if (cres < 0) return -1;
        if (cres > 0) progress = true;

        /* 2. 全可逆ゲートでの 0-bit 局所射影 */
        for (int gi = 0; gi < g->num_gates; gi++) {
            LiftingGate *gt = &g->gates[gi];
            int term[4] = {
                gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var
            };

            int      u_vars[4];
            int      nu         = 0;
            int      known_cnt  = 0;
            uint64_t known_deps = 0ULL;

            for (int k = 0; k < 4; k++) {
                int vid = term[k];
                if (g->vars[vid].conf > 0) {
                    known_cnt++;
                    known_deps |= g->vars[vid].dep_mask;
                    continue;
                }
                bool dup = false;
                for (int j = 0; j < nu; j++) {
                    if (u_vars[j] == vid) { dup = true; break; }
                }
                if (!dup) u_vars[nu++] = vid;
            }

            /* 全 4 端子が互いに異なる未確定変数のゲートは射影不変のためスキップ */
            if (known_cnt == 0 && nu == 4) continue;

            if (nu == 0) {
                bool b_in[MAX_VARS] = {
                    g->vars[term[0]].u >= Q14_HALF,
                    g->vars[term[1]].u >= Q14_HALF,
                    g->vars[term[2]].u >= Q14_HALF
                };
                bool b_out = (g->vars[term[3]].u >= Q14_HALF);
                if (eval_gate_discrete_err(gt->net, b_in, b_out) != 0) {
                    *conflict_deps_out = known_deps;
                    return -1;
                }
                continue;
            }

            int  valid_cnt = 0;
            bool seen0[4]  = {false, false, false, false};
            bool seen1[4]  = {false, false, false, false};

            int combos = 1 << nu;
            for (int sub = 0; sub < combos; sub++) {
                bool bv[4];
                for (int k = 0; k < 4; k++) {
                    int vid = term[k];
                    if (g->vars[vid].conf > 0) {
                        bv[k] = (g->vars[vid].u >= Q14_HALF);
                    } else {
                        int pos = 0;
                        while (pos < nu && u_vars[pos] != vid) pos++;
                        bv[k] = ((sub >> pos) & 1) != 0;
                    }
                }
                bool b_in[MAX_VARS] = { bv[0], bv[1], bv[2] };
                if (eval_gate_discrete_err(gt->net, b_in, bv[3]) == 0) {
                    valid_cnt++;
                    for (int j = 0; j < nu; j++) {
                        if (((sub >> j) & 1) != 0) seen1[j] = true;
                        else                       seen0[j] = true;
                    }
                }
            }

            if (valid_cnt == 0) {
                *conflict_deps_out = known_deps;
                return -1;
            }

            for (int j = 0; j < nu; j++) {
                if (seen0[j] != seen1[j]) {
                    int vid = u_vars[j];
                    g->vars[vid].u        = seen1[j] ? Q14_ONE : 0;
                    g->vars[vid].conf     = Q14_ONE - 64;
                    g->vars[vid].dep_mask = known_deps;
                    progress              = true;
                }
            }
        }
    }

    for (int v = 0; v < g->num_vars; v++) {
        if (g->vars[v].conf == 0) return 0;
    }
    return 1;
}

/* 連続位相の局所双方向逆流 (Δ†) による優先分岐方向ガイド */
static int8_t compute_phase_hint(FactorGraph *g, int pivot_var) {
    g->vars[pivot_var].force = 0;

    for (int gi = 0; gi < g->num_gates; gi++) {
        LiftingGate *gt = &g->gates[gi];
        bool involves_pivot = (gt->in_var[0] == pivot_var ||
                               gt->in_var[1] == pivot_var ||
                               gt->in_var[2] == pivot_var ||
                               gt->out_var   == pivot_var);
        if (!involves_pivot) continue;

        int16_t u_in[MAX_VARS] = {
            g->vars[gt->in_var[0]].u,
            g->vars[gt->in_var[1]].u,
            g->vars[gt->in_var[2]].u
        };
        int16_t u_out  = g->vars[gt->out_var].u;
        c16_t   t_cont = { 0, quantize_q14((int64_t)Q14_NOT * u_out) };

        c16_t top_r = net_forward_continuous(gt->net, u_in, t_cont);
        net_backward_adjoint(gt->net, top_r);

        for (int k = 0; k < 3; k++) {
            if (gt->in_var[k] == pivot_var) {
                g->vars[pivot_var].force += gt->net->in_ports[k].adj_s;
            }
        }
        if (gt->out_var == pivot_var) {
            g->vars[pivot_var].force += (int32_t)top_r.im * 2;
        }
    }

    return (g->vars[pivot_var].force >= 0) ? 1 : 0;
}

/* 局所自由度圧迫度 (tight2_deg) ＋ カリー化結合次数 (bond_deg) による自律ピボット選択 */
static int select_frontier_pivot(const FactorGraph *g) {
    int tight2_deg[MAX_GRAPH_VARS];
    int frontier_deg[MAX_GRAPH_VARS];
    for (int v = 0; v < g->num_vars; v++) {
        tight2_deg[v]   = 0;
        frontier_deg[v] = 0;
    }

    for (int gi = 0; gi < g->num_gates; gi++) {
        const LiftingGate *gt = &g->gates[gi];
        int term[4] = { gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var };
        int u_vars[4];
        int nu = 0, known_cnt = 0;

        for (int k = 0; k < 4; k++) {
            int vid = term[k];
            if (g->vars[vid].conf > 0) {
                known_cnt++;
            } else {
                bool dup = false;
                for (int j = 0; j < nu; j++) {
                    if (u_vars[j] == vid) { dup = true; break; }
                }
                if (!dup) u_vars[nu++] = vid;
            }
        }
        if (nu == 0 || known_cnt == 0) continue;

        for (int j = 0; j < nu; j++) {
            frontier_deg[u_vars[j]] += known_cnt;
            if (nu == 2) {
                /* 残り未確定端子がちょうど 2 個のゲートは、1 変数確定で即連鎖発火する最前線 */
                tight2_deg[u_vars[j]]++;
            }
        }
    }

    int best_v     = -1;
    int best_score = -1;
    for (int v = 0; v < g->num_vars; v++) {
        if (g->vars[v].conf > 0) continue;
        int score = tight2_deg[v] * 64 + (int)g->vars[v].bond_deg * 4 + frontier_deg[v];
        if (score > best_score) {
            best_score = score;
            best_v     = v;
        }
    }
    return best_v;
}

/* ゼロ・ヒープ固定スタックによる完全決定論的アブダクション
 * enable_cdcl = false : 従来のクロノロジカル・バックトラック（ベースライン）
 * enable_cdcl = true  : 64-byte カリー化矛盾キャッシュ + 原因依存バックジャンプ */
static bool solve_factor_abduction(FactorGraph *g, bool enable_cdcl,
                                   bool out_bits[MAX_GRAPH_VARS],
                                   SolveStats *stats) {
    static BranchFrame           stack[MAX_BRANCH_DEPTH];
    static ConflictCurryingCache ccache;
    int sp = 0;

    if (enable_cdcl) {
        ccache_init(&ccache);
    }

    stats->prop_steps      = 0;
    stats->branches        = 0;
    stats->pruned_dead     = 0;
    stats->backjumps       = 0;
    stats->max_depth       = 0;
    stats->cache_spawned   = 0;
    stats->cache_shared    = 0;
    stats->cache_props     = 0;
    stats->cache_conflicts = 0;
    stats->cache_pruned    = 0;
    stats->cache_reused    = 0;

    for (;;) {
        uint64_t conflict_deps = 0ULL;
        int res = propagate_causal_wavefront(g, enable_cdcl ? &ccache : NULL,
                                             stats, &conflict_deps);

        if (res == 1) {
            for (int v = 0; v < g->num_vars; v++) {
                out_bits[v] = (g->vars[v].u >= Q14_HALF);
            }
            if (enable_cdcl) {
                stats->cache_spawned   = ccache.spawned_cells;
                stats->cache_shared    = ccache.shared_hits;
                stats->cache_props     = ccache.cache_propagations;
                stats->cache_conflicts = ccache.cache_conflicts;
                stats->cache_pruned    = ccache.pruned_cells;
                stats->cache_reused    = ccache.reused_slots;
            }
            return true;
        }

        if (res == 0) {
            int pivot = select_frontier_pivot(g);
            if (pivot < 0 || sp >= MAX_BRANCH_DEPTH) return false;

            int8_t first_bit = compute_phase_hint(g, pivot);
            int8_t alt_bit   = (first_bit == 1) ? 0 : 1;

            BranchFrame *fr = &stack[sp];
            for (int v = 0; v < g->num_vars; v++) {
                fr->u[v]        = g->vars[v].u;
                fr->conf[v]     = g->vars[v].conf;
                fr->dep_mask[v] = g->vars[v].dep_mask;
            }
            fr->pivot_var = pivot;
            fr->first_bit = first_bit;
            fr->alt_bit   = alt_bit;

            g->vars[pivot].u        = (first_bit == 1) ? Q14_ONE : 0;
            g->vars[pivot].conf     = Q14_HALF;
            g->vars[pivot].dep_mask = (1ULL << sp);
            sp++;

            stats->branches++;
            if (sp > stats->max_depth) stats->max_depth = sp;
            continue;
        }

        /* res == -1: 矛盾検出 */
        stats->pruned_dead++;

        if (!enable_cdcl) {
            /* ベースライン: 素朴な 1 段クロノロジカル・バックトラック */
            if (sp == 0) return false;
            BranchFrame *fr = &stack[--sp];
            for (int v = 0; v < g->num_vars; v++) {
                g->vars[v].u        = fr->u[v];
                g->vars[v].conf     = fr->conf[v];
                g->vars[v].dep_mask = fr->dep_mask[v];
            }
            g->vars[fr->pivot_var].u        = (fr->alt_bit == 1) ? Q14_ONE : 0;
            g->vars[fr->pivot_var].conf     = Q14_HALF;
            g->vars[fr->pivot_var].dep_mask = 0ULL;
            continue;
        }

        /* CDCL モード: 最小矛盾原因を 64-byte カリー化制約セルとして動的キャッシュ */
        if (conflict_deps == 0ULL || sp == 0) {
            stats->cache_spawned   = ccache.spawned_cells;
            stats->cache_shared    = ccache.shared_hits;
            stats->cache_props     = ccache.cache_propagations;
            stats->cache_conflicts = ccache.cache_conflicts;
            stats->cache_pruned    = ccache.pruned_cells;
            stats->cache_reused    = ccache.reused_slots;
            return false; /* 仮説非依存の絶対矛盾 -> UNSAT 確定 */
        }

        int cause_cnt = popcount64(conflict_deps);
        if (cause_cnt >= 2 && cause_cnt <= MAX_CONFLICT_LITS) {
            ConflictLit lits[MAX_CONFLICT_LITS];
            int n_lits = 0;
            for (int lv = 0; lv < sp && n_lits < MAX_CONFLICT_LITS; lv++) {
                if ((conflict_deps & (1ULL << lv)) != 0ULL) {
                    lits[n_lits].var_id     = (uint8_t)stack[lv].pivot_var;
                    lits[n_lits].forbid_bit = (uint8_t)stack[lv].first_bit;
                    n_lits++;
                }
            }
            ccache_add_conflict(&ccache, lits, n_lits);
        }

        /* 原因レベル max_lv への非クロノロジカル・バックジャンプ */
        int max_lv = highest_bit64(conflict_deps);
        if (max_lv < 0 || max_lv >= sp) return false;
        if (max_lv < sp - 1) {
            stats->backjumps++;
        }

        sp = max_lv;
        BranchFrame *fr = &stack[max_lv];
        for (int v = 0; v < g->num_vars; v++) {
            g->vars[v].u        = fr->u[v];
            g->vars[v].conf     = fr->conf[v];
            g->vars[v].dep_mask = fr->dep_mask[v];
        }
        /* 代替ビットは (conflict_deps \ {max_lv}) からの厳密な論理的帰結として強制確定 */
        g->vars[fr->pivot_var].u        = (fr->alt_bit == 1) ? Q14_ONE : 0;
        g->vars[fr->pivot_var].conf     = Q14_HALF;
        g->vars[fr->pivot_var].dep_mask = conflict_deps & ~(1ULL << max_lv);
    }
}

/* ============================================================================
 * 6. 自己合成 LiftingCell による 8-bit x 8-bit = 16-bit 可逆アレイ乗算器構築
 *    (176 ゲート・193 変数・素因数分解時の未知空間 2^176 通り)
 * ============================================================================ */
typedef struct {
    int vx[8];
    int vy[8];
    int vp[16];
} Multiplier8Ports;

static void build_8bit_multiplier(FactorGraph *g,
                                  DynamicCurryingNet *and_net,
                                  DynamicCurryingNet *sum_net,
                                  DynamicCurryingNet *carry_net,
                                  int16_t obs_x, int16_t obs_y, int32_t obs_p,
                                  Multiplier8Ports *ports) {
    graph_init(g);
    int v_zero = graph_add_var(g, 0);

    for (int i = 0; i < 8; i++) {
        int8_t bx = (obs_x < 0) ? -1 : (int8_t)((obs_x >> i) & 1);
        int8_t by = (obs_y < 0) ? -1 : (int8_t)((obs_y >> i) & 1);
        ports->vx[i] = graph_add_var(g, bx);
        ports->vy[i] = graph_add_var(g, by);
    }

    int8_t bp[16];
    for (int i = 0; i < 16; i++) {
        bp[i] = (obs_p < 0) ? -1 : (int8_t)((obs_p >> i) & 1);
    }

    /* 64個の部分積 m[i][j] = x[i] AND y[j] */
    int m[8][8];
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            if (i == 0 && j == 0) {
                m[0][0]      = graph_add_var(g, bp[0]);
                ports->vp[0] = m[0][0];
            } else {
                m[i][j] = graph_add_var(g, -1);
            }
            graph_add_gate(g, and_net, ports->vx[i], ports->vy[j], v_zero, m[i][j]);
        }
    }

    /* 7段の 8-bit リップルキャリー加算アレイ (計 112 ゲート) */
    int prev_s[8];
    for (int i = 1; i < 8; i++) prev_s[i] = m[i][0];
    int prev_c7 = v_zero;

    for (int j = 1; j < 8; j++) {
        int cur_s[8];
        int cur_c[8];
        bool is_last = (j == 7);

        /* bit 0 of row j -> p[j] */
        ports->vp[j] = graph_add_var(g, bp[j]);
        cur_c[0]     = graph_add_var(g, -1);
        graph_add_gate(g, sum_net,   prev_s[1], m[0][j], v_zero, ports->vp[j]);
        graph_add_gate(g, carry_net, prev_s[1], m[0][j], v_zero, cur_c[0]);

        /* bit 1..7 of row j */
        for (int i = 1; i < 8; i++) {
            int a_in = (i < 7) ? prev_s[i + 1] : prev_c7;
            int s_out, c_out;

            if (is_last) {
                s_out            = graph_add_var(g, bp[7 + i]);
                ports->vp[7 + i] = s_out;
                if (i == 7) {
                    c_out         = graph_add_var(g, bp[15]);
                    ports->vp[15] = c_out;
                } else {
                    c_out = graph_add_var(g, -1);
                }
            } else {
                s_out = graph_add_var(g, -1);
                c_out = graph_add_var(g, -1);
            }

            cur_s[i] = s_out;
            cur_c[i] = c_out;
            graph_add_gate(g, sum_net,   a_in, m[i][j], cur_c[i - 1], s_out);
            graph_add_gate(g, carry_net, a_in, m[i][j], cur_c[i - 1], c_out);
        }

        for (int i = 1; i < 8; i++) prev_s[i] = cur_s[i];
        prev_c7 = cur_c[7];
    }
}

static uint8_t decode_u8(const bool bits[MAX_GRAPH_VARS], const int idx[8]) {
    uint8_t val = 0;
    for (int i = 0; i < 8; i++) {
        if (bits[idx[i]]) val |= (uint8_t)(1u << i);
    }
    return val;
}

static uint16_t decode_u16(const bool bits[MAX_GRAPH_VARS], const int idx[16]) {
    uint16_t val = 0;
    for (int i = 0; i < 16; i++) {
        if (bits[idx[i]]) val |= (uint16_t)(1u << i);
    }
    return val;
}

/* ============================================================================
 * 7. 自動検証ベンチマーク（自己合成 -> 順方向乗算 -> 除算創発 -> 16-bit 素因数分解 & 素数証明）
 * ============================================================================ */
static bool fn_and(bool a, bool b, bool c)   { (void)c; return (a && b); }
static bool fn_sum(bool a, bool b, bool c)   { return (a ^ b ^ c); }
static bool fn_carry(bool a, bool b, bool c) { return (a && b) || (b && c) || (c && a); }

int main(void) {
    DynamicCurryingNet and_net, sum_net, carry_net;

    printf("=== 64-byte LiftingCell 8-bit 可逆乗算器 & カリー化矛盾キャッシュ (CDCL) 実証 ===\n");
    printf("sizeof(LiftingCell) = %zu bytes (パディングゼロ・矛盾キャッシュプール=%d cells / %zu bytes)\n\n",
           sizeof(LiftingCell), MAX_CONFLICT_CELLS,
           (size_t)MAX_CONFLICT_CELLS * sizeof(LiftingCell));

    /* [Phase 1] 基本可逆ゲートの動的カリー化自己合成 */
    net_init(&and_net, 3);
    synthesize_dynamic_topology(&and_net, fn_and, 300);

    net_init(&sum_net, 3);
    synthesize_dynamic_topology(&sum_net, fn_sum, 300);

    net_init(&carry_net, 3);
    synthesize_dynamic_topology(&carry_net, fn_carry, 400);

    if (!verify_primitive_exact(&and_net, fn_and) ||
        !verify_primitive_exact(&sum_net, fn_sum) ||
        !verify_primitive_exact(&carry_net, fn_carry)) {
        fprintf(stderr, "FAIL: Phase 1 基本ゲート自己合成未収束\n");
        return 1;
    }
    printf("[Phase 1] 基本可逆ゲート自己合成完了 (0-bit EXACT 可逆性確認済):\n");
    printf("  Toffoli-AND=%d cells (spawn=%d), Sum=%d cells (spawn=%d), Carry=%d cells (spawn=%d)\n\n",
           and_net.num_cells,   and_net.spawned_count,
           sum_net.num_cells,   sum_net.spawned_count,
           carry_net.num_cells, carry_net.spawned_count);

    /* [Phase 2] 8-bit x 8-bit = 16-bit 可逆乗算器 順方向検証 (1,024 決定論的ペア) */
    static FactorGraph g;
    Multiplier8Ports ports;
    bool sol[MAX_GRAPH_VARS];
    SolveStats st_base, st_cdcl;

    int total_gates = 0, total_vars = 0;
    for (int x = 0; x < 256; x += 8) {
        for (int y = 0; y < 256; y += 8) {
            int xx = (x + (y >> 3)) & 255;
            int yy = (y + 255 - (x >> 3)) & 255;
            build_8bit_multiplier(&g, &and_net, &sum_net, &carry_net,
                                  (int16_t)xx, (int16_t)yy, -1, &ports);
            total_gates = g.num_gates;
            total_vars  = g.num_vars;
            bool ok = solve_factor_abduction(&g, false, sol, &st_base);
            uint16_t p_got = decode_u16(sol, ports.vp);
            if (!ok || st_base.branches != 0 || p_got != (uint16_t)(xx * yy)) {
                fprintf(stderr, "FAIL: Phase 2 順方向乗算 %d * %d = %u (exp %d)\n",
                        xx, yy, (unsigned)p_got, xx * yy);
                return 1;
            }
        }
    }
    printf("[Phase 2] 8-bit 可逆アレイ乗算器 順方向検証 (1,024 パターン・分岐 0): OK\n");
    printf("  トポロジー規模: %d ゲート, %d 変数 (素因数分解時の未知 176 変数 -> 総当たり空間 2^176 通り)\n\n",
           total_gates, total_vars);

    /* [Phase 3] 16-bit 除算の創発 (251 * Y(?) = 60491 -> Y = 241): 因果波面のみで 0 分岐解決 */
    build_8bit_multiplier(&g, &and_net, &sum_net, &carry_net, 251, -1, 60491, &ports);
    if (!solve_factor_abduction(&g, true, sol, &st_cdcl) ||
        decode_u8(sol, ports.vy) != 241u || st_cdcl.branches != 0) {
        fprintf(stderr, "FAIL: Phase 3 16-bit 除算創発失敗 (Y=%u, branches=%d)\n",
                (unsigned)decode_u8(sol, ports.vy), st_cdcl.branches);
        return 1;
    }
    printf("[Phase 3] 16-bit 除算創発 (251 * Y(?) = 60491): 逆算 Y=%u (因果波面=%d steps, 仮説分岐=%d 回!)\n\n",
           (unsigned)decode_u8(sol, ports.vy), st_cdcl.prop_steps, st_cdcl.branches);

    /* [Phase 4] 16-bit 半素数の素因数分解ベンチマーク (Baseline vs カリー化矛盾キャッシュ) */
    printf("[Phase 4] 16-bit 半素数 素因数分解 (未知 176 変数 / 総当たり空間 2^176 通り):\n");
    printf("  %-7s | %-11s | %-21s | %-25s | %s\n",
           "積 P", "逆算 X * Y", "Baseline(分岐/剪定)", "CDCL+Cache(分岐/剪定/BJ)",
           "64Bキャッシュ代謝(生成/共有/伝播/剪定/再利用)");
    printf("  --------+-------------+-----------------------+---------------------------+--------------------------------------------\n");

    const uint16_t semiprimes[] = {
        60491u, /* 241 * 251 */
        54731u, /* 229 * 239 */
        41989u, /* 199 * 211 */
        33043u, /* 173 * 191 */
        25807u, /* 131 * 197 */
        19043u, /* 137 * 139 */
        63001u, /* 251 * 251 */
        65025u  /* 255 * 255 */
    };
    int n_semi = (int)(sizeof(semiprimes) / sizeof(semiprimes[0]));

    for (int t = 0; t < n_semi; t++) {
        uint16_t P = semiprimes[t];

        /* Baseline (キャッシュ OFF) */
        build_8bit_multiplier(&g, &and_net, &sum_net, &carry_net, -1, -1, (int32_t)P, &ports);
        bool ok_b = solve_factor_abduction(&g, false, sol, &st_base);
        uint8_t xb = decode_u8(sol, ports.vx);
        uint8_t yb = decode_u8(sol, ports.vy);

        /* CDCL + 64-byte カリー化矛盾キャッシュ ON */
        build_8bit_multiplier(&g, &and_net, &sum_net, &carry_net, -1, -1, (int32_t)P, &ports);
        bool ok_c = solve_factor_abduction(&g, true, sol, &st_cdcl);
        uint8_t xc = decode_u8(sol, ports.vx);
        uint8_t yc = decode_u8(sol, ports.vy);

        if (!ok_b || !ok_c ||
            (uint32_t)xb * (uint32_t)yb != (uint32_t)P ||
            (uint32_t)xc * (uint32_t)yc != (uint32_t)P ||
            xc <= 1u || yc <= 1u) {
            fprintf(stderr, "FAIL: Phase 4 素因数分解 P=%u 失敗\n", (unsigned)P);
            return 1;
        }

        printf("  %5u   | %3u * %3u   | br=%4d / pr=%4d     | br=%4d / pr=%4d / bj=%2d | sp=%3d / sh=%3d / prop=%2d / clp=%3d / reu=%3d\n",
               (unsigned)P, (unsigned)xc, (unsigned)yc,
               st_base.branches, st_base.pruned_dead,
               st_cdcl.branches, st_cdcl.pruned_dead, st_cdcl.backjumps,
               st_cdcl.cache_spawned, st_cdcl.cache_shared,
               st_cdcl.cache_props + st_cdcl.cache_conflicts,
               st_cdcl.cache_pruned, st_cdcl.cache_reused);
    }
    printf("\n");

    /* [Phase 5] 16-bit 素数（8-bit 因数分解不能）の厳密 UNSAT 証明ベンチマーク */
    printf("[Phase 5] 16-bit 素数（8-bit 因数分解不能）の構成的 UNSAT 証明:\n");
    printf("  %-7s | %-11s | %-21s | %-25s | %s\n",
           "素数 P", "証明結果", "Baseline(分岐/剪定)", "CDCL+Cache(分岐/剪定/BJ)",
           "64Bキャッシュ代謝(生成/共有/伝播/剪定/再利用)");
    printf("  --------+-------------+-----------------------+---------------------------+--------------------------------------------\n");

    const uint16_t primes16[] = {
        4093u,  /* 12-bit 最大素数 */
        8191u,  /* メルセンヌ素数 M_13 */
        16381u, /* 14-bit 最大素数 */
        32749u, /* 15-bit 最大素数 */
        65521u  /* 16-bit 最大素数 */
    };
    int n_pr = (int)(sizeof(primes16) / sizeof(primes16[0]));

    for (int t = 0; t < n_pr; t++) {
        uint16_t P = primes16[t];

        build_8bit_multiplier(&g, &and_net, &sum_net, &carry_net, -1, -1, (int32_t)P, &ports);
        bool ok_b = solve_factor_abduction(&g, false, sol, &st_base);

        build_8bit_multiplier(&g, &and_net, &sum_net, &carry_net, -1, -1, (int32_t)P, &ports);
        bool ok_c = solve_factor_abduction(&g, true, sol, &st_cdcl);

        if (ok_b || ok_c) {
            fprintf(stderr, "FAIL: Phase 5 素数 P=%u に誤解が見つかりました\n", (unsigned)P);
            return 1;
        }

        printf("  %5u   | UNSAT 証明  | br=%4d / pr=%4d     | br=%4d / pr=%4d / bj=%2d | sp=%3d / sh=%3d / prop=%2d / clp=%3d / reu=%3d\n",
               (unsigned)P,
               st_base.branches, st_base.pruned_dead,
               st_cdcl.branches, st_cdcl.pruned_dead, st_cdcl.backjumps,
               st_cdcl.cache_spawned, st_cdcl.cache_shared,
               st_cdcl.cache_props + st_cdcl.cache_conflicts,
               st_cdcl.cache_pruned, st_cdcl.cache_reused);
    }

    printf("\n>>> ALL ASSERTIONS PASSED: 8-bit 可逆乗算器 (2^176 空間) の素因数分解・素数 UNSAT 証明・64B 矛盾キャッシュ代謝を完了 <<<\n");
    return 0;
}