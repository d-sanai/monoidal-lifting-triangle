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
    uint8_t cnt = 0;
    while (x != 0) {
        cnt += (uint8_t)(x & 1u);
        x >>= 1;
    }
    return cnt;
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
 * 2. 64-byte 均一 LiftingCell（L1キャッシュライン完全整列）
 * ============================================================================ */
typedef enum {
    MODE_EXACT = 0,   /* ビット完全可逆性検証 (μ ∘ Δ = id) */
    MODE_LEARN = 1,   /* 基底直交学習 + 動的カリー化 */
    MODE_EVAL  = 2    /* 純粋演繹評価 */
} ExecMode;

typedef struct LiftingCell {
    /* --- 動的配線トポロジー (24 bytes) --- */
    struct LiftingCell *in_s;       /* 前提（骨格）入力 */
    struct LiftingCell *in_g;       /* カリー化ゲート入力 (NULL なら定数 1.0) */
    struct LiftingCell *in_v;       /* 下位からの変分（残差）直列入力 */

    /* --- 回転子重みと Δ / μ 状態 (32 bytes) --- */
    c16_t w;                        /* 複素回転子重み */
    c16_t s_eff;                    /* 実効前提: Q(g * s) */
    c16_t p;                        /* 局所予測変分: Q(w * s_eff) */
    c16_t r;                        /* 未説明残差変分: v ⊖ p */
    c16_t up_s;                     /* 上位へ渡す骨格出力 */
    c16_t up_r;                     /* 上位へ渡す変分出力 */
    c16_t td_r;                     /* 上位から降りてくる合流変分 */
    c16_t restored;                 /* モノイド合流後の復元値 */

    /* --- 局所双方向伝播 & メタデータ (8 bytes -> 64-bit 環境で厳密に 64 bytes) --- */
    int32_t adj_s;                  /* 骨格チャネルへの逆流位相トルク蓄積 */
    uint8_t active;                 /* 実効前提 s_eff が非ゼロか */
    uint8_t var_mask;               /* 配線トポロジーから自動導出される依存変数集合 */
    uint8_t order;                  /* カリー化次数 */
    uint8_t flags;                  /* アライメント整合用 */
} LiftingCell;

#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFULL
typedef char lifting_cell_must_be_64_bytes[(sizeof(LiftingCell) == 64) ? 1 : -1];
#endif

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
    c->in_s     = NULL;
    c->in_g     = NULL;
    c->in_v     = NULL;
    c->w        = (c16_t){0, 0};
    c->s_eff    = (c16_t){0, 0};
    c->p        = (c16_t){0, 0};
    c->r        = (c16_t){0, 0};
    c->up_s     = (c16_t){0, 0};
    c->up_r     = (c16_t){0, 0};
    c->td_r     = (c16_t){0, 0};
    c->restored = (c16_t){0, 0};
    c->adj_s    = 0;
    c->active   = 0;
    c->var_mask = 0;
    c->order    = 0;
    c->flags    = 0;
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

    net->num_cells++;
    net->spawned_count++;
    return true;
}

/* ============================================================================
 * 3. 全セル同一のボトムアップ (Δ) / トップダウン (μ) / 局所逆流伝播 (Δ†)
 * ============================================================================ */
static void cell_bottom_up(LiftingCell *c) {
    c16_t g = (c->in_g != NULL) ? c->in_g->up_s : (c16_t){Q14_ONE, 0};
    c->s_eff  = c16_mul_q14(g, c->in_s->up_s);
    c->active = (uint8_t)(c16_l1_norm(c->s_eff) > 0);
    c->p      = c16_mul_q14(c->w, c->s_eff);
    c->r      = c16_sub_mod(c->in_v->up_r, c->p);
    c->up_s   = c->s_eff;
    c->up_r   = c->r;
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
    if (!c->active) return;
    if (c->var_mask != active_input_mask) return;
    update_complex_weight_mod(&c->w, c->r, c->s_eff, lr_shift);
}

static c16_t net_forward_continuous(DynamicCurryingNet *net,
                                    const int16_t u_in[MAX_VARS],
                                    c16_t target_var) {
    for (uint8_t i = 0; i < net->num_vars; i++) {
        c16_t val = { u_in[i], 0 };
        net->in_ports[i].up_s  = val;
        net->in_ports[i].up_r  = val;
        net->in_ports[i].adj_s = 0;
    }
    net->in_target.up_s  = target_var;
    net->in_target.up_r  = target_var;
    net->in_target.adj_s = 0;

    for (int i = 0; i < net->num_cells; i++) {
        net->cells[i].adj_s = 0;
        cell_bottom_up(&net->cells[i]);
    }
    return net->cells[net->num_cells - 1].r;
}

static void net_backward_adjoint(DynamicCurryingNet *net, c16_t top_r, int32_t mem_w) {
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
        int32_t total   = c->adj_s + (int32_t)(((int64_t)f_local * mem_w) >> 4);

        if (total == 0) continue;

        if (c->in_g == NULL) {
            c->in_s->adj_s += total;
        } else {
            int32_t g_val = c->in_g->up_s.re;
            int32_t s_val = c->in_s->up_s.re;
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

/* 1ゲートの離散残差ノルムを 0-bit 厳密評価 (O(num_cells)) */
static inline int32_t eval_gate_discrete_err(DynamicCurryingNet *net,
                                             const bool b_in[MAX_VARS],
                                             bool b_out) {
    c16_t t_var = b_out ? (c16_t){0, Q14_NOT} : (c16_t){0, 0};
    (void)net_step_discrete(net, b_in, t_var, MODE_EXACT);
    return c16_l1_norm(net->cells[net->num_cells - 1].r);
}

/* ============================================================================
 * 4. 動的カリー化トポロジー自己合成
 * ============================================================================ */
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
            bool bits[MAX_VARS] = {
                (mask & 1) != 0,
                (mask & 2) != 0,
                (mask & 4) != 0
            };
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
                    if (!net->cells[u].active) continue;
                    for (int v = u + 1; v < net->num_cells; v++) {
                        if (!net->cells[v].active) continue;
                        if ((net->cells[u].var_mask & net->cells[v].var_mask) != 0) continue;
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

/* ============================================================================
 * 5. 計算誤差ゼロ因果波面 ＋ トポロジー結合・完全決定論的アブダクションエンジン
 * ============================================================================ */
#define MAX_GRAPH_VARS  300
#define MAX_GRAPH_GATES 160
#define MAX_CURRY_BONDS 320

typedef struct {
    int16_t u;          /* 連続位相状態 [0, 16384] */
    int32_t conf;       /* 観測データ由来の因果確信度 [0, 16384] (仮位相とは厳密分離) */
    int32_t vel;        /* 位相慣性モーメンタム (未確定コアの連続緩和用) */
    int32_t force;      /* 局所逆流トルク蓄積 */
    bool    clamped;    /* 外部観測データによる固定変数フラグ */
    bool    dirty;      /* 未解決ゲートに関与中か */
} PhaseVar;

typedef struct {
    DynamicCurryingNet *net;          /* 64-byte LiftingCell で自己合成された回路 */
    int                 in_var[3];    /* 入力変数インデックス */
    int                 out_var;      /* 出力変数インデックス */
    int32_t             mem_w;        /* 残差記憶重み (Q4) */
} LiftingGate;

typedef struct {
    int var_u;                        /* カリー化セルで結ばれた変数ペア (u, v) */
    int var_v;
} CurriedBond;

typedef struct {
    PhaseVar    vars[MAX_GRAPH_VARS];
    LiftingGate gates[MAX_GRAPH_GATES];
    CurriedBond bonds[MAX_CURRY_BONDS];
    int         num_vars;
    int         num_gates;
    int         num_bonds;
} AbductionGraph;

static void graph_init(AbductionGraph *g) {
    g->num_vars  = 0;
    g->num_gates = 0;
    g->num_bonds = 0;
}

static int graph_add_var(AbductionGraph *g, int8_t obs_val) {
    int id = g->num_vars++;
    PhaseVar *v = &g->vars[id];
    v->vel   = 0;
    v->force = 0;
    v->dirty = false;
    if (obs_val < 0) {
        /* 未知変数は因果確信度 conf=0、初期位相は閾値未満 8191 (離散 0) に置く */
        v->u       = Q14_HALF - 1;
        v->conf    = 0;
        v->clamped = false;
    } else {
        v->u       = (obs_val != 0) ? Q14_ONE : 0;
        v->conf    = Q14_ONE;
        v->clamped = true;
    }
    return id;
}

static void graph_register_bond(AbductionGraph *g, int u, int v) {
    if (u == v || g->num_bonds >= MAX_CURRY_BONDS) return;
    if (u > v) { int tmp = u; u = v; v = tmp; }
    for (int i = 0; i < g->num_bonds; i++) {
        if (g->bonds[i].var_u == u && g->bonds[i].var_v == v) return;
    }
    g->bonds[g->num_bonds].var_u = u;
    g->bonds[g->num_bonds].var_v = v;
    g->num_bonds++;
}

static void graph_add_gate(AbductionGraph *g, DynamicCurryingNet *net,
                           int in0, int in1, int in2, int out_v) {
    LiftingGate *gt = &g->gates[g->num_gates++];
    gt->net       = net;
    gt->in_var[0] = in0;
    gt->in_var[1] = in1;
    gt->in_var[2] = in2;
    gt->out_var   = out_v;
    gt->mem_w     = 16;

    for (int c = 0; c < net->num_cells; c++) {
        if (net->cells[c].in_g == NULL) continue;
        uint8_t mask = net->cells[c].var_mask;
        for (uint8_t a = 0; a < net->num_vars; a++) {
            if ((mask & (1u << a)) == 0) continue;
            for (uint8_t b = (uint8_t)(a + 1); b < net->num_vars; b++) {
                if ((mask & (1u << b)) == 0) continue;
                graph_register_bond(g, gt->in_var[a], gt->in_var[b]);
            }
        }
    }
}

static inline int32_t var_polarization(const PhaseVar *pv) {
    if (pv->clamped) return Q14_ONE;
    int32_t d = (int32_t)pv->u * 2 - Q14_ONE;
    return (d >= 0) ? d : -d;
}

/* 乱数ゼロ・総当たりゼロの完全決定論的アブダクションソルバ
 * 1) 因果確信度 conf に基づく離散局所微分 (計算誤差 0-bit の因果波面伝播)
 * 2) 因果波面で残った多変数同時未知コア (conf==0) のみへの連続位相緩和 + カリー化結合相転移 */
static int solve_abduction_local(AbductionGraph *g, int max_iters,
                                 bool out_bits[MAX_GRAPH_VARS],
                                 int *kicks_out) {
    int     best_violated  = g->num_gates + 1;
    int64_t best_pol       = -1;
    int     stagnation_cnt = 0;
    int     var_cursor     = 0;
    int     bond_cursor    = 0;
    int     total_kicks    = 0;

    int32_t prop_conf[MAX_GRAPH_VARS];
    bool    prop_val[MAX_GRAPH_VARS];
    bool    prop_conflict[MAX_GRAPH_VARS];

    for (int iter = 0; iter < max_iters; iter++) {
        /* --------------------------------------------------------------------
         * Step A: 局所離散残差チェック (O(G)) と全変数の分極検証
         * -------------------------------------------------------------------- */
        int     violated      = 0;
        int64_t total_pol     = 0;
        bool    all_polarized = true;

        for (int v = 0; v < g->num_vars; v++) {
            g->vars[v].dirty = false;
            g->vars[v].force = 0;
            prop_conf[v]     = 0;
            prop_val[v]      = false;
            prop_conflict[v] = false;

            int32_t pol = var_polarization(&g->vars[v]);
            total_pol += pol;
            if (!g->vars[v].clamped && pol < 8192) {
                all_polarized = false;
            }
        }

        for (int gi = 0; gi < g->num_gates; gi++) {
            LiftingGate *gt = &g->gates[gi];
            bool b_in[MAX_VARS] = {
                g->vars[gt->in_var[0]].u >= Q14_HALF,
                g->vars[gt->in_var[1]].u >= Q14_HALF,
                g->vars[gt->in_var[2]].u >= Q14_HALF
            };
            bool b_out = (g->vars[gt->out_var].u >= Q14_HALF);

            if (eval_gate_discrete_err(gt->net, b_in, b_out) != 0) {
                violated++;
                g->vars[gt->in_var[0]].dirty = true;
                g->vars[gt->in_var[1]].dirty = true;
                g->vars[gt->in_var[2]].dirty = true;
                g->vars[gt->out_var].dirty   = true;
            }
        }

        if (violated == 0 && all_polarized) {
            for (int v = 0; v < g->num_vars; v++) {
                out_bits[v] = (g->vars[v].u >= Q14_HALF);
            }
            if (kicks_out != NULL) *kicks_out = total_kicks;
            return iter;
        }

        /* --------------------------------------------------------------------
         * Step B: 計算誤差 0-bit の因果波面伝播 (確信度 conf > 0 からの離散局所微分)
         * -------------------------------------------------------------------- */
        for (int gi = 0; gi < g->num_gates; gi++) {
            LiftingGate *gt = &g->gates[gi];
            int term[4] = {
                gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var
            };
            bool b_cur[4] = {
                g->vars[term[0]].u >= Q14_HALF,
                g->vars[term[1]].u >= Q14_HALF,
                g->vars[term[2]].u >= Q14_HALF,
                g->vars[term[3]].u >= Q14_HALF
            };

            for (int k = 0; k < 4; k++) {
                int vid = term[k];
                if (g->vars[vid].clamped) continue;

                /* 他の3端子すべての因果確信度の最小値 m_k を算出 */
                int32_t m_k = Q14_ONE;
                for (int j = 0; j < 4; j++) {
                    if (j != k && g->vars[term[j]].conf < m_k) {
                        m_k = g->vars[term[j]].conf;
                    }
                }
                if (m_k <= 0) continue;

                int32_t next_conf = (m_k > 64) ? (m_k - 64) : 1;
                if (next_conf <= g->vars[vid].conf) continue;

                /* 端子 k に対する 0-bit 離散局所微分 (E_0 vs E_1) */
                bool b0_in[MAX_VARS] = { b_cur[0], b_cur[1], b_cur[2] };
                bool b1_in[MAX_VARS] = { b_cur[0], b_cur[1], b_cur[2] };
                bool b0_out = b_cur[3];
                bool b1_out = b_cur[3];

                if (k < 3) {
                    b0_in[k] = false;
                    b1_in[k] = true;
                } else {
                    b0_out = false;
                    b1_out = true;
                }

                int32_t e0 = eval_gate_discrete_err(gt->net, b0_in, b0_out);
                int32_t e1 = eval_gate_discrete_err(gt->net, b1_in, b1_out);

                if ((e0 == 0) != (e1 == 0)) {
                    bool deduced = (e1 == 0);
                    if (next_conf > prop_conf[vid]) {
                        prop_conf[vid]     = next_conf;
                        prop_val[vid]      = deduced;
                        prop_conflict[vid] = false;
                    } else if (next_conf == prop_conf[vid] && prop_val[vid] != deduced) {
                        prop_conflict[vid] = true;
                    }
                }
            }
        }

        bool causal_progress = false;
        for (int v = 0; v < g->num_vars; v++) {
            if (!g->vars[v].clamped && !prop_conflict[v] && prop_conf[v] > g->vars[v].conf) {
                g->vars[v].u    = prop_val[v] ? Q14_ONE : 0;
                g->vars[v].conf = prop_conf[v];
                g->vars[v].vel  = 0;
                causal_progress = true;
            }
        }

        /* 因果波面が前進している間は計算誤差を一切混ぜずに次ステップへ進む */
        if (causal_progress) {
            stagnation_cnt = 0;
            continue;
        }

        /* --------------------------------------------------------------------
         * Step C: 因果波面で残った多変数同時未知コア (conf == 0) への連続位相緩和
         *         確定済み変数 (conf > 0) は保護し、未確定コアだけを協調収束させる
         * -------------------------------------------------------------------- */
        if (violated == 0) {
            for (int v = 0; v < g->num_vars; v++) {
                PhaseVar *pv = &g->vars[v];
                if (pv->clamped || pv->conf > 0) continue;
                int32_t snap = (pv->u >= Q14_HALF) ? 2048 : -2048;
                pv->vel = 0;
                pv->u   = (int16_t)clamp_i32((int32_t)pv->u + snap, 0, Q14_ONE);
            }
            stagnation_cnt = 0;
            continue;
        }

        for (int gi = 0; gi < g->num_gates; gi++) {
            LiftingGate *gt = &g->gates[gi];
            int term[4] = {
                gt->in_var[0], gt->in_var[1], gt->in_var[2], gt->out_var
            };
            bool has_unsolved_core = false;
            for (int k = 0; k < 4; k++) {
                if (!g->vars[term[k]].clamped && g->vars[term[k]].conf == 0) {
                    has_unsolved_core = true;
                    break;
                }
            }
            if (!has_unsolved_core) continue;

            int16_t u_in[MAX_VARS] = {
                g->vars[term[0]].u, g->vars[term[1]].u, g->vars[term[2]].u
            };
            int16_t u_out  = g->vars[term[3]].u;
            c16_t   t_cont = { 0, quantize_q14((int64_t)Q14_NOT * u_out) };

            c16_t   top_r    = net_forward_continuous(gt->net, u_in, t_cont);
            int32_t err_norm = c16_l1_norm(top_r);

            if (err_norm > 2048) {
                if (gt->mem_w < 32) gt->mem_w += 1;
            } else {
                if (gt->mem_w > 16) gt->mem_w -= 1;
            }

            net_backward_adjoint(gt->net, top_r, gt->mem_w);
            for (int k = 0; k < 3; k++) {
                int vid = term[k];
                if (g->vars[vid].clamped || g->vars[vid].conf > 0) continue;
                g->vars[vid].force += gt->net->in_ports[k].adj_s;
            }

            if (!g->vars[term[3]].clamped && g->vars[term[3]].conf == 0 && err_norm > 0) {
                int32_t out_e_im = top_r.im;
                if (top_r.im == Q14_NOT) {
                    out_e_im = (u_out > Q14_HALF) ? -32768 : 32767;
                }
                g->vars[term[3]].force += (int32_t)(((int64_t)(out_e_im * 2) * gt->mem_w) >> 4);
            }
        }

        for (int v = 0; v < g->num_vars; v++) {
            PhaseVar *pv = &g->vars[v];
            if (pv->clamped || pv->conf > 0) continue;

            int32_t f_step   = clamp_i32(pv->force >> 5, -2048, 2048);
            int32_t bin_pull = (!pv->dirty) ? (((int32_t)pv->u - Q14_HALF) >> 6) : 0;

            pv->vel = (pv->vel >> 2) + f_step + bin_pull;
            pv->vel = clamp_i32(pv->vel, -2048, 2048);
            pv->u   = (int16_t)clamp_i32((int32_t)pv->u + pv->vel, 0, Q14_ONE);
        }

        /* Step D: 未確定コアがパリティ障壁で停滞したときのみ、カリー化結合ペアを反対極へ決定論的反転 */
        if (violated < best_violated || total_pol > best_pol + 128) {
            if (violated < best_violated) best_violated = violated;
            if (total_pol > best_pol)     best_pol      = total_pol;
            stagnation_cnt = 0;
        } else {
            stagnation_cnt++;
            if (stagnation_cnt >= 12) {
                bool jumped = false;

                if ((total_kicks % 2 == 0) && g->num_bonds > 0) {
                    for (int scan = 0; scan < g->num_bonds; scan++) {
                        int bidx = (bond_cursor + scan) % g->num_bonds;
                        PhaseVar *pu = &g->vars[g->bonds[bidx].var_u];
                        PhaseVar *pv = &g->vars[g->bonds[bidx].var_v];
                        if (pu->clamped || pv->clamped || pu->conf > 0 || pv->conf > 0) continue;
                        if (!pu->dirty || !pv->dirty) continue;

                        pu->u   = (pu->u >= Q14_HALF) ? 0 : Q14_ONE;
                        pv->u   = (pv->u >= Q14_HALF) ? 0 : Q14_ONE;
                        pu->vel = 0;
                        pv->vel = 0;
                        bond_cursor = (bidx + 1) % g->num_bonds;
                        jumped = true;
                        break;
                    }
                }

                if (!jumped) {
                    for (int scan = 0; scan < g->num_vars; scan++) {
                        int idx = (var_cursor + scan) % g->num_vars;
                        PhaseVar *pv = &g->vars[idx];
                        if (pv->clamped || pv->conf > 0 || !pv->dirty) continue;

                        pv->u      = (pv->u >= Q14_HALF) ? 0 : Q14_ONE;
                        pv->vel    = 0;
                        var_cursor = (idx + 1) % g->num_vars;
                        jumped     = true;
                        break;
                    }
                }

                if (jumped) {
                    best_pol      = -1;
                    best_violated = g->num_gates + 1;
                    total_kicks++;
                }
                stagnation_cnt = 0;
            }
        }
    }

    for (int v = 0; v < g->num_vars; v++) {
        out_bits[v] = (g->vars[v].u >= Q14_HALF);
    }
    if (kicks_out != NULL) *kicks_out = total_kicks;
    return -1;
}

/* ============================================================================
 * 6. 自動検証ベンチマーク（1-bit / Toffoli3 / 16-bit複合 / 32-bit / 64-bit）
 * ============================================================================ */
static bool fn_sum(bool a, bool b, bool c)      { return (a ^ b ^ c); }
static bool fn_carry(bool a, bool b, bool c)    { return (a && b) || (b && c) || (c && a); }
static bool fn_toffoli3(bool a, bool b, bool c) { return (a && b && c); }

static void verify_1bit_case(const char *label,
                             DynamicCurryingNet *sum_net,
                             DynamicCurryingNet *carry_net,
                             int8_t obs_a, int8_t obs_b, int8_t obs_c,
                             int8_t obs_s, int8_t obs_co,
                             bool exp_a, bool exp_b, bool exp_c,
                             bool exp_s, bool exp_co) {
    AbductionGraph g;
    graph_init(&g);
    int va  = graph_add_var(&g, obs_a);
    int vb  = graph_add_var(&g, obs_b);
    int vc  = graph_add_var(&g, obs_c);
    int vs  = graph_add_var(&g, obs_s);
    int vco = graph_add_var(&g, obs_co);

    graph_add_gate(&g, sum_net,   va, vb, vc, vs);
    graph_add_gate(&g, carry_net, va, vb, vc, vco);

    bool sol[MAX_GRAPH_VARS] = {false};
    int  kicks = 0;
    int  iters = solve_abduction_local(&g, 1000, sol, &kicks);
    if (iters < 0 ||
        sol[va] != exp_a || sol[vb] != exp_b || sol[vc] != exp_c ||
        sol[vs] != exp_s || sol[vco] != exp_co) {
        fprintf(stderr, "FAIL: %s (iters=%d, kicks=%d)\n", label, iters, kicks);
        exit(1);
    }
    printf("  %-34s -> A=%d B=%d C=%d | Sum=%d Carry=%d  (収束=%3d steps, kick=%d)\n",
           label, sol[va], sol[vb], sol[vc], sol[vs], sol[vco], iters, kicks);
}

int main(void) {
    DynamicCurryingNet sum_net, carry_net, tof3_net;

    printf("=== 64-byte LiftingCell: 計算誤差ゼロ因果波面 ＋ トポロジー結合アブダクション ===\n");
    printf("sizeof(LiftingCell) = %zu bytes (L1キャッシュライン整列・乱数ゼロ・総当たりゼロ)\n\n",
           sizeof(LiftingCell));

    /* [Phase 1] 動的カリー化による基本回路の自己合成 */
    net_init(&sum_net, 3);
    synthesize_dynamic_topology(&sum_net, fn_sum, 300);

    net_init(&carry_net, 3);
    synthesize_dynamic_topology(&carry_net, fn_carry, 400);

    net_init(&tof3_net, 3);
    synthesize_dynamic_topology(&tof3_net, fn_toffoli3, 400);

    printf("[Phase 1] 可変トポロジー自己合成完了: Sum=%d cells, Carry=%d cells, Toffoli3=%d cells\n\n",
           sum_net.num_cells, carry_net.num_cells, tof3_net.num_cells);

    /* [Phase 2] 1-bit 全加算器 & 3入力 Toffoli */
    printf("[Phase 2] 1-bit 可逆全加算器 & Toffoli3 局所アブダクション:\n");
    verify_1bit_case("1. 順方向加算 (1+1+1)",             &sum_net, &carry_net,  1,  1,  1, -1, -1, 1, 1, 1, 1, 1);
    verify_1bit_case("2. 1変数逆算 (1+?+0 -> S=0,C=1)",   &sum_net, &carry_net,  1, -1,  0,  0,  1, 1, 1, 0, 0, 1);
    verify_1bit_case("3. 2変数逆算 (?+?+1 -> S=1,C=1)",   &sum_net, &carry_net, -1, -1,  1,  1,  1, 1, 1, 1, 1, 1);
    verify_1bit_case("4. 2変数逆算 (?+?+1 -> S=1,C=0)",   &sum_net, &carry_net, -1, -1,  1,  1,  0, 0, 0, 1, 1, 0);
    verify_1bit_case("5. 3変数全逆算 (?+?+? -> S=1,C=1)", &sum_net, &carry_net, -1, -1, -1,  1,  1, 1, 1, 1, 1, 1);

    {
        AbductionGraph g;
        graph_init(&g);
        int va = graph_add_var(&g, -1);
        int vb = graph_add_var(&g, -1);
        int vc = graph_add_var(&g, -1);
        int vo = graph_add_var(&g,  1);
        graph_add_gate(&g, &tof3_net, va, vb, vc, vo);

        bool sol[MAX_GRAPH_VARS] = {false};
        int  kicks = 0;
        int  iters = solve_abduction_local(&g, 1000, sol, &kicks);
        if (iters < 0 || !sol[va] || !sol[vb] || !sol[vc]) {
            fprintf(stderr, "FAIL: Toffoli3 局所逆算失敗 (iters=%d)\n", iters);
            return 1;
        }
        printf("  %-34s -> A=%d B=%d C=%d | Out=%d          (収束=%3d steps, kick=%d)\n\n",
               "6. Toffoli3 全逆算 (?&?&?=1)", sol[va], sol[vb], sol[vc], sol[vo], iters, kicks);
    }

    /* [Phase 3] 16-bit / 32-bit / 64-bit 大規模可逆加算器アブダクション */
    printf("[Phase 3] 大規模可逆加算器 局所双方向アブダクション (総当たり空間 2^31 〜 2^127 通り):\n");

    /* ケース 3-A: 16-bit 減算創発 (12345 + B(?) = 54321 -> B = 41976, 未知31変数) */
    {
        const uint16_t A_val = 12345u;
        const uint16_t B_exp = 41976u;
        const uint32_t S_obs = (uint32_t)A_val + (uint32_t)B_exp;

        AbductionGraph g;
        graph_init(&g);
        int va[16], vb[16], vs[16], vc[17];

        vc[0] = graph_add_var(&g, 0);
        for (int b = 0; b < 16; b++) {
            va[b]     = graph_add_var(&g, (int8_t)((A_val >> b) & 1u));
            vb[b]     = graph_add_var(&g, -1);
            vs[b]     = graph_add_var(&g, (int8_t)((S_obs >> b) & 1u));
            vc[b + 1] = graph_add_var(&g, (b == 15) ? (int8_t)((S_obs >> 16) & 1u) : -1);

            graph_add_gate(&g, &sum_net,   va[b], vb[b], vc[b], vs[b]);
            graph_add_gate(&g, &carry_net, va[b], vb[b], vc[b], vc[b + 1]);
        }

        bool sol[MAX_GRAPH_VARS] = {false};
        int  kicks = 0;
        int  iters = solve_abduction_local(&g, 1000, sol, &kicks);
        uint16_t B_rec = 0;
        for (int b = 0; b < 16; b++) {
            if (sol[vb[b]]) B_rec |= (uint16_t)(1u << b);
        }
        if (iters < 0 || B_rec != B_exp) {
            fprintf(stderr, "FAIL: Phase 3-A (B_rec=%u, exp=%u, iters=%d, kicks=%d)\n",
                    (unsigned)B_rec, (unsigned)B_exp, iters, kicks);
            return 1;
        }
        printf("  3-A 16-bit 減算創発   : %u + B(?) = %u  ==>  B=%u (31未知, 収束=%3d steps, kick=%d)\n",
               (unsigned)A_val, (unsigned)S_obs, (unsigned)B_rec, iters, kicks);
    }

    /* ケース 3-B: 16-bit 複合 (b=0..14 交互歯抜け + 最上位 b=15 は A,B 両方未知のハイブリッド!) */
    {
        const uint16_t A_exp = 0xD53Cu;
        const uint16_t B_exp = 0xCE9Bu;
        const uint32_t S_obs = (uint32_t)A_exp + (uint32_t)B_exp; /* 0x1A3D7 */

        AbductionGraph g;
        graph_init(&g);
        int va[16], vb[16], vs[16], vc[17];

        vc[0] = graph_add_var(&g, 0);
        for (int b = 0; b < 16; b++) {
            int8_t oa, ob;
            if (b == 15) {
                oa = -1; ob = -1; /* 最上位ビットは A15, B15 両方同時未知! */
            } else {
                oa = (b % 2 == 1) ? -1 : (int8_t)((A_exp >> b) & 1u);
                ob = (b % 2 == 0) ? -1 : (int8_t)((B_exp >> b) & 1u);
            }
            va[b]     = graph_add_var(&g, oa);
            vb[b]     = graph_add_var(&g, ob);
            vs[b]     = graph_add_var(&g, (int8_t)((S_obs >> b) & 1u));
            vc[b + 1] = graph_add_var(&g, (b == 15) ? (int8_t)((S_obs >> 16) & 1u) : -1);

            graph_add_gate(&g, &sum_net,   va[b], vb[b], vc[b], vs[b]);
            graph_add_gate(&g, &carry_net, va[b], vb[b], vc[b], vc[b + 1]);
        }

        bool sol[MAX_GRAPH_VARS] = {false};
        int  kicks = 0;
        int  iters = solve_abduction_local(&g, 1000, sol, &kicks);
        uint16_t A_rec = 0, B_rec = 0;
        for (int b = 0; b < 16; b++) {
            if (sol[va[b]]) A_rec |= (uint16_t)(1u << b);
            if (sol[vb[b]]) B_rec |= (uint16_t)(1u << b);
        }
        if (iters < 0 || A_rec != A_exp || B_rec != B_exp) {
            fprintf(stderr, "FAIL: Phase 3-B (A=%04X, B=%04X, iters=%d, kicks=%d)\n",
                    (unsigned)A_rec, (unsigned)B_rec, iters, kicks);
            return 1;
        }
        printf("  3-B 16-bit 複合逆算   : 歯抜け+MSB両未知  ==>  A=0x%04X, B=0x%04X (32未知, 収束=%3d steps, kick=%d)\n",
               (unsigned)A_rec, (unsigned)B_rec, iters, kicks);
    }

    /* ケース 3-C: 32-bit 減算創発 (63個の未知変数 / 総当たり空間 2^63 通り) */
    {
        const uint32_t A_val = 0xBA7F5C9Bu;
        const uint32_t B_exp = 0x9E8B6D7Fu;
        const uint64_t S_obs = (uint64_t)A_val + (uint64_t)B_exp;

        AbductionGraph g;
        graph_init(&g);
        int va[32], vb[32], vs[32], vc[33];

        vc[0] = graph_add_var(&g, 0);
        for (int b = 0; b < 32; b++) {
            va[b]     = graph_add_var(&g, (int8_t)((A_val >> b) & 1u));
            vb[b]     = graph_add_var(&g, -1);
            vs[b]     = graph_add_var(&g, (int8_t)((S_obs >> b) & 1ULL));
            vc[b + 1] = graph_add_var(&g, (b == 31) ? (int8_t)((S_obs >> 32) & 1ULL) : -1);

            graph_add_gate(&g, &sum_net,   va[b], vb[b], vc[b], vs[b]);
            graph_add_gate(&g, &carry_net, va[b], vb[b], vc[b], vc[b + 1]);
        }

        bool sol[MAX_GRAPH_VARS] = {false};
        int  kicks = 0;
        int  iters = solve_abduction_local(&g, 1000, sol, &kicks);
        uint32_t B_rec = 0;
        for (int b = 0; b < 32; b++) {
            if (sol[vb[b]]) B_rec |= (1u << b);
        }
        if (iters < 0 || B_rec != B_exp) {
            fprintf(stderr, "FAIL: Phase 3-C (B_rec=0x%08X, exp=0x%08X, iters=%d, kicks=%d)\n",
                    (unsigned)B_rec, (unsigned)B_exp, iters, kicks);
            return 1;
        }
        printf("  3-C 32-bit 減算創発   : 0x%08X + B(?) ==>  B=0x%08X (63未知, 収束=%3d steps, kick=%d)\n",
               (unsigned)A_val, (unsigned)B_rec, iters, kicks);
    }

    /* ケース 3-D: 64-bit 減算創発 (127個の未知変数 / 総当たり空間 2^127 ≈ 1.7e38 通り!) */
    {
        const uint64_t A_val = 0xF1E2D3C4B5A69788ULL;
        const uint64_t B_exp = 0x89ABCDEF01234567ULL;
        const uint64_t S_low = A_val + B_exp;
        const uint8_t  S_cry = (S_low < A_val) ? 1u : 0u; /* 65ビット目の桁上げ */

        AbductionGraph g;
        graph_init(&g);
        int va[64], vb[64], vs[64], vc[65];

        vc[0] = graph_add_var(&g, 0);
        for (int b = 0; b < 64; b++) {
            va[b]     = graph_add_var(&g, (int8_t)((A_val >> b) & 1ULL));
            vb[b]     = graph_add_var(&g, -1);
            vs[b]     = graph_add_var(&g, (int8_t)((S_low >> b) & 1ULL));
            vc[b + 1] = graph_add_var(&g, (b == 63) ? (int8_t)S_cry : -1);

            graph_add_gate(&g, &sum_net,   va[b], vb[b], vc[b], vs[b]);
            graph_add_gate(&g, &carry_net, va[b], vb[b], vc[b], vc[b + 1]);
        }

        bool sol[MAX_GRAPH_VARS] = {false};
        int  kicks = 0;
        int  iters = solve_abduction_local(&g, 1000, sol, &kicks);
        uint64_t B_rec = 0;
        for (int b = 0; b < 64; b++) {
            if (sol[vb[b]]) B_rec |= (1ULL << b);
        }
        if (iters < 0 || B_rec != B_exp) {
            fprintf(stderr, "FAIL: Phase 3-D (B_rec=0x%016llX, exp=0x%016llX, iters=%d, kicks=%d)\n",
                    (unsigned long long)B_rec, (unsigned long long)B_exp, iters, kicks);
            return 1;
        }
        printf("  3-D 64-bit 減算創発   : 0x%016llX + B(?) ==> B=0x%016llX (127未知, 収束=%3d steps, kick=%d)\n",
               (unsigned long long)A_val, (unsigned long long)B_rec, iters, kicks);
    }

    printf("\n>>> ALL ASSERTIONS PASSED: 計算誤差ゼロ因果波面 ＋ トポロジー結合で 2^127 空間の逆算収束を確認 <<<\n");
    return 0;
}