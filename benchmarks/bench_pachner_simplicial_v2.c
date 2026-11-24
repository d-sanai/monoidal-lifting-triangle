/* bench_pachner_simplicial_v2.c — Zero-Heap 2D Pachner Moves (1-3 / 3-1 / 2-2)
 * on Simplicial Meshes (S² Icosahedron & T² Torus) with 0-bit Topological Invariance:
 *   [Test 1] Exact 1-3 / 3-1 Reversibility, 2-2 Vortex Balancing & Link/Tetrahedron Guards
 *   [Test 2] Adaptive 1-3 Refinement -> 3-1 Metabolic Contraction & Free-List Slot Reuse
 *   [Test 3] S² Icosahedron: 100 Random Mixed Pachner Moves (χ=2 & Global H² Monopole 0-bit)
 *   [Test 4] T² Torus: 100 Random Pachner Moves (χ=0, Harmonic H¹ Winding & H² Monopole 0-bit)
 *   [Test 5] 3-Stream (R-G-B) 0-bit Reversible Transport on Deformed S² & T² Meshes */
#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LIFTING_CORE_V2_IMPLEMENTATION
#include "lifting_core_v2.h"

#define PM_MAX_V 256
#define PM_MAX_E 768
#define PM_MAX_F 512

typedef struct {
    /* 固定長ゼロヒーププール & フリーリスト再利用統計 */
    bool vert_alive[PM_MAX_V], vert_ever[PM_MAX_V];
    bool edge_alive[PM_MAX_E], edge_ever[PM_MAX_E];
    bool face_alive[PM_MAX_F], face_ever[PM_MAX_F];

    int n_v_alive, n_e_alive, n_f_alive;
    int reused_v,  reused_e,  reused_f;

    /* 単体接続関係 (辺は標準向き edge_u < edge_v) */
    int    edge_u[PM_MAX_E];
    int    edge_v[PM_MAX_E];
    int    edge_faces[PM_MAX_E][2];
    int8_t edge_f_sign[PM_MAX_E][2];

    /* トーラス等の周期境界シーム巻き数 (edge_u -> edge_v 方向) */
    int16_t edge_wx[PM_MAX_E];
    int16_t edge_wy[PM_MAX_E];

    int    face_verts[PM_MAX_F][3];
    int    face_edges[PM_MAX_F][3];
    int8_t face_signs[PM_MAX_F][3];

    /* 1-形式接続 (LiftingCellLog) & 2-形式外部フラックス (H² モノポール担体) */
    LiftingCellLog edge_cells[PM_MAX_E];
    lc_logc16_t    face_ext[PM_MAX_F];
} lc_pachner_mesh_t;

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

static inline lc_logc16_t log_inv(lc_logc16_t a) {
    if (a.rho == LC_LOG_ZERO) {
        lc_logc16_t z = { LC_LOG_ZERO, (uint16_t)(0u - a.theta) };
        return z;
    }
    lc_logc16_t out = { (int16_t)(-a.rho), (uint16_t)(0u - a.theta) };
    return out;
}

/* ============================================================================
 * 1. ゼロヒープ固定長プール管理 & 単体メッシュ基本操作
 * ============================================================================ */
static void pm_init(lc_pachner_mesh_t *m) {
    memset(m, 0, sizeof(*m));
    for (int e = 0; e < PM_MAX_E; e++) {
        m->edge_faces[e][0] = -1;
        m->edge_faces[e][1] = -1;
    }
}

static int pm_alloc_vert(lc_pachner_mesh_t *m) {
    for (int v = 0; v < PM_MAX_V; v++) {
        if (!m->vert_alive[v]) {
            if (m->vert_ever[v]) m->reused_v++;
            m->vert_ever[v]  = true;
            m->vert_alive[v] = true;
            m->n_v_alive++;
            return v;
        }
    }
    return -1;
}

static void pm_free_vert(lc_pachner_mesh_t *m, int v) {
    if (v >= 0 && v < PM_MAX_V && m->vert_alive[v]) {
        m->vert_alive[v] = false;
        m->n_v_alive--;
    }
}

static int pm_find_edge(const lc_pachner_mesh_t *m, int a, int b, int8_t *out_sign) {
    int u = (a < b) ? a : b;
    int v = (a < b) ? b : a;
    if (out_sign != NULL) *out_sign = (a < b) ? 1 : -1;
    for (int e = 0; e < PM_MAX_E; e++) {
        if (m->edge_alive[e] && m->edge_u[e] == u && m->edge_v[e] == v) return e;
    }
    return -1;
}

static int pm_alloc_edge(lc_pachner_mesh_t *m, int a, int b,
                         lc_logc16_t w_dir, int16_t wx_dir, int16_t wy_dir) {
    int u = (a < b) ? a : b;
    int v = (a < b) ? b : a;
    int8_t sgn = (a < b) ? 1 : -1;
    for (int e = 0; e < PM_MAX_E; e++) {
        if (!m->edge_alive[e]) {
            if (m->edge_ever[e]) m->reused_e++;
            m->edge_ever[e]     = true;
            m->edge_alive[e]    = true;
            m->edge_u[e]        = u;
            m->edge_v[e]        = v;
            m->edge_faces[e][0] = -1;
            m->edge_faces[e][1] = -1;
            m->edge_f_sign[e][0] = 0;
            m->edge_f_sign[e][1] = 0;
            memset(&m->edge_cells[e], 0, sizeof(LiftingCellLog));
            m->edge_cells[e].flags = LC_FLAG_ALIVE;
            m->edge_cells[e].w     = (sgn > 0) ? w_dir : log_inv(w_dir);
            m->edge_wx[e]          = (int16_t)(sgn * (int)wx_dir);
            m->edge_wy[e]          = (int16_t)(sgn * (int)wy_dir);
            m->n_e_alive++;
            return e;
        }
    }
    return -1;
}

static void pm_free_edge(lc_pachner_mesh_t *m, int e) {
    if (e >= 0 && e < PM_MAX_E && m->edge_alive[e]) {
        m->edge_alive[e]    = false;
        m->edge_faces[e][0] = -1;
        m->edge_faces[e][1] = -1;
        m->edge_f_sign[e][0] = 0;
        m->edge_f_sign[e][1] = 0;
        m->edge_wx[e] = 0;
        m->edge_wy[e] = 0;
        memset(&m->edge_cells[e], 0, sizeof(LiftingCellLog));
        m->n_e_alive--;
    }
}

static int pm_get_or_create_edge(lc_pachner_mesh_t *m, int a, int b, int8_t *out_sign) {
    int e = pm_find_edge(m, a, b, out_sign);
    if (e >= 0) return e;
    if (out_sign != NULL) *out_sign = (a < b) ? 1 : -1;
    return pm_alloc_edge(m, a, b, (lc_logc16_t){0, 0u}, 0, 0);
}

static int pm_alloc_face(lc_pachner_mesh_t *m, int v0, int v1, int v2, lc_logc16_t ext_flux) {
    int fid = -1;
    for (int f = 0; f < PM_MAX_F; f++) {
        if (!m->face_alive[f]) { fid = f; break; }
    }
    if (fid < 0) return -1;
    if (m->face_ever[fid]) m->reused_f++;
    m->face_ever[fid]     = true;
    m->face_alive[fid]    = true;
    m->face_verts[fid][0] = v0;
    m->face_verts[fid][1] = v1;
    m->face_verts[fid][2] = v2;
    m->face_ext[fid]      = ext_flux;

    const int pairs[3][2] = { {v0, v1}, {v1, v2}, {v2, v0} };
    for (int k = 0; k < 3; k++) {
        int8_t sgn = 1;
        int e = pm_get_or_create_edge(m, pairs[k][0], pairs[k][1], &sgn);
        if (e < 0) return -1;
        m->face_edges[fid][k] = e;
        m->face_signs[fid][k] = sgn;
        if (m->edge_faces[e][0] < 0) {
            m->edge_faces[e][0]  = fid;
            m->edge_f_sign[e][0] = sgn;
        } else {
            m->edge_faces[e][1]  = fid;
            m->edge_f_sign[e][1] = sgn;
        }
    }
    m->n_f_alive++;
    return fid;
}

static void pm_detach_face(lc_pachner_mesh_t *m, int fid) {
    if (fid < 0 || fid >= PM_MAX_F || !m->face_alive[fid]) return;
    for (int k = 0; k < 3; k++) {
        int e = m->face_edges[fid][k];
        if (m->edge_faces[e][0] == fid) {
            m->edge_faces[e][0]  = -1;
            m->edge_f_sign[e][0] = 0;
        } else if (m->edge_faces[e][1] == fid) {
            m->edge_faces[e][1]  = -1;
            m->edge_f_sign[e][1] = 0;
        }
    }
    m->face_alive[fid] = false;
    m->face_ext[fid]   = (lc_logc16_t){ 0, 0u };
    m->n_f_alive--;
}

static void pm_get_dir_edge_data(const lc_pachner_mesh_t *m, int a, int b,
                                 lc_logc16_t *out_w, int16_t *out_wx, int16_t *out_wy) {
    int8_t sgn = 1;
    int e = pm_find_edge(m, a, b, &sgn);
    if (out_w != NULL)  *out_w  = (sgn > 0) ? m->edge_cells[e].w : log_inv(m->edge_cells[e].w);
    if (out_wx != NULL) *out_wx = (int16_t)(sgn * (int)m->edge_wx[e]);
    if (out_wy != NULL) *out_wy = (int16_t)(sgn * (int)m->edge_wy[e]);
}

static bool pm_has_face_with_verts(const lc_pachner_mesh_t *m, int a, int b, int c) {
    for (int f = 0; f < PM_MAX_F; f++) {
        if (!m->face_alive[f]) continue;
        int v0 = m->face_verts[f][0], v1 = m->face_verts[f][1], v2 = m->face_verts[f][2];
        bool ha = (v0 == a || v1 == a || v2 == a);
        bool hb = (v0 == b || v1 == b || v2 == b);
        bool hc = (v0 == c || v1 == c || v2 == c);
        if (ha && hb && hc) return true;
    }
    return false;
}

/* ============================================================================
 * 2. 単体位相整合・面曲率・大域 H² モノポール・トーラス調和 H¹ 監査器
 * ============================================================================ */
static lc_2morph_t pm_face_curvature(const lc_pachner_mesh_t *m, int fid) {
    int32_t  dr = (int32_t)m->face_ext[fid].rho;
    uint16_t dt = m->face_ext[fid].theta;
    for (int k = 0; k < 3; k++) {
        int e   = m->face_edges[fid][k];
        int sgn = (int)m->face_signs[fid][k];
        dr += sgn * (int32_t)m->edge_cells[e].w.rho;
        dt  = (uint16_t)(dt + (uint16_t)(sgn * (int32_t)(int16_t)m->edge_cells[e].w.theta));
    }
    lc_2morph_t out = { dr, (int16_t)dt, false };
    return out;
}

static lc_2morph_t pm_global_h2(const lc_pachner_mesh_t *m) {
    int32_t  tot_dr = 0;
    uint16_t tot_dt = 0u;
    for (int f = 0; f < PM_MAX_F; f++) {
        if (!m->face_alive[f]) continue;
        lc_2morph_t cf = pm_face_curvature(m, f);
        tot_dr += cf.drho;
        tot_dt  = (uint16_t)(tot_dt + (uint16_t)cf.dtheta);
    }
    lc_2morph_t out = { tot_dr, (int16_t)tot_dt, false };
    return out;
}

static int pm_count_vortex_faces(const lc_pachner_mesh_t *m, int64_t *out_l1) {
    int cnt = 0;
    int64_t l1 = 0;
    for (int f = 0; f < PM_MAX_F; f++) {
        if (!m->face_alive[f]) continue;
        lc_2morph_t cf = pm_face_curvature(m, f);
        if (!lc_2morph_is_flat(cf)) {
            cnt++;
            l1 += (int64_t)lc_2morph_l1(cf);
        }
    }
    if (out_l1 != NULL) *out_l1 = l1;
    return cnt;
}

/* 閉じた向き付け可能単体多様体条件 + 全生存面上のシーム巻き数閉路和ゼロ検証 */
static bool pm_verify_manifold(const lc_pachner_mesh_t *m, int expected_chi) {
    int cv = 0, ce = 0, cf = 0;
    for (int v = 0; v < PM_MAX_V; v++) if (m->vert_alive[v]) cv++;
    for (int e = 0; e < PM_MAX_E; e++) {
        if (!m->edge_alive[e]) continue;
        ce++;
        int f0 = m->edge_faces[e][0], f1 = m->edge_faces[e][1];
        if (f0 < 0 || f1 < 0 || f0 == f1) return false;
        if (!m->face_alive[f0] || !m->face_alive[f1]) return false;
        if ((int)m->edge_f_sign[e][0] + (int)m->edge_f_sign[e][1] != 0) return false;
        if (!m->vert_alive[m->edge_u[e]] || !m->vert_alive[m->edge_v[e]]) return false;
    }
    for (int f = 0; f < PM_MAX_F; f++) {
        if (!m->face_alive[f]) continue;
        cf++;
        int swx = 0, swy = 0;
        for (int k = 0; k < 3; k++) {
            int e = m->face_edges[f][k];
            if (!m->edge_alive[e]) return false;
            swx += (int)m->face_signs[f][k] * (int)m->edge_wx[e];
            swy += (int)m->face_signs[f][k] * (int)m->edge_wy[e];
        }
        if (swx != 0 || swy != 0) return false;
    }
    if (cv != m->n_v_alive || ce != m->n_e_alive || cf != m->n_f_alive) return false;
    return (cv - ce + cf) == expected_chi;
}

/* 不規則変形後のトーラス T² 上で BFS 全域木を張り、全基本閉路 (E-V+1 本) の調和 H¹ 整合を 0-bit 検証 */
static bool pm_verify_torus_harmonic_h1(const lc_pachner_mesh_t *m,
                                        lc_logc16_t exp_wx, lc_logc16_t exp_wy,
                                        int *out_checked_cycles) {
    bool vis[PM_MAX_V] = { false };
    int32_t  pot_wx[PM_MAX_V]  = { 0 };
    int32_t  pot_wy[PM_MAX_V]  = { 0 };
    int32_t  pot_rho[PM_MAX_V] = { 0 };
    uint16_t pot_th[PM_MAX_V]  = { 0 };
    bool is_tree_e[PM_MAX_E]   = { false };

    int root = -1;
    for (int v = 0; v < PM_MAX_V; v++) if (m->vert_alive[v]) { root = v; break; }
    if (root < 0) return false;

    int q[PM_MAX_V], qh = 0, qt = 0;
    vis[root] = true;
    q[qt++] = root;

    while (qh < qt) {
        int cur = q[qh++];
        for (int e = 0; e < PM_MAX_E; e++) {
            if (!m->edge_alive[e]) continue;
            int nb = -1, sgn = 0;
            if (m->edge_u[e] == cur)      { nb = m->edge_v[e]; sgn =  1; }
            else if (m->edge_v[e] == cur) { nb = m->edge_u[e]; sgn = -1; }
            if (nb < 0 || vis[nb]) continue;
            vis[nb]      = true;
            is_tree_e[e] = true;
            pot_wx[nb]   = pot_wx[cur]  + sgn * (int32_t)m->edge_wx[e];
            pot_wy[nb]   = pot_wy[cur]  + sgn * (int32_t)m->edge_wy[e];
            pot_rho[nb]  = pot_rho[cur] + sgn * (int32_t)m->edge_cells[e].w.rho;
            pot_th[nb]   = (uint16_t)(pot_th[cur] + (uint16_t)(sgn * (int32_t)(int16_t)m->edge_cells[e].w.theta));
            q[qt++] = nb;
        }
    }
    if (qt != m->n_v_alive) return false;

    int checked = 0;
    bool saw_x = false, saw_y = false;
    for (int e = 0; e < PM_MAX_E; e++) {
        if (!m->edge_alive[e] || is_tree_e[e]) continue;
        int u = m->edge_u[e], v = m->edge_v[e];
        int32_t  cx = pot_wx[u] + (int32_t)m->edge_wx[e] - pot_wx[v];
        int32_t  cy = pot_wy[u] + (int32_t)m->edge_wy[e] - pot_wy[v];
        int32_t  cr = pot_rho[u] + (int32_t)m->edge_cells[e].w.rho - pot_rho[v];
        uint16_t ct = (uint16_t)(pot_th[u] + m->edge_cells[e].w.theta - pot_th[v]);

        int32_t  er = cx * (int32_t)exp_wx.rho + cy * (int32_t)exp_wy.rho;
        uint16_t et = (uint16_t)((uint32_t)(cx * (int32_t)exp_wx.theta + cy * (int32_t)exp_wy.theta));
        if (cr != er || ct != et) return false;
        if (cx != 0) saw_x = true;
        if (cy != 0) saw_y = true;
        checked++;
    }
    if (out_checked_cycles != NULL) *out_checked_cycles = checked;
    return saw_x && saw_y;
}

/* ============================================================================
 * 3. 2 次元パッフナー移動の 3 基底 (1-3 細分化 / 3-1 縮約 / 2-2 対角フリップ)
 * ============================================================================ */

/* 1-3 移動 (Stellar Subdivision): ΔV=+1, ΔE=+3, ΔF=+2, Δχ=0, Ω_{H²} 0-bit 保存 */
static bool pm_pachner_1_3(lc_pachner_mesh_t *m, int fid, int *out_new_v) {
    if (fid < 0 || fid >= PM_MAX_F || !m->face_alive[fid]) return false;
    if (m->n_v_alive + 1 > PM_MAX_V || m->n_e_alive + 3 > PM_MAX_E || m->n_f_alive + 2 > PM_MAX_F) {
        return false;
    }

    int a = m->face_verts[fid][0], b = m->face_verts[fid][1], c = m->face_verts[fid][2];
    lc_logc16_t phi = m->face_ext[fid];

    lc_logc16_t wab, wca;
    int16_t wab_x, wab_y, wca_x, wca_y;
    pm_get_dir_edge_data(m, a, b, &wab, &wab_x, &wab_y);
    pm_get_dir_edge_data(m, c, a, &wca, &wca_x, &wca_y);

    /* 親面を切り離し、重心頂点 vc と内部 3 辺を挿入 (平坦接続 & 巻き数閉路和を 0-bit 厳密継承) */
    pm_detach_face(m, fid);
    int vc = pm_alloc_vert(m);
    if (vc < 0) return false;

    if (pm_alloc_edge(m, a,  vc, (lc_logc16_t){0, 0u}, 0, 0) < 0 ||
        pm_alloc_edge(m, vc, b,  wab,                  wab_x, wab_y) < 0 ||
        pm_alloc_edge(m, vc, c,  log_inv(wca),         (int16_t)(-wca_x), (int16_t)(-wca_y)) < 0) {
        return false;
    }

    /* 親フラックス Φ を 3 子面へ 0-bit 厳密整数分配 (ΣΦ_i == Φ) */
    int16_t  r0 = (int16_t)(phi.rho / 3), r1 = r0, r2 = (int16_t)(phi.rho - r0 - r1);
    uint16_t t0 = (uint16_t)(phi.theta / 3u), t1 = t0, t2 = (uint16_t)(phi.theta - t0 - t1);

    if (pm_alloc_face(m, a, b, vc, (lc_logc16_t){r0, t0}) < 0 ||
        pm_alloc_face(m, b, c, vc, (lc_logc16_t){r1, t1}) < 0 ||
        pm_alloc_face(m, c, a, vc, (lc_logc16_t){r2, t2}) < 0) {
        return false;
    }
    if (out_new_v != NULL) *out_new_v = vc;
    return true;
}

/* 3-1 移動 (Inverse Subdivision / Contraction): ΔV=-1, ΔE=-3, ΔF=-2, Δχ=0, Ω_{H²} 0-bit 保存 */
static bool pm_pachner_3_1(lc_pachner_mesh_t *m, int vid, int *out_new_f) {
    if (vid < 0 || vid >= PM_MAX_V || !m->vert_alive[vid]) return false;

    /* 1. 次数 3 検証: vid に接続する生存辺がちょうど 3 本、生存面がちょうど 3 枚 */
    int inc_e[4], ne = 0;
    for (int e = 0; e < PM_MAX_E; e++) {
        if (!m->edge_alive[e]) continue;
        if (m->edge_u[e] == vid || m->edge_v[e] == vid) {
            if (ne >= 3) return false;
            inc_e[ne++] = e;
        }
    }
    if (ne != 3) return false;

    int inc_f[4], nf = 0;
    for (int k = 0; k < 3; k++) {
        for (int s = 0; s < 2; s++) {
            int f = m->edge_faces[inc_e[k]][s];
            if (f < 0 || !m->face_alive[f]) return false;
            bool dup = false;
            for (int j = 0; j < nf; j++) if (inc_f[j] == f) dup = true;
            if (!dup) {
                if (nf >= 3) return false;
                inc_f[nf++] = f;
            }
        }
    }
    if (nf != 3) return false;

    /* 2. 3 面の外周有向辺 (p -> q) を抽出し、向き付き閉路 (a -> b -> c -> a) を復元 */
    int pe[3], qe[3];
    int32_t  sum_r = 0;
    uint16_t sum_t = 0u;
    for (int i = 0; i < 3; i++) {
        int f = inc_f[i], pos = -1;
        for (int k = 0; k < 3; k++) if (m->face_verts[f][k] == vid) pos = k;
        if (pos < 0) return false;
        pe[i] = m->face_verts[f][(pos + 1) % 3];
        qe[i] = m->face_verts[f][(pos + 2) % 3];
        sum_r += (int32_t)m->face_ext[f].rho;
        sum_t  = (uint16_t)(sum_t + m->face_ext[f].theta);
    }
    if (sum_r > 16000 || sum_r < -16000) return false;

    int a = pe[0], b = qe[0], c = -1;
    if (pe[1] == b && qe[1] == pe[2] && qe[2] == a)      c = qe[1];
    else if (pe[2] == b && qe[2] == pe[1] && qe[1] == a) c = qe[2];
    else return false;

    if (a == b || b == c || c == a) return false;

    /* 3. 非退化ガード: 外周 (a,b,c) がすでに別面を成している場合 (最小正四面体崩壊) は拒否 */
    if (pm_has_face_with_verts(m, a, b, c)) return false;

    /* 4. 3 子面・内部 3 辺・中心頂点をフリーリストへ返却し、1 親面 (a,b,c) へ統合 */
    for (int i = 0; i < 3; i++) pm_detach_face(m, inc_f[i]);
    for (int k = 0; k < 3; k++) pm_free_edge(m, inc_e[k]);
    pm_free_vert(m, vid);

    int nf_id = pm_alloc_face(m, a, b, c, (lc_logc16_t){(int16_t)sum_r, sum_t});
    if (nf_id < 0) return false;
    if (out_new_f != NULL) *out_new_f = nf_id;
    return true;
}

/* 2-2 移動 (Edge Flip / Associator 結合組換): ΔV=0, ΔE=0, ΔF=0, 渦均等化 & 0-bit 保存 */
static bool pm_pachner_2_2(lc_pachner_mesh_t *m, int eid, int *out_new_e) {
    if (eid < 0 || eid >= PM_MAX_E || !m->edge_alive[eid]) return false;
    int f0 = m->edge_faces[eid][0], f1 = m->edge_faces[eid][1];
    if (f0 < 0 || f1 < 0 || f0 == f1) return false;
    if (!m->face_alive[f0] || !m->face_alive[f1]) return false;

    /* 標準向き a -> c で接する面を fL=(a,c,d)、逆向き c -> a で接する面を fR=(a,b,c) へ同定 */
    int a = m->edge_u[eid], c = m->edge_v[eid];
    int fL = (m->edge_f_sign[eid][0] > 0) ? f0 : f1;
    int fR = (m->edge_f_sign[eid][0] > 0) ? f1 : f0;

    int d = -1, b = -1;
    for (int k = 0; k < 3; k++) {
        int vL = m->face_verts[fL][k];
        if (vL != a && vL != c) d = vL;
        int vR = m->face_verts[fR][k];
        if (vR != a && vR != c) b = vR;
    }
    if (b < 0 || d < 0 || b == d || b == a || b == c || d == a || d == c) return false;

    /* リンク条件: 新対角辺 (b, d) が既存でないこと (二重辺・非単体複体化の完全防止) */
    if (pm_find_edge(m, b, d, NULL) >= 0) return false;

    /* 外周 4 辺の接続・巻き数、および 2 面の外部フラックスを採取 */
    lc_logc16_t wab, wbc, wcd, wda;
    int16_t xbc, ybc, xcd, ycd;
    pm_get_dir_edge_data(m, a, b, &wab, NULL, NULL);
    pm_get_dir_edge_data(m, b, c, &wbc, &xbc, &ybc);
    pm_get_dir_edge_data(m, c, d, &wcd, &xcd, &ycd);
    pm_get_dir_edge_data(m, d, a, &wda, NULL, NULL);

    int32_t  sum_er = (int32_t)m->face_ext[fR].rho + (int32_t)m->face_ext[fL].rho;
    uint16_t sum_et = (uint16_t)(m->face_ext[fR].theta + m->face_ext[fL].theta);
    if (sum_er > 16000 || sum_er < -16000) return false;

    int16_t  e1_r = (int16_t)(sum_er / 2), e2_r = (int16_t)(sum_er - e1_r);
    uint16_t e1_t = (uint16_t)(sum_et / 2u), e2_t = (uint16_t)(sum_et - e1_t);

    /* 新対角辺 w(b->d) = ω_half - A (ブランチ曖昧さゼロ・平坦保存・渦 50/50 均等分散) */
    int32_t  A_r = (int32_t)wab.rho + (int32_t)wda.rho + (int32_t)e1_r;
    uint16_t A_t = (uint16_t)(wab.theta + wda.theta + e1_t);
    int32_t  B_r = (int32_t)wbc.rho + (int32_t)wcd.rho + (int32_t)e2_r;
    uint16_t B_t = (uint16_t)(wbc.theta + wcd.theta + e2_t);

    int32_t  tot_r  = A_r + B_r;
    int16_t  tot_t  = (int16_t)((uint16_t)(A_t + B_t));
    int32_t  half_r = tot_r / 2;
    int16_t  half_t = (int16_t)(tot_t / 2);

    int32_t  new_r  = half_r - A_r;
    uint16_t new_t  = (uint16_t)((uint16_t)half_t - A_t);
    if (new_r > 16000 || new_r < -16000) return false;

    int16_t new_wx = (int16_t)(xbc + xcd);
    int16_t new_wy = (int16_t)(ybc + ycd);

    /* 旧 2 面・旧共有辺を解放し、新対角辺 (b,d) と新 2 面 (a,b,d), (b,c,d) を接続 */
    pm_detach_face(m, fR);
    pm_detach_face(m, fL);
    pm_free_edge(m, eid);

    int ne_id = pm_alloc_edge(m, b, d, (lc_logc16_t){(int16_t)new_r, new_t}, new_wx, new_wy);
    if (ne_id < 0) return false;
    if (pm_alloc_face(m, a, b, d, (lc_logc16_t){e1_r, e1_t}) < 0 ||
        pm_alloc_face(m, b, c, d, (lc_logc16_t){e2_r, e2_t}) < 0) {
        return false;
    }
    if (out_new_e != NULL) *out_new_e = ne_id;
    return true;
}

/* ============================================================================
 * 4. 標準閉多様体ビルダー (最小正四面体 / 正二十面体 S² / 4x4 トーラス T²)
 * ============================================================================ */
static void pm_build_tetrahedron(lc_pachner_mesh_t *m) {
    pm_init(m);
    for (int v = 0; v < 4; v++) (void)pm_alloc_vert(m);
    (void)pm_alloc_face(m, 0, 2, 1, (lc_logc16_t){0, 0u});
    (void)pm_alloc_face(m, 0, 1, 3, (lc_logc16_t){0, 0u});
    (void)pm_alloc_face(m, 1, 2, 3, (lc_logc16_t){0, 0u});
    (void)pm_alloc_face(m, 2, 0, 3, (lc_logc16_t){0, 0u});
}

static void pm_build_icosahedron(lc_pachner_mesh_t *m) {
    pm_init(m);
    for (int v = 0; v < 12; v++) (void)pm_alloc_vert(m);
    for (int i = 0; i < 5; i++) {
        int u0 = 1 + i, u1 = 1 + ((i + 1) % 5);
        int l0 = 6 + i, l1 = 6 + ((i + 1) % 5);
        (void)pm_alloc_face(m, 0,  u0, u1, (lc_logc16_t){0, 0u});
        (void)pm_alloc_face(m, u0, l0, u1, (lc_logc16_t){0, 0u});
        (void)pm_alloc_face(m, u1, l0, l1, (lc_logc16_t){0, 0u});
        (void)pm_alloc_face(m, 11, l1, l0, (lc_logc16_t){0, 0u});
    }
}

static void pm_build_torus4x4(lc_pachner_mesh_t *m, lc_logc16_t harm_wx, lc_logc16_t harm_wy) {
    pm_init(m);
    for (int v = 0; v < 16; v++) (void)pm_alloc_vert(m);
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int v00 = y * 4 + x;
            int v10 = y * 4 + ((x + 1) % 4);
            int v01 = ((y + 1) % 4) * 4 + x;
            int v11 = ((y + 1) % 4) * 4 + ((x + 1) % 4);
            (void)pm_alloc_face(m, v00, v10, v11, (lc_logc16_t){0, 0u});
            (void)pm_alloc_face(m, v00, v11, v01, (lc_logc16_t){0, 0u});
        }
    }
    /* シーム交差巻き数 (edge_wx, edge_wy) と純粋調和 H¹ 1-形式を全 48 辺へ正確に設定 */
    uint16_t stx = (uint16_t)(harm_wx.theta / 4u);
    uint16_t sty = (uint16_t)(harm_wy.theta / 4u);
    int16_t  srx = (int16_t)(harm_wx.rho / 4);
    int16_t  sry = (int16_t)(harm_wy.rho / 4);

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int v00 = y * 4 + x;
            int v10 = y * 4 + ((x + 1) % 4);
            int v01 = ((y + 1) % 4) * 4 + x;
            int v11 = ((y + 1) % 4) * 4 + ((x + 1) % 4);
            int8_t sh, sv, sd;
            int eh = pm_find_edge(m, v00, v10, &sh);
            int ev = pm_find_edge(m, v00, v01, &sv);
            int ed = pm_find_edge(m, v00, v11, &sd);

            m->edge_wx[eh] = (int16_t)(sh * ((x == 3) ? 1 : 0));
            m->edge_wy[eh] = 0;
            m->edge_cells[eh].w.rho   = (int16_t)(sh * srx);
            m->edge_cells[eh].w.theta = (uint16_t)(sh * (int32_t)stx);

            m->edge_wx[ev] = 0;
            m->edge_wy[ev] = (int16_t)(sv * ((y == 3) ? 1 : 0));
            m->edge_cells[ev].w.rho   = (int16_t)(sv * sry);
            m->edge_cells[ev].w.theta = (uint16_t)(sv * (int32_t)sty);

            m->edge_wx[ed] = (int16_t)(sd * ((x == 3) ? 1 : 0));
            m->edge_wy[ed] = (int16_t)(sd * ((y == 3) ? 1 : 0));
            m->edge_cells[ed].w.rho   = (int16_t)(sd * (srx + sry));
            m->edge_cells[ed].w.theta = (uint16_t)(sd * (int32_t)(uint16_t)(stx + sty));
        }
    }
}

/* ============================================================================
 * 5. [Test 1..4] パッフナー移動の 0-bit 位相保存・ガード・100 連ストレステスト
 * ============================================================================ */
static bool run_random_pachner_stress(lc_pachner_mesh_t *m, int expected_chi,
                                      lc_2morph_t expected_h2, int n_moves,
                                      uint32_t *rng,
                                      int *cnt_13, int *cnt_31, int *cnt_22) {
    *cnt_13 = *cnt_31 = *cnt_22 = 0;
    int done = 0, guard = 0;

    while (done < n_moves && guard < n_moves * 20) {
        guard++;
        uint32_t r = next_rng(rng);
        int choice = (int)(r % 10u);

        /* 適度に膨張と収縮がバランスする確率配分: 2-2 (50%), 1-3 (25%), 3-1 (25%) */
        if (choice < 5) {
            /* 2-2 対角フリップ */
            int start = (int)((r >> 8) % PM_MAX_E);
            for (int off = 0; off < PM_MAX_E; off++) {
                int e = (start + off) % PM_MAX_E;
                if (m->edge_alive[e] && pm_pachner_2_2(m, e, NULL)) {
                    (*cnt_22)++; done++; break;
                }
            }
        } else if (choice < 8 && m->n_f_alive < 120) {
            /* 1-3 細分化 */
            int start = (int)((r >> 8) % PM_MAX_F);
            for (int off = 0; off < PM_MAX_F; off++) {
                int f = (start + off) % PM_MAX_F;
                if (m->face_alive[f] && pm_pachner_1_3(m, f, NULL)) {
                    (*cnt_13)++; done++; break;
                }
            }
        } else {
            /* 3-1 縮約 (次数 3 頂点が存在する場合のみ発火) */
            int start = (int)((r >> 8) % PM_MAX_V);
            for (int off = 0; off < PM_MAX_V; off++) {
                int v = (start + off) % PM_MAX_V;
                if (m->vert_alive[v] && pm_pachner_3_1(m, v, NULL)) {
                    (*cnt_31)++; done++; break;
                }
            }
        }

        if (!pm_verify_manifold(m, expected_chi)) return false;
        lc_2morph_t cur_h2 = pm_global_h2(m);
        if (cur_h2.drho != expected_h2.drho || cur_h2.dtheta != expected_h2.dtheta) return false;
    }
    return done == n_moves;
}

static bool run_test1_basic_moves_and_guards(void) {
    printf("[Test 1] 2 次元パッフナー移動 3 基底の厳密性・渦均等化・ガード検証:\n");

    /* 1-A: 1-3 細分化 -> 内部 3 辺へ非ゼロゲージ注入 -> 3-1 縮約で 0-bit 完全復帰 */
    lc_pachner_mesh_t ico;
    pm_build_icosahedron(&ico);
    ico.face_ext[0] = (lc_logc16_t){ -256, 0x4000u };
    lc_2morph_t h2_pre = pm_global_h2(&ico);
    lc_2morph_t c0_pre = pm_face_curvature(&ico, 0);

    int vc = -1, nf = -1;
    if (!pm_pachner_1_3(&ico, 0, &vc) || !pm_verify_manifold(&ico, 2) ||
        ico.n_v_alive != 13 || ico.n_e_alive != 33 || ico.n_f_alive != 22) {
        fprintf(stderr, "FAIL: Test 1-A 1-3 細分化失敗\n");
        return false;
    }
    /* 新設された内部 3 辺へ非ゼロの局所ゲージを載せても 3-1 縮約で厳密相殺されることを検証 */
    for (int e = 0; e < PM_MAX_E; e++) {
        if (ico.edge_alive[e] && (ico.edge_u[e] == vc || ico.edge_v[e] == vc)) {
            ico.edge_cells[e].w = (lc_logc16_t){ 177, 0x2B73u };
        }
    }
    lc_2morph_t h2_mid = pm_global_h2(&ico);
    if (!pm_pachner_3_1(&ico, vc, &nf) || !pm_verify_manifold(&ico, 2) ||
        ico.n_v_alive != 12 || ico.n_e_alive != 30 || ico.n_f_alive != 20) {
        fprintf(stderr, "FAIL: Test 1-A 3-1 縮約失敗\n");
        return false;
    }
    lc_2morph_t h2_post = pm_global_h2(&ico);
    lc_2morph_t c0_post = pm_face_curvature(&ico, nf);
    if (h2_mid.drho != h2_pre.drho || h2_mid.dtheta != h2_pre.dtheta ||
        h2_post.drho != h2_pre.drho || h2_post.dtheta != h2_pre.dtheta ||
        c0_post.drho != c0_pre.drho || c0_post.dtheta != c0_pre.dtheta) {
        fprintf(stderr, "FAIL: Test 1-A 3-1 ∘ 1-3 ホロノミー不変違反\n");
        return false;
    }
    printf("  1-A 1-3 細分化 -> 内部辺ゲージ注入 -> 3-1 縮約: (12,30,20)->(13,33,22)->(12,30,20), Ω_H2=(-256,0x4000) 0-bit 一致 OK\n");

    /* 1-B: 2-2 対角フリップによる偏在渦の 50/50 均等分散 & Ω_H2 保存 */
    pm_build_icosahedron(&ico);
    int e_shared = ico.face_edges[0][1];
    int f_a = ico.edge_faces[e_shared][0], f_b = ico.edge_faces[e_shared][1];
    ico.face_ext[f_a] = (lc_logc16_t){ -256, 0x4000u };
    ico.face_ext[f_b] = (lc_logc16_t){    0, 0x0000u };

    int ne = -1;
    if (!pm_pachner_2_2(&ico, e_shared, &ne) || !pm_verify_manifold(&ico, 2)) {
        fprintf(stderr, "FAIL: Test 1-B 2-2 フリップ失敗\n");
        return false;
    }
    lc_2morph_t cf0 = pm_face_curvature(&ico, ico.edge_faces[ne][0]);
    lc_2morph_t cf1 = pm_face_curvature(&ico, ico.edge_faces[ne][1]);
    lc_2morph_t h2_f = pm_global_h2(&ico);
    if (cf0.drho != -128 || cf0.dtheta != 0x2000 || cf1.drho != -128 || cf1.dtheta != 0x2000 ||
        h2_f.drho != -256 || h2_f.dtheta != 0x4000) {
        fprintf(stderr, "FAIL: Test 1-B 2-2 渦均等化検証失敗\n");
        return false;
    }
    printf("  1-B 2-2 対角フリップ渦分散: (-256,0x4000)+(0,0) -> (%+d,0x%04X)+(%+d,0x%04X) [50/50 分散 & Ω_H2 保存 OK]\n",
           (int)cf0.drho, (unsigned)(uint16_t)cf0.dtheta, (int)cf1.drho, (unsigned)(uint16_t)cf1.dtheta);

    /* 1-C: 最小正四面体 (V=4, E=6, F=4) 上で全 2-2 (リンク条件) & 全 3-1 (非退化条件) を安全拒否 */
    lc_pachner_mesh_t tet;
    pm_build_tetrahedron(&tet);
    int rej_22 = 0, rej_31 = 0;
    for (int e = 0; e < PM_MAX_E; e++) {
        if (tet.edge_alive[e] && !pm_pachner_2_2(&tet, e, NULL)) rej_22++;
    }
    for (int v = 0; v < PM_MAX_V; v++) {
        if (tet.vert_alive[v] && !pm_pachner_3_1(&tet, v, NULL)) rej_31++;
    }
    if (rej_22 != 6 || rej_31 != 4 || !pm_verify_manifold(&tet, 2)) {
        fprintf(stderr, "FAIL: Test 1-C ガード検証失敗 (rej22=%d, rej31=%d)\n", rej_22, rej_31);
        return false;
    }
    printf("  1-C 最小正四面体ガード検証: リンク条件 2-2 拒否=%d/6, 非退化 3-1 拒否=%d/4 (メッシュ無傷 OK)\n\n",
           rej_22, rej_31);
    return true;
}

static bool run_test2_adaptive_metabolism_and_reuse(void) {
    lc_pachner_mesh_t ico;
    pm_build_icosahedron(&ico);
    ico.face_ext[0] = (lc_logc16_t){ -300, 0x6000u }; /* 高曲率 H² 集中面 */

    /* 1. 適応 1-3 細分化: 最大曲率面を 3 連続細分化 */
    int spawned_vs[3];
    for (int step = 0; step < 3; step++) {
        int hot_f = -1;
        int32_t max_l1 = -1;
        for (int f = 0; f < PM_MAX_F; f++) {
            if (!ico.face_alive[f]) continue;
            int32_t l1 = lc_2morph_l1(pm_face_curvature(&ico, f));
            if (l1 > max_l1) { max_l1 = l1; hot_f = f; }
        }
        if (hot_f < 0 || !pm_pachner_1_3(&ico, hot_f, &spawned_vs[step])) return false;
    }

    /* 2. ヒステリシス代謝 3-1 縮約: 新設次数 3 頂点を逆順に 2 個回収 */
    if (!pm_pachner_3_1(&ico, spawned_vs[2], NULL) ||
        !pm_pachner_3_1(&ico, spawned_vs[1], NULL)) return false;

    /* 3. 別面 (面 10) を 1-3 細分化 -> 解放された頂点・辺・面スロットがゼロ確保再利用される */
    int rv = -1;
    if (!pm_pachner_1_3(&ico, 10, &rv) || !pm_verify_manifold(&ico, 2)) return false;
    lc_2morph_t h2 = pm_global_h2(&ico);

    if (ico.reused_v < 1 || ico.reused_e < 3 || ico.reused_f < 2 ||
        h2.drho != -300 || h2.dtheta != 0x6000) {
        fprintf(stderr, "FAIL: Test 2 フリーリスト再利用または Ω_H2 違反\n");
        return false;
    }
    printf("[Test 2] 適応 1-3 細分化 -> 3-1 代謝縮約 & ゼロヒープ固定長フリーリスト再利用:\n");
    printf("  再利用スロット数: 頂点=%d, 辺=%d, 面=%d (動的確保 0 B) | 最終 (V=%d,E=%d,F=%d,χ=2) | Ω_H2=(-300,0x6000) OK\n\n",
           ico.reused_v, ico.reused_e, ico.reused_f, ico.n_v_alive, ico.n_e_alive, ico.n_f_alive);
    return true;
}

static bool run_test3_s2_100moves(lc_pachner_mesh_t *out_deformed_s2) {
    pm_build_icosahedron(out_deformed_s2);
    uint32_t rng = 0x20261123u;

    /* ランダムな局所 H¹ 渦 (30 辺) + 大域 H² モノポール (-384, 0x4000) を注入 */
    for (int e = 0; e < PM_MAX_E; e++) {
        if (!out_deformed_s2->edge_alive[e]) continue;
        uint32_t r = next_rng(&rng);
        out_deformed_s2->edge_cells[e].w = (lc_logc16_t){
            (int16_t)((int32_t)(r & 0x1FFu) - 256), (uint16_t)(r >> 16)
        };
    }
    out_deformed_s2->face_ext[0] = (lc_logc16_t){ -384, 0x4000u };
    lc_2morph_t h2_init = pm_global_h2(out_deformed_s2);

    int c13 = 0, c31 = 0, c22 = 0;
    if (!run_random_pachner_stress(out_deformed_s2, 2, h2_init, 100, &rng, &c13, &c31, &c22)) {
        fprintf(stderr, "FAIL: Test 3 S² 100 連パッフナー移動失敗\n");
        return false;
    }
    printf("[Test 3] 正二十面体 S² 上の 100 連ランダムパッフナー移動 (1-3=%d, 3-1=%d, 2-2=%d):\n", c13, c31, c22);
    printf("  変形後 (V=%d, E=%d, F=%d, χ=2 厳密不変) | 再利用=(V:%d,E:%d,F:%d) | 大域 Ω_H2=(%+d,0x%04X) 0-bit 保存 OK\n\n",
           out_deformed_s2->n_v_alive, out_deformed_s2->n_e_alive, out_deformed_s2->n_f_alive,
           out_deformed_s2->reused_v, out_deformed_s2->reused_e, out_deformed_s2->reused_f,
           (int)h2_init.drho, (unsigned)(uint16_t)h2_init.dtheta);
    return true;
}

static bool run_test4_t2_100moves_harmonic_and_h2(lc_pachner_mesh_t *out_deformed_t2) {
    printf("[Test 4] トーラス T² (χ=0, b1=2, b2=1) 上の 100 連パッフナー移動 0-bit 保存検証:\n");
    uint32_t rng = 0x20261124u;

    /* 4-A: 純粋調和 H¹ 巻き数 (W_x = 0x8000, W_y = 0x2B74, 全 32 面平坦) 上での 100 連移動 */
    const lc_logc16_t wx = { 0, 0x8000u };
    const lc_logc16_t wy = { 0, 0x2B74u };
    pm_build_torus4x4(out_deformed_t2, wx, wy);

    if (!pm_verify_manifold(out_deformed_t2, 0) || pm_count_vortex_faces(out_deformed_t2, NULL) != 0) {
        fprintf(stderr, "FAIL: Test 4-A 初期調和トーラス検証失敗\n");
        return false;
    }

    int c13 = 0, c31 = 0, c22 = 0;
    if (!run_random_pachner_stress(out_deformed_t2, 0, (lc_2morph_t){0, 0, false}, 100, &rng, &c13, &c31, &c22)) {
        fprintf(stderr, "FAIL: Test 4-A T² 100 連移動失敗\n");
        return false;
    }
    int rem_vort = pm_count_vortex_faces(out_deformed_t2, NULL);
    int n_cycles = 0;
    bool h1_ok   = pm_verify_torus_harmonic_h1(out_deformed_t2, wx, wy, &n_cycles);
    if (rem_vort != 0 || !h1_ok) {
        fprintf(stderr, "FAIL: Test 4-A 調和 H¹ 巻き数保存違反 (vort=%d, h1_ok=%d)\n", rem_vort, h1_ok ? 1 : 0);
        return false;
    }
    printf("  4-A 純粋調和 T² 100 連移動 (1-3=%d,3-1=%d,2-2=%d -> V=%d,E=%d,F=%d,χ=0):\n",
           c13, c31, c22, out_deformed_t2->n_v_alive, out_deformed_t2->n_e_alive, out_deformed_t2->n_f_alive);
    printf("      全 %d 面 0-bit 平坦維持 & 全 %d 本の基本閉路で W=(0x%04X, 0x%04X) 0-bit 不変 OK\n",
           out_deformed_t2->n_f_alive, n_cycles, (unsigned)wx.theta, (unsigned)wy.theta);

    /* 4-B: 大域 H² モノポール (-256, 0x4000) + 調和 H¹ 重畳 T² 上での 100 連移動 */
    lc_pachner_mesh_t tor_h2;
    pm_build_torus4x4(&tor_h2, wx, wy);
    tor_h2.face_ext[0]  = (lc_logc16_t){ -192, 0x3000u };
    tor_h2.face_ext[17] = (lc_logc16_t){  -64, 0x1000u };
    lc_2morph_t h2_exp  = pm_global_h2(&tor_h2);

    int d13 = 0, d31 = 0, d22 = 0;
    if (!run_random_pachner_stress(&tor_h2, 0, h2_exp, 100, &rng, &d13, &d31, &d22)) {
        fprintf(stderr, "FAIL: Test 4-B T² H² モノポール 100 連移動失敗\n");
        return false;
    }
    printf("  4-B モノポール重畳 T² 100 連移動 (V=%d,E=%d,F=%d,χ=0): 大域 Ω_H2=(%+d,0x%04X) 0-bit 保存 OK\n\n",
           tor_h2.n_v_alive, tor_h2.n_e_alive, tor_h2.n_f_alive,
           (int)h2_exp.drho, (unsigned)(uint16_t)h2_exp.dtheta);
    return true;
}

/* ============================================================================
 * 6. [Test 5] 100 連パッフナー変形後の不規則 S² / T² メッシュ上の 3 ストリーム可逆輸送
 * ============================================================================ */
typedef struct { lc_c16_t r, g, b; } lc_rgb_t;

static inline void pm_face_rgb_step(const lc_pachner_mesh_t *m, int fid, lc_rgb_t *st, bool inverse) {
    lc_logc16_t w[3];
    for (int k = 0; k < 3; k++) {
        w[k] = m->edge_cells[m->face_edges[fid][k]].w;
        if (m->face_signs[fid][k] < 0) w[k] = log_inv(w[k]);
        /* 輸送用ゲインは安定のため -384 減衰 (θ はメッシュの位相接続を 100% そのまま使用) */
        w[k].rho = (int16_t)((w[k].rho / 4) - 384);
    }
    if (!inverse) {
        lc_logc16_t p0 = lc_fhrr_bind(w[0], lc_cart_to_log(st->g.re, st->g.im));
        st->r = lc_c16_sub_mod(st->r, lc_log_to_cart(p0.rho, p0.theta));
        lc_logc16_t p1 = lc_fhrr_bind(w[1], lc_cart_to_log(st->b.re, st->b.im));
        st->g = lc_c16_sub_mod(st->g, lc_log_to_cart(p1.rho, p1.theta));
        lc_logc16_t p2 = lc_fhrr_bind(w[2], lc_cart_to_log(st->r.re, st->r.im));
        st->b = lc_c16_sub_mod(st->b, lc_log_to_cart(p2.rho, p2.theta));
    } else {
        lc_logc16_t p2 = lc_fhrr_bind(w[2], lc_cart_to_log(st->r.re, st->r.im));
        st->b = lc_c16_add_mod(st->b, lc_log_to_cart(p2.rho, p2.theta));
        lc_logc16_t p1 = lc_fhrr_bind(w[1], lc_cart_to_log(st->b.re, st->b.im));
        st->g = lc_c16_add_mod(st->g, lc_log_to_cart(p1.rho, p1.theta));
        lc_logc16_t p0 = lc_fhrr_bind(w[0], lc_cart_to_log(st->g.re, st->g.im));
        st->r = lc_c16_add_mod(st->r, lc_log_to_cart(p0.rho, p0.theta));
    }
}

static bool verify_mesh_rgb_transport(const lc_pachner_mesh_t *m, const char *label) {
    enum { N_SAMP = 256 };
    static lc_rgb_t s0[N_SAMP], st[N_SAMP];
    int alive_faces[PM_MAX_F], nf = 0;
    for (int f = 0; f < PM_MAX_F; f++) if (m->face_alive[f]) alive_faces[nf++] = f;

    uint32_t rng = 0x20261125u;
    for (int i = 0; i < N_SAMP; i++) {
        uint32_t r0 = next_rng(&rng), r1 = next_rng(&rng), r2 = next_rng(&rng);
        s0[i].r = (i == 0) ? (lc_c16_t){0, 0}          : (lc_c16_t){(int16_t)(r0 & 0xFFFFu), (int16_t)(r0 >> 16)};
        s0[i].g = (i == 1) ? (lc_c16_t){-32768, -32768}: (lc_c16_t){(int16_t)(r1 & 0xFFFFu), (int16_t)(r1 >> 16)};
        s0[i].b = (lc_c16_t){(int16_t)(r2 & 0xFFFFu), (int16_t)(r2 >> 16)};
        st[i]   = s0[i];
    }

    double t0 = now_ns();
    for (int fi = 0; fi < nf; fi++) {
        for (int i = 0; i < N_SAMP; i++) pm_face_rgb_step(m, alive_faces[fi], &st[i], false);
    }
    int scrambled = 0;
    for (int i = 0; i < N_SAMP; i++) {
        if (!lc_c16_eq(st[i].r, s0[i].r) && !lc_c16_eq(st[i].g, s0[i].g) && !lc_c16_eq(st[i].b, s0[i].b)) {
            scrambled++;
        }
    }
    for (int fi = nf - 1; fi >= 0; fi--) {
        for (int i = 0; i < N_SAMP; i++) pm_face_rgb_step(m, alive_faces[fi], &st[i], true);
    }
    double dt_ms = (now_ns() - t0) * 1e-6;

    if (scrambled != N_SAMP || memcmp(st, s0, sizeof(s0)) != 0) {
        fprintf(stderr, "FAIL: Test 5 %s 3 ストリーム可逆輸送違反\n", label);
        return false;
    }
    printf("  %-22s (全 %2d 生存面連続通過, 攪拌=%d/256): 逆走 0-bit 完全復元 (%.2f ms) OK\n",
           label, nf, scrambled, dt_ms);
    return true;
}

int main(void) {
    if (!lc_v2_init()) {
        fprintf(stderr, "FATAL: lc_v2_init 失敗\n");
        return 1;
    }

    printf("=== ゼロヒープ 2 次元パッフナー移動 (1-3 / 3-1 / 2-2) 0-bit 位相保存実証 ===\n");
    printf("プール上限 = (V:%d, E:%d, F:%d) | sizeof(lc_pachner_mesh_t) = %.1f KB (ゼロヒープ)\n\n",
           PM_MAX_V, PM_MAX_E, PM_MAX_F, (double)sizeof(lc_pachner_mesh_t) / 1024.0);

    static lc_pachner_mesh_t deformed_s2, deformed_t2;

    if (!run_test1_basic_moves_and_guards())                    return 1;
    if (!run_test2_adaptive_metabolism_and_reuse())             return 1;
    if (!run_test3_s2_100moves(&deformed_s2))                   return 1;
    if (!run_test4_t2_100moves_harmonic_and_h2(&deformed_t2))   return 1;

    printf("[Test 5] 100 連パッフナー変形後の不規則 S² / T² メッシュ上の 3 ストリーム (R-G-B) 可逆輸送:\n");
    if (!verify_mesh_rgb_transport(&deformed_s2, "変形後 S² 不規則メッシュ")) return 1;
    if (!verify_mesh_rgb_transport(&deformed_t2, "変形後 T² 不規則メッシュ")) return 1;
    printf("\n>>> ALL 5 ZERO-HEAP 2D PACHNER MOVE & 0-BIT INVARIANCE TESTS PASSED <<<\n");
    return 0;
}