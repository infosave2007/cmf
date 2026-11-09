// A sealed layer's attention for a chunk of rows, on Metal: the twin of `o1b_far_e` and the
// `o1c_*` stages of `gpu_wgpu/o1_stages.rs`. One row is a chunk of one, so a token, a verify
// and a prompt chunk take the same kernels. Layouts as there:
//   meta[g*4] = win_len | win_head | far_len | group window
//   ring k/v [g][w][d], sink k/v [g][ns][d], k_tilde [g][m][d]
//   q_tilde [gh][m][d], mu [gh][m][m], mz [gh][2m], t_hat [gh][m][d]
//   q [row][nh][d], k/v [row][g][d], out [row][nh][d]
#include <metal_stdlib>
using namespace metal;

struct OC {
    uint hpg, m, w, ns;
    uint d, nh, ng, k;
    uint r0, ghm;
    float scale;
    uint pad;
};
#define OB_K 64u

static inline uint wgf(uint gw, constant OC& u) { return gw != 0u ? gw : u.w; }
static inline float hs(float4 v) { return v.x + v.y + v.z + v.w; }

// One thread per (group, head, landmark): every key the chunk evicts from the window against
// the landmark's query. The far max goes to its value after the chunk, and each row gets its
// prefix of the far mass.
kernel void o1c_far_e(
    device const uint*   meta [[buffer(0)]],
    device const float4* rk   [[buffer(1)]],
    device const float4* qt   [[buffer(2)]],
    device const float*  mz   [[buffer(3)]],
    device float*        s    [[buffer(4)]],
    constant OC&         u    [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint hm = u.hpg * u.m;
    if (id >= u.ng * hm) return;
    uint g = id / hm;
    uint rr = id % hm;
    uint h = rr / u.m;
    uint a = rr % u.m;
    uint len0 = meta[g * 4u];
    uint head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint j0 = wg - len0;
    uint e4 = u.d / 4u;
    uint gha = (g * u.hpg + h) * u.m + a;
    uint mzb = (g * u.hpg + h) * 2u * u.m;
    float m_old = mz[mzb + a];
    float mx = m_old;
    float fe[OB_K];
    device const float4* q = qt + gha * e4;
    for (uint j = j0; j < u.k; ++j) {
        device const float4* key = rk + (g * u.w + (head0 + j - j0) % wg) * e4;
        float4 acc = 0.0f;
        for (uint e = 0u; e < e4; ++e) acc += q[e] * key[e];
        fe[j] = hs(acc) * u.scale;
        mx = max(mx, fe[j]);
    }
    // An empty far field starts at -inf: no eviction keeps it (R = 1), the first one
    // rescales the zero mass by exp(-inf) = 0.
    float r = mx == m_old ? 1.0f : exp(m_old - mx);
    float z = mz[mzb + u.m + a] * r;
    for (uint j = 0u; j < u.k; ++j) {
        float e = j >= j0 ? exp(fe[j] - mx) : 0.0f;
        z += e;
        s[gha * OB_K + j] = e;
        s[(u.ghm + gha) * OB_K + j] = z;
    }
    s[2u * u.ghm * OB_K + gha] = mx;
    s[2u * u.ghm * OB_K + u.ghm + gha] = r;
}

// Near scores: a thread per (key, group, row). A row's window runs over the ring and the
// chunk's rows up to its own.
kernel void o1c_score(
    device const uint*   meta [[buffer(0)]],
    device const float4* q    [[buffer(1)]],
    device const float4* rk   [[buffer(2)]],
    device const float4* sk   [[buffer(3)]],
    device const float4* kb   [[buffer(4)]],
    device float*        sc   [[buffer(5)]],
    constant OC&         u    [[buffer(6)]],
    uint3 id [[thread_position_in_grid]])
{
    uint s = id.x, g = id.y, i = id.z;
    uint len0 = meta[g * 4u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint win = min(wg, len0 + i + 1u);
    if (s >= u.ns + win) return;
    uint e4 = u.d / 4u;
    device const float4* key;
    if (s < u.ns) {
        key = sk + (g * u.ns + s) * e4;
    } else {
        uint x = len0 + i + 1u - win + s - u.ns;
        if (x < len0) {
            key = rk + (g * u.w + (meta[g * 4u + 1u] + x) % wg) * e4;
        } else {
            key = kb + ((u.r0 + x - len0) * u.ng + g) * e4;
        }
    }
    device const float4* qq = q + ((u.r0 + i) * u.nh + g * u.hpg) * e4;
    uint at = ((i * u.ng + g) * (u.w + u.ns) + s) * u.hpg;
    for (uint h = 0u; h < u.hpg; ++h) {
        float4 acc = 0.0f;
        for (uint e = 0u; e < e4; ++e) acc += key[e] * qq[h * e4 + e];
        sc[at + h] = hs(acc) * u.scale;
    }
}

// Landmark scores.
kernel void o1c_land(
    device const float4* q  [[buffer(0)]],
    device const float4* kt [[buffer(1)]],
    device float*        fs [[buffer(2)]],
    constant OC&         u  [[buffer(3)]],
    uint3 id [[thread_position_in_grid]])
{
    uint a = id.x, g = id.y, i = id.z;
    if (a >= u.m) return;
    uint e4 = u.d / 4u;
    device const float4* key = kt + (g * u.m + a) * e4;
    device const float4* qq = q + ((u.r0 + i) * u.nh + g * u.hpg) * e4;
    uint at = ((i * u.ng + g) * u.m + a) * u.hpg;
    for (uint h = 0u; h < u.hpg; ++h) {
        float4 acc = 0.0f;
        for (uint e = 0u; e < e4; ++e) acc += key[e] * qq[h * e4 + e];
        fs[at + h] = hs(acc) * u.scale;
    }
}

// The largest near score of a (row, query head): one simdgroup each.
kernel void o1c_max(
    device const uint*  meta [[buffer(0)]],
    device const float* sc   [[buffer(1)]],
    device float*       nm   [[buffer(2)]],
    constant OC&        u    [[buffer(3)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint gh = tg.x, i = tg.y;
    uint g = gh / u.hpg, h = gh % u.hpg;
    uint n = u.ns + min(wgf(meta[g * 4u + 3u], u), meta[g * 4u] + i + 1u);
    uint sb = (i * u.ng + g) * (u.w + u.ns);
    float m = -1.0e30f;
    for (uint j = lane; j < n; j += 32u) m = max(m, sc[(sb + j) * u.hpg + h]);
    float mx = simd_max(m);
    if (lane == 0u) nm[i * u.nh + gh] = mx;
}

// The far part of a (row, query head), with the far state as of this row: the mass up to it
// and the weight of each row the chunk has evicted so far.
kernel void o1c_far(
    device const uint*  meta [[buffer(0)]],
    device const float* fs   [[buffer(1)]],
    device const float* nm   [[buffer(2)]],
    device const float* mu   [[buffer(3)]],
    device const float* s    [[buffer(4)]],
    device float4*      hd   [[buffer(5)]],
    device float*       gn   [[buffer(6)]],
    device float*       cf   [[buffer(7)]],
    constant OC&        u    [[buffer(8)]],
    uint2 id [[thread_position_in_grid]])
{
    uint gh = id.x, i = id.y;
    if (gh >= u.nh) return;
    uint g = gh / u.hpg, h = gh % u.hpg, m = u.m;
    uint j0 = wgf(meta[g * 4u + 3u], u) - meta[g * 4u];
    uint farl = meta[g * 4u + 2u];
    if (i >= j0) farl += i - j0 + 1u;
    uint at0 = i * u.nh + gh;
    float c_all = nm[at0];
    float far_den = 0.0f, have = 0.0f;
    if (farl > 0u) {
        float f = -3.0e38f;
        float fe[32];
        float w[32];
        float gain[32];
        for (uint a = 0u; a < m; ++a) {
            fe[a] = fs[((i * u.ng + g) * m + a) * u.hpg + h];
            f = max(f, fe[a]);
            w[a] = 0.0f;
        }
        for (uint a = 0u; a < m; ++a) {
            float e = exp(fe[a] - f);
            device const float* row = mu + (gh * m + a) * m;
            for (uint b = 0u; b < m; ++b) w[b] += e * row[b];
        }
        uint mb = 2u * u.ghm * OB_K;
        for (uint b = 0u; b < m; ++b) c_all = max(c_all, f + s[mb + gh * m + b]);
        for (uint b = 0u; b < m; ++b) {
            gain[b] = w[b] * exp(f + s[mb + gh * m + b] - c_all);
            far_den += gain[b] * s[(u.ghm + gh * m + b) * OB_K + i];
            gn[at0 * m + b] = gain[b] * s[mb + u.ghm + gh * m + b];
        }
        if (far_den >= 0.0f) { have = 1.0f; } else { far_den = 0.0f; }
        if (have > 0.5f && i >= j0) {
            for (uint j = j0; j <= i; ++j) {
                float c = 0.0f;
                for (uint b = 0u; b < m; ++b) c += gain[b] * s[(gh * m + b) * OB_K + j];
                cf[at0 * OB_K + j] = c;
            }
        }
    }
    hd[at0] = float4(c_all, far_den, have, 0.0f);
}

// The near weights and 1/Σ: one simdgroup per (row, query head).
kernel void o1c_soft(
    device const uint*   meta [[buffer(0)]],
    device float*        sc   [[buffer(1)]],
    device const float4* hd   [[buffer(2)]],
    device float*        inv  [[buffer(3)]],
    constant OC&         u    [[buffer(4)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint gh = tg.x, i = tg.y;
    uint g = gh / u.hpg, h = gh % u.hpg;
    uint n = u.ns + min(wgf(meta[g * 4u + 3u], u), meta[g * 4u] + i + 1u);
    uint sb = (i * u.ng + g) * (u.w + u.ns);
    float4 hh = hd[i * u.nh + gh];
    float den = 0.0f;
    for (uint j = lane; j < n; j += 32u) {
        uint at = (sb + j) * u.hpg + h;
        float e = exp(sc[at] - hh.x);
        sc[at] = e;
        den += e;
    }
    float total = simd_sum(den) + hh.y;
    if (lane == 0u) inv[i * u.nh + gh] = 1.0f / max(total, 1.0e-30f);
}

// The near value sum in eight pieces: a thread per (row, KV head, piece, vec4 of the head).
kernel void o1c_sum(
    device const uint*   meta [[buffer(0)]],
    device const float*  sc   [[buffer(1)]],
    device const float4* rv   [[buffer(2)]],
    device const float4* sv   [[buffer(3)]],
    device const float4* vb   [[buffer(4)]],
    device float4*       pz   [[buffer(5)]],
    constant OC&         u    [[buffer(6)]],
    uint2 id [[thread_position_in_grid]])
{
    uint e4 = u.d / 4u;
    uint t = id.x, i = id.y;
    uint d = t % e4;
    uint ch = (t / e4) % 8u;
    uint g = t / (e4 * 8u);
    if (g >= u.ng) return;
    uint len0 = meta[g * 4u];
    uint head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint win = min(wg, len0 + i + 1u);
    uint lo = len0 + i + 1u - win;
    uint n = u.ns + win;
    uint per = (n + 7u) / 8u;
    uint sb = (i * u.ng + g) * (u.w + u.ns);
    float4 o[8];
    for (uint h = 0u; h < u.hpg; ++h) o[h] = 0.0f;
    uint last = min((ch + 1u) * per, n);
    for (uint j = ch * per; j < last; ++j) {
        float4 v;
        if (j < u.ns) {
            v = sv[(g * u.ns + j) * e4 + d];
        } else {
            uint x = lo + j - u.ns;
            v = x < len0 ? rv[(g * u.w + (head0 + x) % wg) * e4 + d] : vb[((u.r0 + x - len0) * u.ng + g) * e4 + d];
        }
        device const float* a = sc + (sb + j) * u.hpg;
        for (uint h = 0u; h < u.hpg; ++h) o[h] += a[h] * v;
    }
    for (uint h = 0u; h < u.hpg; ++h) pz[(((i * u.ng + g) * 8u + ch) * u.hpg + h) * e4 + d] = o[h];
}

// The pieces, the far values, the rows evicted so far, 1/Σ.
kernel void o1c_add(
    device const uint*   meta [[buffer(0)]],
    device const float4* pz   [[buffer(1)]],
    device const float4* hd   [[buffer(2)]],
    device const float*  gn   [[buffer(3)]],
    device const float4* th   [[buffer(4)]],
    device const float*  inv  [[buffer(5)]],
    device const float*  cf   [[buffer(6)]],
    device const float4* rv   [[buffer(7)]],
    device float4*       out  [[buffer(8)]],
    constant OC&         u    [[buffer(9)]],
    uint2 id [[thread_position_in_grid]])
{
    uint e = id.x, i = id.y;
    uint e4 = u.d / 4u;
    if (e >= u.nh * e4) return;
    uint d = e % e4;
    uint gh = e / e4;
    uint g = gh / u.hpg, h = gh % u.hpg;
    uint at0 = i * u.nh + gh;
    float4 acc = 0.0f;
    for (uint ch = 0u; ch < 8u; ++ch) acc += pz[(((i * u.ng + g) * 8u + ch) * u.hpg + h) * e4 + d];
    if (hd[at0].z > 0.5f) {
        for (uint b = 0u; b < u.m; ++b) acc += gn[at0 * u.m + b] * th[(gh * u.m + b) * e4 + d];
        uint wg = wgf(meta[g * 4u + 3u], u);
        uint j0 = wg - meta[g * 4u];
        if (i >= j0) {
            uint head0 = meta[g * 4u + 1u];
            for (uint j = j0; j <= i; ++j) {
                acc += cf[at0 * OB_K + j] * rv[(g * u.w + (head0 + j - j0) % wg) * e4 + d];
            }
        }
    }
    out[((u.r0 + i) * u.nh + gh) * e4 + d] = acc * inv[at0];
}

// The far numerator and mass after the chunk: rescaled to its max, plus every evicted row.
// After the attention, which reads the numerator as it was, and before the ring commit,
// which overwrites the evicted rows.
kernel void o1c_far_commit(
    device const uint* meta [[buffer(0)]],
    device const float4* rv [[buffer(1)]],
    device float*      mz   [[buffer(2)]],
    device float4*     th   [[buffer(3)]],
    device const float* s   [[buffer(4)]],
    constant OC&       u    [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    uint e4 = u.d / 4u;
    uint t = id.x;
    uint hm = u.hpg * u.m;
    uint ghb = id.y;
    if (t >= e4 || ghb >= u.ghm) return;
    uint g = ghb / hm;
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint len0 = meta[g * 4u];
    uint j0 = wg - len0;
    if (j0 >= u.k) return;
    uint head0 = meta[g * 4u + 1u];
    float r = s[2u * u.ghm * OB_K + u.ghm + ghb];
    float4 acc = th[ghb * e4 + t] * r;
    for (uint j = j0; j < u.k; ++j) {
        acc += s[ghb * OB_K + j] * rv[(g * u.w + (head0 + j - j0) % wg) * e4 + t];
    }
    th[ghb * e4 + t] = acc;
    if (t == 0u) {
        uint gh = ghb / u.m, b = ghb % u.m;
        uint mzb = gh * 2u * u.m;
        mz[mzb + b] = s[2u * u.ghm * OB_K + ghb];
        mz[mzb + u.m + b] = s[(u.ghm + ghb) * OB_K + u.k - 1u];
    }
}

// The chunk's rotated K and V into the ring. Row j lands where its eviction was.
kernel void o1c_ring_commit(
    device const uint*   meta [[buffer(0)]],
    device float4*       rk   [[buffer(1)]],
    device float4*       rv   [[buffer(2)]],
    device const float4* kb   [[buffer(3)]],
    device const float4* vb   [[buffer(4)]],
    constant OC&         u    [[buffer(5)]],
    uint3 id [[thread_position_in_grid]])
{
    uint t = id.x, g = id.y, j = id.z;
    uint e4 = u.d / 4u;
    if (t >= e4) return;
    uint len0 = meta[g * 4u];
    uint head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint slot = (head0 + len0 + j) % wg;
    uint row = u.r0 + j;
    rk[(g * u.w + slot) * e4 + t] = kb[(row * u.ng + g) * e4 + t];
    rv[(g * u.w + slot) * e4 + t] = vb[(row * u.ng + g) * e4 + t];
}

// Ring length, head and far count after k pushes.
kernel void o1c_meta(
    device uint* meta [[buffer(0)]],
    constant OC& u    [[buffer(1)]],
    uint g [[thread_position_in_grid]])
{
    if (g >= u.ng) return;
    uint len0 = meta[g * 4u];
    uint head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint tot = len0 + u.k;
    if (tot <= wg) {
        meta[g * 4u] = tot;
    } else {
        uint ev = tot - wg;
        meta[g * 4u] = wg;
        meta[g * 4u + 1u] = (head0 + ev) % wg;
        meta[g * 4u + 2u] = meta[g * 4u + 2u] + ev;
    }
}

// A buffer's start into another one that outlives the graph's scratch.
kernel void o1c_copy(
    device const float4* src [[buffer(0)]],
    device float4*       dst [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    dst[i] = src[i];
}

// ── The near window as GEMMs (a chunk of 16 rows or more) ─────────────────────────────────
// Per group the window's keys and values are laid out as one panel: the sinks, the ring from
// its oldest row, then the chunk's rows; `ncap` columns, the unused ones zero. The scores
// `Q·Kᵀ` and the sum `P·V` then run on the engine's tiled GEMMs, and row i of the chunk reads
// the columns its own window covers. Scores are [gh][k][ncap], the near sum [gh][k][d].

// A thread per (vec4 of the head, column, group).
kernel void o1c_gather(
    device const uint*   meta [[buffer(0)]],
    device const float4* rk   [[buffer(1)]],
    device const float4* rv   [[buffer(2)]],
    device const float4* sk   [[buffer(3)]],
    device const float4* sv   [[buffer(4)]],
    device const float4* kb   [[buffer(5)]],
    device const float4* vb   [[buffer(6)]],
    device float4*       kp   [[buffer(7)]],
    device float4*       vp   [[buffer(8)]],
    constant OC&         u    [[buffer(9)]],
    uint3 id [[thread_position_in_grid]])
{
    uint e4 = u.d / 4u;
    uint e = id.x, j = id.y, g = id.z;
    uint ncap = u.ns + u.w + u.k;
    if (e >= e4 || j >= ncap || g >= u.ng) return;
    uint len0 = meta[g * 4u];
    uint head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    float4 kv = 0.0f, vv = 0.0f;
    if (j < u.ns) {
        kv = sk[(g * u.ns + j) * e4 + e];
        vv = sv[(g * u.ns + j) * e4 + e];
    } else if (j < u.ns + len0) {
        uint x = j - u.ns;
        uint at = (g * u.w + (head0 + x) % wg) * e4 + e;
        kv = rk[at];
        vv = rv[at];
    } else if (j < u.ns + len0 + u.k) {
        uint at = ((u.r0 + j - u.ns - len0) * u.ng + g) * e4 + e;
        kv = kb[at];
        vv = vb[at];
    }
    uint o = (g * ncap + j) * e4 + e;
    kp[o] = kv;
    vp[o] = vv;
}

// The columns row i of a group may read: the sinks, then its window over the ring and the
// chunk's rows up to its own.
static inline void o1c_cols(device const uint* meta, constant OC& u, uint g, uint i, thread uint& lo, thread uint& hi) {
    uint len0 = meta[g * 4u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint win = min(wg, len0 + i + 1u);
    lo = u.ns + len0 + i + 1u - win;
    hi = u.ns + len0 + i + 1u;
}

// The largest near score of a (row, query head): one simdgroup each.
kernel void o1c_gmax(
    device const uint*  meta [[buffer(0)]],
    device const float* sc   [[buffer(1)]],
    device float*       nm   [[buffer(2)]],
    constant OC&        u    [[buffer(3)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint gh = tg.x, i = tg.y;
    uint g = gh / u.hpg;
    uint ncap = u.ns + u.w + u.k;
    uint lo, hi;
    o1c_cols(meta, u, g, i, lo, hi);
    device const float* r = sc + ((ulong)gh * u.k + i) * ncap;
    float m = -1.0e30f;
    for (uint j = lane; j < u.ns; j += 32u) m = max(m, r[j]);
    for (uint j = lo + lane; j < hi; j += 32u) m = max(m, r[j]);
    float mx = simd_max(m);
    if (lane == 0u) nm[i * u.nh + gh] = mx;
}

// The near weights in place (zero outside the row's columns) and 1/Σ with the far mass.
kernel void o1c_gsoft(
    device const uint*   meta [[buffer(0)]],
    device float*        sc   [[buffer(1)]],
    device const float4* hd   [[buffer(2)]],
    device float*        inv  [[buffer(3)]],
    constant OC&         u    [[buffer(4)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]])
{
    uint gh = tg.x, i = tg.y;
    uint g = gh / u.hpg;
    uint ncap = u.ns + u.w + u.k;
    uint lo, hi;
    o1c_cols(meta, u, g, i, lo, hi);
    device float* r = sc + ((ulong)gh * u.k + i) * ncap;
    float4 hh = hd[i * u.nh + gh];
    float den = 0.0f;
    for (uint j = lane; j < ncap; j += 32u) {
        bool live = j < u.ns || (j >= lo && j < hi);
        float e = live ? exp(r[j] - hh.x) : 0.0f;
        r[j] = e;
        den += e;
    }
    float total = simd_sum(den) + hh.y;
    if (lane == 0u) inv[i * u.nh + gh] = 1.0f / max(total, 1.0e-30f);
}

// `o1c_add` with the near sum from the GEMM.
kernel void o1c_gadd(
    device const uint*   meta [[buffer(0)]],
    device const float4* no   [[buffer(1)]],
    device const float4* hd   [[buffer(2)]],
    device const float*  gn   [[buffer(3)]],
    device const float4* th   [[buffer(4)]],
    device const float*  inv  [[buffer(5)]],
    device const float*  cf   [[buffer(6)]],
    device const float4* rv   [[buffer(7)]],
    device float4*       out  [[buffer(8)]],
    constant OC&         u    [[buffer(9)]],
    uint2 id [[thread_position_in_grid]])
{
    uint e = id.x, i = id.y;
    uint e4 = u.d / 4u;
    if (e >= u.nh * e4) return;
    uint d = e % e4;
    uint gh = e / e4;
    uint g = gh / u.hpg;
    uint at0 = i * u.nh + gh;
    float4 acc = no[((ulong)gh * u.k + i) * e4 + d];
    if (hd[at0].z > 0.5f) {
        for (uint b = 0u; b < u.m; ++b) acc += gn[at0 * u.m + b] * th[(gh * u.m + b) * e4 + d];
        uint wg = wgf(meta[g * 4u + 3u], u);
        uint j0 = wg - meta[g * 4u];
        if (i >= j0) {
            uint head0 = meta[g * 4u + 1u];
            for (uint j = j0; j <= i; ++j) {
                acc += cf[at0 * OB_K + j] * rv[(g * u.w + (head0 + j - j0) % wg) * e4 + d];
            }
        }
    }
    out[((u.r0 + i) * u.nh + gh) * e4 + d] = acc * inv[at0];
}

// ── A token (one row) in four dispatches ───────────────────────────────────────────────────
// `o1c_tok_*`: the stages above collapsed for a chunk of one row, where their eleven dispatches
// and barriers were the layer's cost (about 0.8 ms a layer on the M4 for a 2052-key window).
// The near window runs as flash-decoding partials — a threadgroup per (group, split), each
// simdgroup a stripe of the window's keys with the row's queries in registers; the far part as
// one simdgroup per query head, lanes = landmarks; a merge over the partials and the far part;
// the state commit as one threadgroup per group. Heads ≤ 4 per group, d ≤ 256 a multiple of
// 128, m ≤ 32 (`tok_ok`). The sums run in another order than the stages' (per-lane partials
// and online rescaling), so not bitwise against them.
#define TOK_SPLIT 4u
#define TOK_SG 8u
#define TOK_PSTRIDE(d) ((d) + 4u)
#define TOK_FO 36u

kernel void o1c_tok_near(
    device const uint*   meta [[buffer(0)]],
    device const float4* q    [[buffer(1)]],
    device const float4* rk   [[buffer(2)]],
    device const float4* sk   [[buffer(3)]],
    device const float4* kb   [[buffer(4)]],
    device const float4* rv   [[buffer(5)]],
    device const float4* sv   [[buffer(6)]],
    device const float4* vb   [[buffer(7)]],
    device float*        part [[buffer(8)]],
    constant OC&         u    [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint g = tg.x, split = tg.y;
    uint len0 = meta[g * 4u], head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint win = min(wg, len0 + 1u);
    uint lo = len0 + 1u - win;
    uint n = u.ns + win;
    uint e4 = u.d / 4u;
    uint per = e4 / 32u;
    device const float4* qq = q + ((ulong)u.r0 * u.nh + g * u.hpg) * e4;
    float4 qr[4][2];
    for (uint h = 0u; h < 4u; ++h) {
        for (uint p = 0u; p < 2u; ++p) {
            qr[h][p] = (h < u.hpg && p < per) ? qq[h * e4 + p * 32u + lane] : float4(0.0f);
        }
    }
    float mh[4], lh[4];
    float4 acc[4][2];
    for (uint h = 0u; h < 4u; ++h) { mh[h] = -1.0e30f; lh[h] = 0.0f; acc[h][0] = 0.0f; acc[h][1] = 0.0f; }
    uint kps = (n + TOK_SPLIT - 1u) / TOK_SPLIT;
    uint jlo = split * kps, jhi = min(n, jlo + kps);
    for (uint j = jlo + sg; j < jhi; j += TOK_SG) {
        device const float4* key;
        device const float4* val;
        if (j < u.ns) {
            key = sk + (ulong)(g * u.ns + j) * e4;
            val = sv + (ulong)(g * u.ns + j) * e4;
        } else {
            uint x = lo + j - u.ns;
            if (x < len0) {
                uint slot = g * u.w + (head0 + x) % wg;
                key = rk + (ulong)slot * e4;
                val = rv + (ulong)slot * e4;
            } else {
                uint row = (u.r0 + x - len0) * u.ng + g;
                key = kb + (ulong)row * e4;
                val = vb + (ulong)row * e4;
            }
        }
        float4 k0 = key[lane], k1 = per > 1u ? key[32u + lane] : float4(0.0f);
        float4 v0 = val[lane], v1 = per > 1u ? val[32u + lane] : float4(0.0f);
        for (uint h = 0u; h < 4u; ++h) {
            if (h >= u.hpg) break;
            float sc = simd_sum(hs(k0 * qr[h][0] + k1 * qr[h][1])) * u.scale;
            float mn = max(mh[h], sc);
            float c = exp(mh[h] - mn), e = exp(sc - mn);
            lh[h] = lh[h] * c + e;
            acc[h][0] = acc[h][0] * c + e * v0;
            acc[h][1] = acc[h][1] * c + e * v1;
            mh[h] = mn;
        }
    }
    uint pi = (g * TOK_SPLIT + split) * TOK_SG + sg;
    for (uint h = 0u; h < u.hpg; ++h) {
        device float* pb = part + (ulong)(pi * u.hpg + h) * TOK_PSTRIDE(u.d);
        if (lane == 0u) { pb[0] = mh[h]; pb[1] = lh[h]; }
        device float4* pa = (device float4*)(pb + 4u);
        pa[lane] = acc[h][0];
        if (per > 1u) pa[32u + lane] = acc[h][1];
    }
}

// The far part of the row for one query head: lane a is landmark a. The eviction (`o1c_far_e`
// for k = 1) and the state scratch `s` it leaves for the commit, the landmark scores, the gains
// relative to the far max `fm` (the merge rescales them to the row's max), the far mass, the
// evicted row's weight `cf`.
kernel void o1c_tok_far(
    device const uint*   meta [[buffer(0)]],
    device const float4* q    [[buffer(1)]],
    device const float4* kt   [[buffer(2)]],
    device const float4* rk   [[buffer(3)]],
    device const float4* qt   [[buffer(4)]],
    device const float*  mz   [[buffer(5)]],
    device const float*  mu   [[buffer(6)]],
    device float*        s    [[buffer(7)]],
    device float*        fo   [[buffer(8)]],
    constant OC&         u    [[buffer(9)]],
    uint gh   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint g = gh / u.hpg, m = u.m;
    uint len0 = meta[g * 4u], head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    bool evict = len0 == wg;
    uint farl = meta[g * 4u + 2u] + (evict ? 1u : 0u);
    uint e4 = u.d / 4u;
    uint a = lane;
    bool live = a < m;
    uint gha = gh * m + a;
    uint mzb = gh * 2u * m;
    float m_old = live ? mz[mzb + a] : 0.0f;
    float mass = live ? mz[mzb + m + a] : 0.0f;
    float mx = m_old, e = 0.0f;
    if (live && evict) {
        device const float4* qa = qt + (ulong)gha * e4;
        device const float4* key = rk + (ulong)(g * u.w + head0 % wg) * e4;
        float4 acc = 0.0f;
        for (uint i = 0u; i < e4; ++i) acc += qa[i] * key[i];
        float fe = hs(acc) * u.scale;
        mx = max(m_old, fe);
        e = exp(fe - mx);
    }
    float rr = mx == m_old ? 1.0f : exp(m_old - mx);
    float z = mass * rr + e;
    if (live) {
        s[gha * OB_K] = e;
        s[(u.ghm + gha) * OB_K] = z;
        s[2u * u.ghm * OB_K + gha] = mx;
        s[2u * u.ghm * OB_K + u.ghm + gha] = rr;
    }
    float fs = -3.0e38f;
    if (live) {
        device const float4* qq = q + ((ulong)u.r0 * u.nh + gh) * e4;
        device const float4* key = kt + (ulong)(g * m + a) * e4;
        float4 acc = 0.0f;
        for (uint i = 0u; i < e4; ++i) acc += key[i] * qq[i];
        fs = hs(acc) * u.scale;
    }
    float fm = -3.0e38f, far_den = 0.0f, have = 0.0f, cf = 0.0f, gn = 0.0f;
    if (farl > 0u) {
        float f = simd_max(fs);
        float ea = live ? exp(fs - f) : 0.0f;
        float w = 0.0f;
        for (uint aa = 0u; aa < m; ++aa) {
            float eaa = simd_shuffle(ea, aa);
            w += eaa * (live ? mu[((ulong)gh * m + aa) * m + a] : 0.0f);
        }
        float fb = live ? f + mx : -3.0e38f;
        fm = simd_max(fb);
        float gain = live ? w * exp(fb - fm) : 0.0f;
        far_den = simd_sum(gain * z);
        gn = gain * rr;
        cf = evict ? simd_sum(gain * e) : 0.0f;
        if (far_den >= 0.0f) { have = 1.0f; } else { far_den = 0.0f; }
    }
    device float* o = fo + (ulong)gh * TOK_FO;
    if (lane == 0u) { o[0] = fm; o[1] = far_den; o[2] = have; o[3] = cf; }
    if (live) o[4u + a] = gn;
}

// The partials and the far part into the row: a thread per (query head, vec4 of the head).
kernel void o1c_tok_merge(
    device const uint*   meta [[buffer(0)]],
    device const float*  part [[buffer(1)]],
    device const float*  fo   [[buffer(2)]],
    device const float4* th   [[buffer(3)]],
    device const float4* rv   [[buffer(4)]],
    device float4*       out  [[buffer(5)]],
    constant OC&         u    [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint e4 = u.d / 4u;
    if (id >= u.nh * e4) return;
    uint d = id % e4, gh = id / e4;
    uint g = gh / u.hpg, h = gh % u.hpg;
    device const float* o = fo + (ulong)gh * TOK_FO;
    float fm = o[0], far_den = o[1], have = o[2], cf = o[3];
    float c = have > 0.5f ? fm : -1.0e30f;
    uint np = TOK_SPLIT * TOK_SG;
    uint ps = TOK_PSTRIDE(u.d);
    for (uint pi = 0u; pi < np; ++pi) c = max(c, part[(ulong)((g * np + pi) * u.hpg + h) * ps]);
    float den = 0.0f;
    float4 acc = 0.0f;
    for (uint pi = 0u; pi < np; ++pi) {
        device const float* pb = part + (ulong)((g * np + pi) * u.hpg + h) * ps;
        float mm = pb[0];
        if (mm > -1.0e29f) {
            float sc = exp(mm - c);
            den += pb[1] * sc;
            acc += ((device const float4*)(pb + 4u))[d] * sc;
        }
    }
    if (have > 0.5f) {
        float sf = exp(fm - c);
        den += far_den * sf;
        float4 fa = 0.0f;
        for (uint b = 0u; b < u.m; ++b) fa += o[4u + b] * th[((ulong)gh * u.m + b) * e4 + d];
        uint len0 = meta[g * 4u], head0 = meta[g * 4u + 1u];
        uint wg = wgf(meta[g * 4u + 3u], u);
        if (len0 == wg) fa += cf * rv[(ulong)(g * u.w + head0 % wg) * e4 + d];
        acc += fa * sf;
    }
    out[((ulong)u.r0 * u.nh + gh) * e4 + d] = acc / max(den, 1.0e-30f);
}

// The state after the row, one threadgroup per group: the evicted row into the far numerators
// with the max and mass after it (`o1c_far_commit`), then — after a barrier, it overwrites that
// row — the row's K/V into the ring (`o1c_ring_commit`) and the meta (`o1c_meta`).
kernel void o1c_tok_commit(
    device uint*         meta [[buffer(0)]],
    device float4*       rk   [[buffer(1)]],
    device float4*       rv   [[buffer(2)]],
    device const float4* kb   [[buffer(3)]],
    device const float4* vb   [[buffer(4)]],
    device float*        mz   [[buffer(5)]],
    device float4*       th   [[buffer(6)]],
    device const float*  s    [[buffer(7)]],
    constant OC&         u    [[buffer(8)]],
    uint g   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint len0 = meta[g * 4u], head0 = meta[g * 4u + 1u];
    uint wg = wgf(meta[g * 4u + 3u], u);
    uint e4 = u.d / 4u, m = u.m;
    bool evict = len0 == wg;
    uint slot = evict ? head0 % wg : (head0 + len0) % wg;
    if (evict) {
        uint hm = u.hpg * m;
        for (uint idx = tid; idx < hm * e4; idx += 256u) {
            uint ha = idx / e4, t = idx % e4;
            uint ghb = g * hm + ha;
            float rr = s[2u * u.ghm * OB_K + u.ghm + ghb];
            float4 acc = th[(ulong)ghb * e4 + t] * rr + s[ghb * OB_K] * rv[(ulong)(g * u.w + slot) * e4 + t];
            th[(ulong)ghb * e4 + t] = acc;
            if (t == 0u) {
                uint gh = ghb / m, b = ghb % m;
                uint mzb = gh * 2u * m;
                mz[mzb + b] = s[2u * u.ghm * OB_K + ghb];
                mz[mzb + m + b] = s[(u.ghm + ghb) * OB_K];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint t = tid; t < e4; t += 256u) {
        rk[(ulong)(g * u.w + slot) * e4 + t] = kb[(ulong)(u.r0 * u.ng + g) * e4 + t];
        rv[(ulong)(g * u.w + slot) * e4 + t] = vb[(ulong)(u.r0 * u.ng + g) * e4 + t];
    }
    if (tid == 0u) {
        if (!evict) {
            meta[g * 4u] = len0 + 1u;
        } else {
            meta[g * 4u] = wg;
            meta[g * 4u + 1u] = (head0 + 1u) % wg;
            meta[g * 4u + 2u] = meta[g * 4u + 2u] + 1u;
        }
    }
}
