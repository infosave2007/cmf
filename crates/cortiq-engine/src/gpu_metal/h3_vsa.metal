// Video sparse attention (VSA) for the Metal MiniMax-H3 chain: FastVideo's
// VSA-H3 as its FastH3 students were trained with it (see mmh3::VsaPlan and
// gpu_wgpu/h3_vsa.rs, the same schedule). Compiled after qi21_msl.metal
// (`qi_fc`) into a library of its own, built only for a VSA student.
//
// The panel keeps its rows in sequence order; `rowof` maps a tile slot to
// its row (0xFFFFFFFF: a pad slot). q in the panel is normed, rotated and
// pre-multiplied by log2(e)/sqrt(hd) (`h3_qkrope`), so the tile scores
// q̄·k̄ come out in log2 units: the selection's order is the same and the
// compression branch's softmax is exp2 of them. v keeps the panel's guard.

struct VPd {
    uint np;     // tile slots (64 a tile)
    uint ldp;    // panel row stride (halves)
    uint inner;  // heads · 128
    uint t;      // tiles
    uint p;      // non-video tiles (first)
    uint k;      // video key tiles a video query tile keeps
    uint wantv;  // the tile means of v too (the gated branch)
    uint n;      // rows
};

// the queries into tile-padded order (pad slots zero): a threadgroup a slot
kernel void vsa_qpad(
    device const half* P [[buffer(0)]],
    device const uint* rowof [[buffer(1)]],
    device half* Q [[buffer(2)]],
    constant VPd& v [[buffer(3)]],
    uint s [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (s >= v.np) return;
    const uint row = rowof[s];
    for (uint i = lid; i < v.inner; i += 256u)
        Q[(ulong)s * v.inner + i] = row != 0xFFFFFFFFu ? P[(ulong)row * v.ldp + i] : half(0.0h);
}

// tile means of q, k (and v): a threadgroup per (tile, head), a lane a dim
kernel void vsa_pool(
    device const half* P [[buffer(0)]],
    device const uint* rowof [[buffer(1)]],
    device const uint* size [[buffer(2)]],
    device float* pq [[buffer(3)]],
    device float* pk [[buffer(4)]],
    device float* pv [[buffer(5)]],
    constant VPd& v [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]])
{
    const uint t = tg.x, h = tg.y, sz = size[t];
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (uint i = 0; i < sz; ++i) {
        const ulong b = (ulong)rowof[t * 64u + i] * v.ldp + h * 128u + d;
        aq += float(P[b]);
        ak += float(P[b + v.inner]);
        if (v.wantv != 0u) av += float(P[b + 2u * v.inner]);
    }
    const float inv = 1.0f / float(sz);
    const uint o = (h * v.t + t) * 128u + d;
    pq[o] = aq * inv;
    pk[o] = ak * inv;
    if (v.wantv != 0u) pv[o] = av * inv;
}

// tile scores, 16×16 a threadgroup (256 threads), the dims through threadgroup memory
kernel void vsa_scores(
    device const float* pq [[buffer(0)]],
    device const float* pk [[buffer(1)]],
    device float* S [[buffer(2)]],
    constant VPd& v [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float a[16 * 129], b[16 * 129];
    const uint i0 = tg.x * 16u, j0 = tg.y * 16u, h = tg.z;
    for (uint c = tid; c < 2048u; c += 256u) {
        const uint r = c / 128u, d = c % 128u;
        a[r * 129u + d] = i0 + r < v.t ? pq[(h * v.t + i0 + r) * 128u + d] : 0.0f;
        b[r * 129u + d] = j0 + r < v.t ? pk[(h * v.t + j0 + r) * 128u + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tx = tid % 16u, ty = tid / 16u;
    float acc = 0.0f;
    for (uint d = 0; d < 128u; ++d) acc += a[ty * 129u + d] * b[tx * 129u + d];
    const uint i = i0 + ty, j = j0 + tx;
    if (i < v.t && j < v.t) S[((ulong)h * v.t + i) * v.t + j] = acc;
}

static inline uint vsa_okey(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// a video query tile's key tiles: the non-video ones, then its top-k video
// tiles (radix select of the k-th largest, a block scan writing them in
// ascending order: the same list every run). 256 threads.
kernel void vsa_select(
    device const float* S [[buffer(0)]],
    device uint* L [[buffer(1)]],
    constant VPd& v [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[256];
    threadgroup uint sh[2];
    const uint i = v.p + tg.x, h = tg.y;
    device const float* row = S + ((ulong)h * v.t + i) * v.t + v.p;
    const uint nv = v.t - v.p;
    device uint* out = L + ((ulong)h * v.t + i) * (v.p + v.k);
    for (uint c = tid; c < v.p; c += 256u) out[c] = c;
    uint prefix = 0u, mask = 0u, left = v.k;
    for (uint pass = 0; pass < 4u; ++pass) {
        const uint shift = 24u - 8u * pass;
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < nv; c += 256u) {
            const uint key = vsa_okey(row[c]);
            if ((key & mask) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint cum = 0u, digit = 0u;
            for (int bb = 255; bb >= 0; --bb) {
                const uint hv = atomic_load_explicit(&hist[bb], memory_order_relaxed);
                if (cum + hv >= left) {
                    digit = (uint)bb;
                    break;
                }
                cum += hv;
            }
            sh[0] = prefix | (digit << shift);
            sh[1] = left - cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh[0];
        left = sh[1];
        mask |= 255u << shift;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint gt_total = v.k - left;
    const uint per = (nv + 255u) / 256u, c0 = tid * per, c1 = min(nv, c0 + per);
    for (uint which = 0; which < 2u; ++which) {
        uint cnt = 0u;
        for (uint c = c0; c < c1; ++c) {
            const uint key = vsa_okey(row[c]);
            cnt += which == 0u ? (key > prefix ? 1u : 0u) : (key == prefix ? 1u : 0u);
        }
        scan[tid] = cnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = 1u; o < 256u; o <<= 1u) {
            const uint val = tid >= o ? scan[tid - o] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scan[tid] += val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint pos = scan[tid] - cnt;
        for (uint c = c0; c < c1; ++c) {
            const uint key = vsa_okey(row[c]);
            if (which == 0u && key > prefix) {
                out[v.p + pos] = v.p + c;
                ++pos;
            } else if (which == 1u && key == prefix) {
                if (pos < left) out[v.p + gt_total + pos] = v.p + c;
                ++pos;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// `qi_flash` over a query tile's key tiles: a threadgroup per (query tile,
// head), its 8 simdgroups the tile's 64 slots; keys 32 a block, two blocks
// a key tile, read from the panel rows the tile names; a key past its tile's
// size masked; the output into the query's own row.
kernel void vsa_flash(
    device const half* Q [[buffer(0)]],
    device const half* P [[buffer(1)]],
    device half* O [[buffer(2)]],
    device const uint* rowof [[buffer(3)]],
    device const uint* size [[buffer(4)]],
    device const uint* L [[buffer(5)]],
    constant VPd& v [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    constexpr ushort NSG = 8;
    constexpr uint NT = 32u * NSG, CH = 512u / NT;
    threadgroup half sk[32 * 128];
    threadgroup half sv[32 * 128];
    const uint qt = tg.x, h = tg.y, hq = h * 128u;
    const uint q0 = qt * 64u + 8u * sg;
    const bool dense = qt < v.p;
    device const uint* list = L + ((ulong)h * v.t + qt) * (v.p + v.k);
    const uint nkb = (dense ? v.t : v.p + v.k) * 2u;
    simdgroup_half8x8 qf[16];
    for (ushort d = 0; d < 16; ++d)
        simdgroup_load(qf[d], Q + (ulong)q0 * v.inner + hq + 8u * d, v.inner);
    simdgroup_float8x8 of[16];
    for (ushort d = 0; d < 16; ++d) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    const short2 fc = qi_fc(lane);
    uint ckey[CH], cpart[CH];
    for (uint c = 0; c < CH; ++c) {
        const uint ch = tid + c * NT;
        ckey[c] = ch / 16u;
        cpart[c] = (ch % 16u) * 8u;
    }
    for (uint kb = 0; kb < nkb; ++kb) {
        const uint kt = dense ? kb / 2u : list[kb / 2u];
        const uint koff = (kb % 2u) * 32u;
        const uint lim = size[kt];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < CH; ++c) {
            const uint row = rowof[kt * 64u + koff + ckey[c]];
            uint4 kq = uint4(0u), vq = uint4(0u);
            if (row != 0xFFFFFFFFu) {
                device const half* kp = P + (ulong)row * v.ldp + v.inner + hq;
                kq = *(device const uint4*)(kp + cpart[c]);
                vq = *(device const uint4*)(kp + v.inner + cpart[c]);
            }
            const uint off = ckey[c] * 128u + cpart[c];
            *(threadgroup uint4*)(sk + off) = kq;
            *(threadgroup uint4*)(sv + off) = vq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[4];
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            s[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, sk + 8u * c * 128u + 8u * d, 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[c], qf[d], kf, s[c]);
            }
        }
        for (ushort c = 0; c < 4; ++c) {
            const uint k0 = koff + 8u * c + (uint)fc.y;
            if (k0 >= lim) s[c].thread_elements()[0] = -INFINITY;
            if (k0 + 1u >= lim) s[c].thread_elements()[1] = -INFINITY;
        }
        float mx = -INFINITY;
        for (ushort c = 0; c < 4; ++c)
            mx = max(mx, max(s[c].thread_elements()[0], s[c].thread_elements()[1]));
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float mn = max(m, mx);
        const float alpha = (m == -INFINITY) ? 0.0f : exp2(m - mn);
        float rsum = 0.0f;
        simdgroup_half8x8 pf[4];
        for (ushort c = 0; c < 4; ++c) {
            float p0 = (mn == -INFINITY) ? 0.0f : exp2(s[c].thread_elements()[0] - mn);
            float p1 = (mn == -INFINITY) ? 0.0f : exp2(s[c].thread_elements()[1] - mn);
            rsum += p0 + p1;
            pf[c].thread_elements()[0] = (half)p0;
            pf[c].thread_elements()[1] = (half)p1;
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * alpha + rsum;
        m = mn;
        if (simd_any(alpha != 1.0f)) {
            for (ushort d = 0; d < 16; ++d) {
                of[d].thread_elements()[0] *= alpha;
                of[d].thread_elements()[1] *= alpha;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, sv + 8u * c * 128u + 8u * d, 128);
                simdgroup_multiply_accumulate(of[d], pf[c], vf, of[d]);
            }
        }
    }
    const uint qs = q0 + (uint)fc.x;
    if (qs >= qt * 64u + size[qt]) return;
    const float inv = 1.0f / l;
    const uint orow = rowof[qs];
    for (ushort d = 0; d < 16; ++d) {
        const uint col = hq + 8u * d + (uint)fc.y;
        *(device half2*)(O + (ulong)orow * v.inner + col) =
            half2(of[d].thread_elements()[0] * inv, of[d].thread_elements()[1] * inv);
    }
}

// the compression branch's tile output: exp2-softmax over a tile-score row
// (log2 units) times the tile means of v; a threadgroup per (tile, head)
kernel void vsa_compress(
    device const float* S [[buffer(0)]],
    device const float* pv [[buffer(1)]],
    device float* oc [[buffer(2)]],
    constant VPd& v [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint d [[thread_index_in_threadgroup]])
{
    threadgroup float red[128];
    const uint i = tg.x, h = tg.y;
    device const float* row = S + ((ulong)h * v.t + i) * v.t;
    float m = -INFINITY;
    for (uint j = d; j < v.t; j += 128u) m = max(m, row[j]);
    red[d] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (d < st) red[d] = max(red[d], red[d + st]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float z = 0.0f;
    for (uint j = d; j < v.t; j += 128u) z += exp2(row[j] - mx);
    red[d] = z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (d < st) red[d] += red[d + st];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];
    float acc = 0.0f;
    for (uint j = 0; j < v.t; ++j) acc += exp2(row[j] - mx) * pv[((ulong)h * v.t + j) * 128u + d];
    oc[((ulong)h * v.t + i) * 128u + d] = acc * inv;
}

// att[r] += gate[r] · oc[tile of r], an element a thread
kernel void vsa_gate_add(
    device half* att [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const float* oc [[buffer(2)]],
    device const uint* slot [[buffer(3)]],
    constant VPd& v [[buffer(4)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= v.n * v.inner) return;
    const uint r = e / v.inner, c = e - r * v.inner;
    const uint h = c / 128u, d = c - h * 128u, t = slot[r] / 64u;
    att[e] = half(float(att[e]) + float(gate[e]) * oc[((ulong)h * v.t + t) * 128u + d]);
}
