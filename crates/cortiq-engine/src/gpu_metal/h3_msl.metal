// MiniMax-H3 stacks on Metal: the kernels the Qwen-Image-2.1 library
// (qi21_msl.metal, compiled into the same library ahead of this file) does
// not have — the row op with an RMS norm and a modulation row per token, the
// q/k norm with the split-half rotation, and the SwiGLU of a `[gate | up]`
// panel. The GEMM (`qi_mm_q4tp`) and the attention (`qi_flash`) are the
// Qwen-Image ones.

// ───────── row op: gated residual + the next modulated RMS norm ─────────
// mode bit 0: x += g ⊙ y        (g = the row's gate chunk under bit 2, else 1;
//             under bit 3 x += s ⊙ (y + b), the bias at g_off and the scale
//             at sh_off of `cw`: a ViT's projection bias and layer scale)
// mode bit 1: o = half((x·inv·w·(1 + scale) + shift)·oscale),
//             inv = 1/sqrt(mean(x²) + eps); scale and shift under bit 2
// A token's chunks start `mrow[row]·6·H` floats past the block's offsets.
struct HRow {
    uint H;
    uint mode;
    float eps;
    float oscale;
    uint g_off;
    uint sh_off;
    uint sc_off;
    uint w_off;
};

kernel void h3_rowop(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device half* o [[buffer(2)]],
    device const float* mods [[buffer(3)]],
    device const uint* mrow [[buffer(4)]],
    device const float* cw [[buffer(5)]],
    constant HRow& p [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    const uint H = p.H, cnt = H / 256u;
    device float* xr = x + (ulong)row * H;
    const bool modded = (p.mode & 4u) != 0u;
    const ulong mo = modded ? (ulong)mrow[row] * 6u * H : 0ul;
    float v[32];
    float ss = 0.0f;
    if ((p.mode & 1u) != 0u) {
        device const float* yr = y + (ulong)row * H;
        device const float* g = mods + p.g_off + mo;
        const bool biased = (p.mode & 8u) != 0u;
        device const float* bb = cw + p.g_off;
        device const float* bs = cw + p.sh_off;
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            float xv;
            if (biased) xv = xr[i] + bs[i] * (yr[i] + bb[i]);
            else xv = xr[i] + (modded ? g[i] : 1.0f) * yr[i];
            xr[i] = xv;
            v[k] = xv;
            ss += xv * xv;
        }
    } else {
        for (uint k = 0; k < cnt; ++k) {
            v[k] = xr[tid + 256u * k];
            ss += v[k] * v[k];
        }
    }
    if ((p.mode & 2u) != 0u) {
        const float inv = rsqrt(qi_block_sum(ss, red, sg, lane) / (float)H + p.eps);
        device const float* w = cw + p.w_off;
        device const float* sc = mods + p.sc_off + mo;
        device const float* sh = mods + p.sh_off + mo;
        const ulong ob = (ulong)row * H;
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            float t = v[k] * inv * w[i];
            if (modded) t = t * (1.0f + sc[i]) + sh[i];
            o[ob + i] = (half)(t * p.oscale);
        }
    }
}

// ───── q/k RMS norm + split-half rotation, in place on the panel ─────
// A simdgroup per (token, head): `nq` query heads from column 0, then `nk`
// key heads from `k_col`. A lane holds four dims; dims j and j + pairs
// (j < pairs, pairs a multiple of 4) turn by the token's angle j, the rest
// only norm. Queries leave multiplied by `qmul` (1/√hd · log2 e).
struct HQk {
    uint ldp;
    uint nq;
    uint nk;
    uint k_col;
    uint pairs;
    uint w_off;
    float eps;
    float qmul;
};

kernel void h3_qkrope(
    device half* P [[buffer(0)]],
    device const float* cw [[buffer(1)]],
    device const float* cs [[buffer(2)]],
    constant HQk& p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x;
    const uint hh = tg.y * 4u + sg;
    if (hh >= p.nq + p.nk) return;
    const bool isk = hh >= p.nq;
    const uint col = isk ? p.k_col + (hh - p.nq) * 128u : hh * 128u;
    device half4* ptr = (device half4*)(P + (ulong)row * p.ldp + col + lane * 4u);
    float4 f = float4(*ptr);
    const float ss = simd_sum(dot(f, f));
    const float inv = rsqrt(ss / 128.0f + p.eps);
    f = f * inv * *(device const float4*)(cw + p.w_off + (isk ? 128u : 0u) + lane * 4u);
    const uint d0 = lane * 4u, np = p.pairs, lp = np / 4u;
    const bool lo = d0 < np, hi = !lo && d0 < 2u * np;
    const ushort other = lo ? (ushort)(lane + lp) : (hi ? (ushort)(lane - lp) : lane);
    const float4 g = simd_shuffle(f, other);
    float4 r = f;
    if (lo || hi) {
        const uint j = lo ? d0 : d0 - np;
        device const float* t = cs + ((ulong)row * np + j) * 2u;
        const float4 c = float4(t[0], t[2], t[4], t[6]);
        const float4 s = float4(t[1], t[3], t[5], t[7]);
        // lo: x[j] = x[j]·c − x[j+np]·s;  hi: x[j+np] = x[j]·s + x[j+np]·c
        r = lo ? f * c - g * s : g * s + f * c;
    }
    if (!isk) r *= p.qmul;
    *ptr = half4(r);
}

// ───────── SwiGLU of a [gate | up] f32 panel: h = silu(g)·u·oscale ─────────
struct HSw {
    uint n4;
    uint ffn;
    float oscale;
    uint has_b;    // add the projection's bias [gate | up] before the nonlinearity
};

kernel void h3_swiglu(
    device const float* gu [[buffer(0)]],
    device half* h [[buffer(1)]],
    constant HSw& p [[buffer(2)]],
    device const float* b [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n4) return;
    const uint f4 = p.ffn / 4u;
    const uint row = i / f4, c4 = i - row * f4;
    device const float* gr = gu + (ulong)row * 2u * p.ffn + 4u * c4;
    float4 g = *(device const float4*)gr;
    float4 u = *(device const float4*)(gr + p.ffn);
    if (p.has_b != 0u) {
        g += *(device const float4*)(b + 4u * c4);
        u += *(device const float4*)(b + p.ffn + 4u * c4);
    }
    const float4 sl = g / (1.0f + exp(-g));
    *(device half4*)(h + (ulong)row * p.ffn + 4u * c4) = half4(sl * u * p.oscale);
}

// ───── MOSS-Image's SiTU of a [hidden | gate] f32 panel, times oscale ─────
// h = 4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25), bounded by 100.
kernel void mo_situ(
    device const float* gu [[buffer(0)]],
    device half* h [[buffer(1)]],
    constant HSw& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n4) return;
    const uint f4 = p.ffn / 4u;
    const uint row = i / f4, c4 = i - row * f4;
    device const float* gr = gu + (ulong)row * 2u * p.ffn + 4u * c4;
    const float4 u = *(device const float4*)gr;
    const float4 g = *(device const float4*)(gr + p.ffn);
    const float4 sg = 1.0f / (1.0f + exp(-clamp(g, -60.0f, 60.0f)));
    const float4 v = 4.0f * tanh(clamp(g * 0.25f, -15.0f, 15.0f)) * sg * 25.0f * tanh(clamp(u * 0.04f, -15.0f, 15.0f));
    *(device half4*)(h + (ulong)row * p.ffn + 4u * c4) = half4(v * p.oscale);
}

// ─────────── flash attention: grouped, causal, batched, hd 64 / 128 ───────────
// `qi_flash`'s schedule for the stacks it does not address: a key head
// shared by `rep` query heads (keys at `k_col`, values at `v_col` of the
// panel), a causal mask (query i sees keys [0, i]), a batch of sequences
// (`nseq` rows apart, one per grid.z) and 64-wide heads. Queries arrive
// normalised, rotated and multiplied by 1/√hd · log2 e.
struct HFa {
    uint nq;       // rows of one sequence (its queries and its keys)
    uint nseq;     // rows between sequences of a batch
    uint ldp;      // panel row stride (halves)
    uint k_col;
    uint v_col;
    uint rep;      // query heads per key head
    uint ldo;      // output row stride (halves)
    float oscale;
    uint causal;
    uint hstride;  // columns between heads: hd, or 3·hd for per-head [q k v] triples
    uint pad1;
    uint pad2;
};

template <ushort HD>
static inline void h3_flash_body(
    device const half* P, device half* O, constant HFa& p,
    threadgroup half* sk, threadgroup half* sv,
    uint tid, ushort sg, ushort lane, uint3 tg)
{
    constexpr ushort NSG = 8;
    constexpr uint NT = 32u * NSG;
    constexpr ushort NF = HD / 8;
    constexpr uint CPR = HD / 8u;              // 8-half chunks a key row
    constexpr uint CH = (32u * CPR) / NT;      // chunks a thread stages a block
    const uint h = tg.y;
    const uint gb = tg.z * p.nseq;
    const uint ntot = p.nq;
    const uint qbase = tg.x * (8u * NSG);
    const uint q0 = qbase + 8u * sg;
    const uint qi = min(q0, p.nq - 8u);
    const uint qr = gb + qi;
    const uint hq = h * p.hstride;
    const uint hk = p.k_col + (h / p.rep) * p.hstride;
    const uint hv = p.v_col + (h / p.rep) * p.hstride;
    // keys this threadgroup must walk: its last query's
    const uint nk = p.causal != 0u ? min(qbase + 8u * NSG, ntot) : ntot;
    simdgroup_half8x8 qf[NF];
    for (ushort d = 0; d < NF; ++d)
        simdgroup_load(qf[d], P + (ulong)qr * p.ldp + hq + 8u * d, p.ldp);
    simdgroup_float8x8 of[NF];
    for (ushort d = 0; d < NF; ++d) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    const short2 fc = qi_fc(lane);
    const uint lim = p.causal != 0u ? min(qi + (uint)fc.x + 1u, ntot) : ntot;
    uint ckey[CH], cpart[CH];
    for (uint c = 0; c < CH; ++c) {
        const uint ch = tid + c * NT;
        ckey[c] = ch / CPR;
        cpart[c] = (ch % CPR) * 8u;
    }
    for (uint kb0 = 0; kb0 < nk; kb0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < CH; ++c) {
            const uint j = min(kb0 + ckey[c], ntot - 1u);
            device const half* row = P + (ulong)(gb + j) * p.ldp;
            const uint off = ckey[c] * HD + cpart[c];
            *(threadgroup uint4*)(sk + off) = *(device const uint4*)(row + hk + cpart[c]);
            *(threadgroup uint4*)(sv + off) = *(device const uint4*)(row + hv + cpart[c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[4];
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            s[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < NF; ++d) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, sk + 8u * c * HD + 8u * d, HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[c], qf[d], kf, s[c]);
            }
        }
        for (ushort c = 0; c < 4; ++c) {
            const uint k0 = kb0 + 8u * c + (uint)fc.y;
            if (k0 >= lim) s[c].thread_elements()[0] = -INFINITY;
            if (k0 + 1u >= lim) s[c].thread_elements()[1] = -INFINITY;
        }
        float mx = -INFINITY;
        for (ushort c = 0; c < 4; ++c)
            mx = max(mx, max(s[c].thread_elements()[0], s[c].thread_elements()[1]));
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float mn = max(m, mx);
        // a row with nothing visible yet keeps m = −inf; exp2 of −inf is 0
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
            for (ushort d = 0; d < NF; ++d) {
                of[d].thread_elements()[0] *= alpha;
                of[d].thread_elements()[1] *= alpha;
            }
        }
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < NF; ++d) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, sv + 8u * c * HD + 8u * d, HD);
                simdgroup_multiply_accumulate(of[d], pf[c], vf, of[d]);
            }
        }
    }
    if (q0 >= p.nq) return;
    const float inv = 1.0f / l;
    const uint orow = qr + (uint)fc.x;
    for (ushort d = 0; d < NF; ++d) {
        const uint col = h * HD + 8u * d + (uint)fc.y;
        half2 v = half2(of[d].thread_elements()[0] * inv * p.oscale,
                        of[d].thread_elements()[1] * inv * p.oscale);
        *(device half2*)(O + (ulong)orow * p.ldo + col) = v;
    }
}

kernel void h3_flash_128(
    device const half* P [[buffer(0)]],
    device half* O [[buffer(1)]],
    constant HFa& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]])
{
    threadgroup half sk[32 * 128];
    threadgroup half sv[32 * 128];
    h3_flash_body<128>(P, O, p, sk, sv, tid, sg, lane, tg);
}

kernel void h3_flash_64(
    device const half* P [[buffer(0)]],
    device half* O [[buffer(1)]],
    constant HFa& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]])
{
    threadgroup half sk[32 * 64];
    threadgroup half sv[32 * 64];
    h3_flash_body<64>(P, O, p, sk, sv, tid, sg, lane, tg);
}

// ───── a ViT's q/k/v step, 64-wide heads, in place on the panel ─────
// The panel is in the projection's own order: per-head [q k v] triples. A
// simdgroup per (token, head, part): the bias (times the panel's guard),
// then for q and k the weightless RMS norm and the split-half rotation of
// the first 2·pairs dims (a rotation table per sequence of `nseq` rows);
// queries leave multiplied by `qmul`. Values only take their bias.
struct HVq {
    uint ldp;
    uint nh;
    uint pairs;
    uint b_off;
    uint nseq;
    float bscale;
    float eps;
    float qmul;
};

kernel void h3_vit_qk(
    device half* P [[buffer(0)]],
    device const float* cw [[buffer(1)]],
    device const float* cs [[buffer(2)]],
    constant HVq& p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = tg.x;
    const uint hh = tg.y * 4u + sg;
    if (hh >= 3u * p.nh) return;
    const uint part = hh % 3u;
    const uint col = hh * 64u;
    device half2* ptr = (device half2*)(P + (ulong)row * p.ldp + col + lane * 2u);
    float2 f = float2(*ptr) + *(device const float2*)(cw + p.b_off + col + lane * 2u) * p.bscale;
    if (part < 2u) {
        const float ss = simd_sum(dot(f, f));
        f *= rsqrt(ss / 64.0f + p.eps);
        const uint d0 = lane * 2u, np = p.pairs, lp = np / 2u;
        const bool lo = d0 < np, hi = !lo && d0 < 2u * np;
        const ushort other = lo ? (ushort)(lane + lp) : (hi ? (ushort)(lane - lp) : lane);
        const float2 g = simd_shuffle(f, other);
        if (lo || hi) {
            const uint j = lo ? d0 : d0 - np;
            device const float* t = cs + ((ulong)(row % p.nseq) * np + j) * 2u;
            const float2 c = float2(t[0], t[2]);
            const float2 s = float2(t[1], t[3]);
            f = lo ? f * c - g * s : g * s + f * c;
        }
        if (part == 0u) f *= p.qmul;
    }
    *ptr = half2(f);
}
