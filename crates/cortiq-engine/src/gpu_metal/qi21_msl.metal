// Qwen-Image-2.1 on Metal: the module-local MSL library, compiled once per
// process by `gpu_metal/qi21.rs` (include_str!).
//
// Conventions (the Z-Image chain's, see zimage_msl.metal):
// - activations feeding a GEMM are half, row-major [token][K], already
//   multiplied by the site's power-of-two guard 2^-s (the GEMM epilogue
//   multiplies it back through `mul`);
// - weights are read in place from the file mapping: q4tp tiles are
//   dequantized to half while staged (nibble − 8 times the group's ladder
//   rung), q8_2f / q8_row int8 are staged times the column field and the
//   row scale is applied in the f32 epilogue;
// - fragment elements are addressed through `qi_fc`, the Apple 8x8 layout.
#include <metal_stdlib>
using namespace metal;

static inline short2 qi_fc(ushort lane) {
    short qid = (short)(lane / 4);
    return short2((qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2);
}

kernel void qi_fragprobe(
    device float* out [[buffer(0)]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float src[64];
    for (ushort i = lane; i < 64; i += 32) src[i] = (float)i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 m;
    simdgroup_load(m, src, 8);
    short2 fc = qi_fc(lane);
    out[lane * 4 + 0] = m.thread_elements()[0];
    out[lane * 4 + 1] = m.thread_elements()[1];
    out[lane * 4 + 2] = (float)(fc.x * 8 + fc.y);
    out[lane * 4 + 3] = (float)(fc.x * 8 + fc.y + 1);
}

// ─────────────────────────── GEMM ───────────────────────────
// Y[t][o] = mul · (rs[o]) · Σ_k X[t][k] · W[o][k]; z selects one of up to
// three same-shape, same-codec weight tensors, its activation slot and its
// output offset.
struct QMm {
    uint n;        // token rows of this dispatch (stores skipped past n)
    uint rows;     // output features (multiple of 64)
    uint K;        // multiple of 32
    uint ldx;      // X row stride (halves)
    uint ldy;      // Y row stride (elements)
    uint epi;      // 0 = f32 store, 1 = half store
    float mul;
    uint params_off;   // q4tp: bytes from the tensor start to the (lo, step) plane
    uint codes_off;    // q4tp: bytes to the 5-bit code plane
    uint code_stride;  // q4tp: code bytes per row
    uint pad0;
    uint pad1;
    uint x_off[4];
    uint y_off[4];
};

static inline float qi_f16(device const uchar* p) {
    const ushort v = (ushort)((uint)p[0] | ((uint)p[1] << 8));
    return (float)as_type<half>(v);
}

// Staging of one thread's 16 weights (row r, k in [kh, kh+16) of the
// current 32-wide k step) into `hv`.
// FMT 0: q4tp (w = the tensor start); FMT 1: q8 int8 with the column
// field `col` (w = the int8 plane).
template <ushort FMT>
struct QiW {
    device const uchar* w;
    uint row;
    uint K;
    float lo, st;              // q4tp row ladder (log2 domain)
    device const uchar* codes; // q4tp row codes
    device const float* col;   // q8: column field
    uint2 nb;                  // prefetched q4tp nibble bytes
    uint4 qb;                  // prefetched int8
    uint code;                 // prefetched q4tp code

    void init(device const uchar* base, uint r, uint K_, constant QMm& p, device const float* colf, uint kh) {
        w = base;
        row = r;
        K = K_;
        col = colf;
        if (FMT == 0) {
            device const uchar* pr = base + p.params_off + (ulong)r * 4u;
            lo = qi_f16(pr);
            st = qi_f16(pr + 2);
            codes = base + p.codes_off + (ulong)r * p.code_stride;
        }
        fetch(0u, kh);
    }
    void fetch(uint k0, uint kh) {
        if (FMT == 0) {
            const uint g = k0 / 32u;
            const ulong gpr = (ulong)(K / 32u);
            nb = *(device const uint2*)(w + ((ulong)row * gpr + g) * 16u + kh / 2u);
            const uint bit = g * 5u, b = bit >> 3, sh = bit & 7u;
            uint v = (uint)codes[b];
            if (sh > 3u) v |= (uint)codes[b + 1] << 8;
            code = (v >> sh) & 31u;
        } else {
            qb = *(device const uint4*)(w + (ulong)row * K + k0 + kh);
        }
    }
    void stage(thread half* hv, uint k0, uint kh) {
        if (FMT == 0) {
            const float s = exp2(lo + (float)code * st);
            for (uint i = 0; i < 4u; ++i) {
                const uint byte0 = (nb.x >> (8u * i)) & 0xFFu;
                const uint byte1 = (nb.y >> (8u * i)) & 0xFFu;
                hv[2u * i] = (half)(((float)(byte0 & 15u) - 8.0f) * s);
                hv[2u * i + 1u] = (half)(((float)(byte0 >> 4) - 8.0f) * s);
                hv[8u + 2u * i] = (half)(((float)(byte1 & 15u) - 8.0f) * s);
                hv[8u + 2u * i + 1u] = (half)(((float)(byte1 >> 4) - 8.0f) * s);
            }
        } else {
            char4 c0 = as_type<char4>(qb.x), c1 = as_type<char4>(qb.y);
            char4 c2 = as_type<char4>(qb.z), c3 = as_type<char4>(qb.w);
            device const float4* cf = (device const float4*)(col + k0 + kh);
            const float4 f0 = cf[0], f1 = cf[1], f2 = cf[2], f3 = cf[3];
            hv[0] = (half)((float)c0.x * f0.x); hv[1] = (half)((float)c0.y * f0.y);
            hv[2] = (half)((float)c0.z * f0.z); hv[3] = (half)((float)c0.w * f0.w);
            hv[4] = (half)((float)c1.x * f1.x); hv[5] = (half)((float)c1.y * f1.y);
            hv[6] = (half)((float)c1.z * f1.z); hv[7] = (half)((float)c1.w * f1.w);
            hv[8] = (half)((float)c2.x * f2.x); hv[9] = (half)((float)c2.y * f2.y);
            hv[10] = (half)((float)c2.z * f2.z); hv[11] = (half)((float)c2.w * f2.w);
            hv[12] = (half)((float)c3.x * f3.x); hv[13] = (half)((float)c3.y * f3.y);
            hv[14] = (half)((float)c3.z * f3.z); hv[15] = (half)((float)c3.w * f3.w);
        }
    }
};

// 64 features × 64 tokens × 32 k per 128-thread group; 2×2 simdgroups,
// each 32 tokens × 32 features (4×4 fragments); operand tiles packed as
// dense 8×8 blocks (the Z-Image `zi_q8mm` schedule).
template <ushort FMT>
static inline void qi_mm_body(
    device const uchar* W, device const float* rs, device const float* colf,
    device const half* X, device float* Y, constant QMm& p, uint z,
    threadgroup half* sw, threadgroup half* sx,
    uint tid, ushort sg, ushort lane, uint2 tg)
{
    const uint t0 = tg.x * 64u, o0 = tg.y * 64u, K = p.K;
    const uint r = tid >> 1, kh = (tid & 1u) * 16u;
    QiW<FMT> wt;
    wt.init(W, o0 + r, K, p, colf, kh);
    device const uint4* xp = (device const uint4*)(X + p.x_off[z] + (ulong)(t0 + r) * p.ldx + kh);
    const ushort sgo = sg & 1, sgt = sg >> 1;
    simdgroup_float8x8 acc[4][4];
    for (ushort i = 0; i < 4; ++i)
        for (ushort j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint4 xa = xp[0], xb = xp[1];
    threadgroup uint4* xd = (threadgroup uint4*)(sx + ((r / 8u) * 4u + kh / 8u) * 64u + (r % 8u) * 8u);
    const uint ob = r / 8u, oi = r % 8u;
    for (uint k0 = 0; k0 < K; k0 += 32u) {
        half hv[16];
        wt.stage(hv, k0, kh);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16u; ++i) {
            uint kk = kh + i;
            sw[((kk / 8u) * 8u + ob) * 64u + (kk % 8u) * 8u + oi] = hv[i];
        }
        xd[0] = xa;
        xd[8] = xb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 32u < K) {
            wt.fetch(k0 + 32u, kh);
            xp += 4;
            xa = xp[0];
            xb = xp[1];
        }
        #pragma clang loop unroll(full)
        for (ushort kb = 0; kb < 4; ++kb) {
            simdgroup_half8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                simdgroup_load(a[i], sx + ((4u * sgt + i) * 4u + kb) * 64u, 8);
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < 4; ++j)
                simdgroup_load(b[j], sw + (kb * 8u + 4u * sgo + j) * 64u, 8);
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
    if (t0 + 32u * sgt >= p.n) return;
    const short2 fc = qi_fc(lane);
    for (ushort j = 0; j < 4; ++j) {
        const uint o = o0 + 32u * sgo + 8u * j + (uint)fc.y;
        const float s0 = (FMT == 0 ? 1.0f : rs[o]) * p.mul;
        const float s1 = (FMT == 0 ? 1.0f : rs[o + 1]) * p.mul;
        for (ushort i = 0; i < 4; ++i) {
            const uint t = t0 + 32u * sgt + 8u * i + (uint)fc.x;
            if (t >= p.n) continue;
            const ulong yi = (ulong)p.y_off[z] + (ulong)t * p.ldy + o;
            float v0 = acc[i][j].thread_elements()[0] * s0;
            float v1 = acc[i][j].thread_elements()[1] * s1;
            if (p.epi == 0u) {
                *(device float2*)(Y + yi) = float2(v0, v1);
            } else {
                *(device half2*)((device half*)Y + yi) = half2(v0, v1);
            }
        }
    }
}

#define QI_MM_ARGS \
    device const uchar* w0 [[buffer(0)]], \
    device const uchar* w1 [[buffer(1)]], \
    device const uchar* w2 [[buffer(2)]], \
    device const float* rs0 [[buffer(3)]], \
    device const float* rs1 [[buffer(4)]], \
    device const float* rs2 [[buffer(5)]], \
    device const half* X [[buffer(6)]], \
    device float* Y [[buffer(7)]], \
    constant QMm& p [[buffer(8)]], \
    device const float* cf0 [[buffer(9)]], \
    device const float* cf1 [[buffer(10)]], \
    device const float* cf2 [[buffer(11)]], \
    uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], \
    uint3 tg [[threadgroup_position_in_grid]]

kernel void qi_mm_q4tp(QI_MM_ARGS)
{
    threadgroup half sw[2048];
    threadgroup half sx[2048];
    const uint z = tg.z;
    device const uchar* W = z == 0u ? w0 : (z == 1u ? w1 : w2);
    qi_mm_body<0>(W, rs0, cf0, X, Y, p, z, sw, sx, tid, sg, lane, tg.xy);
}

kernel void qi_mm_q8(QI_MM_ARGS)
{
    threadgroup half sw[2048];
    threadgroup half sx[2048];
    const uint z = tg.z;
    device const uchar* W = z == 0u ? w0 : (z == 1u ? w1 : w2);
    device const float* rs = z == 0u ? rs0 : (z == 1u ? rs1 : rs2);
    device const float* cf = z == 0u ? cf0 : (z == 1u ? cf1 : cf2);
    qi_mm_body<1>(W, rs, cf, X, Y, p, z, sw, sx, tid, sg, lane, tg.xy);
}

// ───────────────────── flash attention (hd 128) ─────────────────────
// Queries: panel rows [q_row0, q_row0 + nq) (q normalised, roped and
// pre-multiplied by 1/√hd·log2 e). Keys j < n_pre come from the cached
// prefix (`pkv` [n_pre][2H]: K then V), keys j ≥ n_pre from panel rows
// q_row0 + (j − n_pre). With `use_vis`, query i sees only keys
// [0, vis[i]) (the prefill's block-causal mask; `vis` non-decreasing).
struct QFa {
    uint q_row0;
    uint nq;
    uint n_pre;
    uint n_own;
    uint ldp;
    uint H;
    uint ldo;
    float oscale;
    uint use_vis;
    uint pad0;
    uint pad1;
    uint pad2;
};

static inline void qi_kv_ptr(constant QFa& p, device const half* P, device const half* pkv, uint j, uint hq,
                             thread device const half*& kp, thread device const half*& vp) {
    if (j < p.n_pre) {
        kp = pkv + (ulong)j * (2u * p.H) + hq;
        vp = kp + p.H;
    } else {
        kp = P + (ulong)(p.q_row0 + j - p.n_pre) * p.ldp + p.H + hq;
        vp = kp + p.H;
    }
}

kernel void qi_flash(
    device const half* P [[buffer(0)]],
    device half* O [[buffer(1)]],
    device const half* pkv [[buffer(2)]],
    device const uint* vis [[buffer(3)]],
    constant QFa& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    constexpr ushort NSG = 8;
    constexpr uint NT = 32u * NSG, CH = 512u / NT;
    threadgroup half sk[32 * 128];
    threadgroup half sv[32 * 128];
    const uint h = tg.y;
    const uint ntot = p.n_pre + p.n_own;
    const uint qbase = tg.x * (8u * NSG);
    const uint q0 = qbase + 8u * sg;
    const uint qi = min(q0, p.nq - 8u);
    const uint qr = p.q_row0 + qi;
    const uint hq = h * 128u;
    // keys this threadgroup must walk
    uint nk = ntot;
    if (p.use_vis != 0u) nk = vis[min(qbase + 8u * NSG, p.nq) - 1u];
    simdgroup_half8x8 qf[16];
    for (ushort d = 0; d < 16; ++d)
        simdgroup_load(qf[d], P + (ulong)qr * p.ldp + hq + 8u * d, p.ldp);
    simdgroup_float8x8 of[16];
    for (ushort d = 0; d < 16; ++d) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    const short2 fc = qi_fc(lane);
    const uint my_vis = p.use_vis != 0u ? vis[qi + (uint)fc.x] : ntot;
    uint ckey[CH], cpart[CH];
    uint4 kq[CH], vq[CH];
    for (uint c = 0; c < CH; ++c) {
        const uint ch = tid + c * NT;
        ckey[c] = ch / 16u;
        cpart[c] = (ch % 16u) * 8u;
    }
    for (uint kb0 = 0; kb0 < nk; kb0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < CH; ++c) {
            const uint j = min(kb0 + ckey[c], ntot - 1u);
            device const half* kp;
            device const half* vp;
            qi_kv_ptr(p, P, pkv, j, hq, kp, vp);
            kq[c] = *(device const uint4*)(kp + cpart[c]);
            vq[c] = *(device const uint4*)(vp + cpart[c]);
            const uint off = ckey[c] * 128u + cpart[c];
            *(threadgroup uint4*)(sk + off) = kq[c];
            *(threadgroup uint4*)(sv + off) = vq[c];
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
        // mask keys past this query's visibility (and past the end)
        const uint lim = min(my_vis, ntot);
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
    if (q0 >= p.nq) return;
    const float inv = 1.0f / l;
    const uint orow = qr + (uint)fc.x;
    for (ushort d = 0; d < 16; ++d) {
        const uint col = hq + 8u * d + (uint)fc.y;
        half2 v = half2(of[d].thread_elements()[0] * inv * p.oscale,
                        of[d].thread_elements()[1] * inv * p.oscale);
        *(device half2*)(O + (ulong)orow * p.ldo + col) = v;
    }
}

// ───────── qk RMSNorm + interleaved RoPE, in place on the panel ─────────
struct QQk {
    uint H;
    uint nh;
    uint ldp;
    uint row0;
    uint rope0;   // first rope-table row of this dispatch
    float eps;
    float qmul;
    uint pad0;
};

kernel void qi_qkrope(
    device half* P [[buffer(0)]],
    device const float* nq [[buffer(1)]],
    device const float* nk [[buffer(2)]],
    device const float* cs [[buffer(3)]],
    device const float* sn [[buffer(4)]],
    constant QQk& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = p.row0 + tg.x;
    const uint rr = p.rope0 + tg.x;
    const uint hh = tg.y * 4u + sg;
    if (hh >= 2u * p.nh) return;
    const bool isk = hh >= p.nh;
    const uint head = isk ? hh - p.nh : hh;
    device half4* ptr = (device half4*)(P + (ulong)row * p.ldp + (isk ? p.H : 0u) + head * 128u + lane * 4u);
    float4 f = float4(*ptr);
    const float ss = simd_sum(dot(f, f));
    const float inv = rsqrt(ss / 128.0f + p.eps);
    device const float* w = isk ? nk : nq;
    f = f * inv * *(device const float4*)(w + lane * 4u);
    const uint j = lane * 2u;
    const float2 c = *(device const float2*)(cs + (ulong)rr * 64u + j);
    const float2 s = *(device const float2*)(sn + (ulong)rr * 64u + j);
    float4 r;
    r.x = f.x * c.x - f.y * s.x;
    r.y = f.x * s.x + f.y * c.x;
    r.z = f.z * c.y - f.w * s.y;
    r.w = f.z * s.y + f.w * c.y;
    if (!isk) r *= p.qmul;
    *ptr = half4(r);
}

// ───────────── row op: gated residual + next modulated LayerNorm ─────────────
// mode bit 0: x += tanh(g) ⊙ y
// mode bit 1: o = half(LN(x) · (1 + s) · oscale)      (LN: no affine)
struct QRow {
    uint H;
    uint row0;
    uint mode;
    float eps;
    float oscale;
    uint pad0;
    uint pad1;
    uint pad2;
};

static inline float qi_block_sum(float v, threadgroup float* red, ushort sg, ushort lane) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (ushort i = 0; i < 8; ++i) t += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void qi_rowop(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device half* o [[buffer(2)]],
    device const float* g [[buffer(3)]],
    device const float* s [[buffer(4)]],
    constant QRow& p [[buffer(5)]],
    uint tgi [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    const uint row = p.row0 + tgi;
    const uint H = p.H, cnt = H / 256u;
    device float* xr = x + (ulong)row * H;
    float v[16];
    if ((p.mode & 1u) != 0u) {
        device const float* yr = y + (ulong)row * H;
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            const float xv = xr[i] + precise::tanh(g[i]) * yr[i];
            xr[i] = xv;
            v[k] = xv;
        }
    } else {
        for (uint k = 0; k < cnt; ++k) v[k] = xr[tid + 256u * k];
    }
    if ((p.mode & 2u) != 0u) {
        float sm = 0.0f;
        for (uint k = 0; k < cnt; ++k) sm += v[k];
        const float mean = qi_block_sum(sm, red, sg, lane) / (float)H;
        float q = 0.0f;
        for (uint k = 0; k < cnt; ++k) { const float d = v[k] - mean; q += d * d; }
        const float inv = rsqrt(qi_block_sum(q, red, sg, lane) / (float)H + p.eps);
        const ulong ob = (ulong)row * H;
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            o[ob + i] = (half)((v[k] - mean) * inv * (1.0f + s[i]) * p.oscale);
        }
    }
}

// ───────────── SwiGLU: h = silu(g)·u·oscale → half ─────────────
struct QSw {
    uint n4;
    uint g_off;   // floats
    uint u_off;   // floats
    uint h_off;   // halves
    float oscale;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void qi_swiglu(
    device const float* gu [[buffer(0)]],
    device half* h [[buffer(1)]],
    constant QSw& p [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n4) return;
    const float4 g = *(device const float4*)(gu + p.g_off + 4u * i);
    const float4 u = *(device const float4*)(gu + p.u_off + 4u * i);
    const float4 sl = g / (1.0f + exp(-g));
    *(device half4*)(h + p.h_off + 4u * i) = half4(sl * u * p.oscale);
}

// ───────────── img_in (64 → H, no bias) ─────────────
struct QEm {
    uint H;
    uint n;
    uint pad0;
    uint pad1;
};

kernel void qi_embed(
    device const float* xt [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant QEm& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, t = gid.y;
    if (c >= p.H || t >= p.n) return;
    device const float4* w4 = (device const float4*)(W + (ulong)c * 64u);
    device const float4* x4 = (device const float4*)(xt + (ulong)t * 64u);
    float acc = 0.0f;
    for (ushort k = 0; k < 16; ++k) acc += dot(w4[k], x4[k]);
    x[(ulong)t * p.H + c] = acc;
}

// ─────── final: LN(x)·fs → Linear(H → 64), no bias ───────
struct QFi {
    uint H;
    uint row0;
    float eps;
    uint pad0;
};

kernel void qi_final(
    device const float* x [[buffer(0)]],
    device const float* fs [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant QFi& p [[buffer(4)]],
    uint tgi [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float yr[4096];
    threadgroup float red[8];
    const uint H = p.H, cnt = H / 256u;
    device const float* xr = x + (ulong)(p.row0 + tgi) * H;
    float v[16];
    float s = 0.0f;
    for (uint k = 0; k < cnt; ++k) { v[k] = xr[tid + 256u * k]; s += v[k]; }
    const float mean = qi_block_sum(s, red, sg, lane) / (float)H;
    float q = 0.0f;
    for (uint k = 0; k < cnt; ++k) { float d = v[k] - mean; q += d * d; }
    const float inv = rsqrt(qi_block_sum(q, red, sg, lane) / (float)H + p.eps);
    for (uint k = 0; k < cnt; ++k) {
        const uint i = tid + 256u * k;
        yr[i] = (v[k] - mean) * inv * fs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = sg * 8u; o < sg * 8u + 8u; ++o) {
        device const float* wr = W + (ulong)o * H;
        float acc = 0.0f;
        for (uint i = lane; i < H; i += 32u) acc += yr[i] * wr[i];
        acc = simd_sum(acc);
        if (lane == 0) out[(ulong)tgi * 64u + o] = acc;
    }
}

// ───────────── prefix K/V: panel rows → the per-layer cache ─────────────
struct QKv {
    uint H;
    uint row0;
    uint n;
    uint ldp;
};

kernel void qi_kvcopy(
    device const half* P [[buffer(0)]],
    device half* kv [[buffer(1)]],
    constant QKv& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c8 = gid.x, t = gid.y;   // 8 halves per thread, 2H wide
    if (t >= p.n || c8 * 8u >= 2u * p.H) return;
    const uint4 v = *(device const uint4*)(P + (ulong)(p.row0 + t) * p.ldp + p.H + c8 * 8u);
    *(device uint4*)(kv + (ulong)t * 2u * p.H + c8 * 8u) = v;
}

// ───────────── |x| max of a half range into slot (diagnostics) ─────────────
kernel void qi_amax(
    device const half* x [[buffer(0)]],
    device atomic_uint* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& slot [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint gsz [[threads_per_grid]])
{
    float mx = 0.0f;
    for (uint i = gid; i < n; i += gsz) mx = max(mx, fabs((float)x[i]));
    mx = simd_max(mx);
    atomic_fetch_max_explicit(&out[slot], as_type<uint>(mx), memory_order_relaxed);
}
