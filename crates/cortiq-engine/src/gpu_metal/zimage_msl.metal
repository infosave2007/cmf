// Z-Image on Metal (plan WP3): the module-local MSL library, compiled
// once per process by `gpu_metal/zimage.rs` (included with include_str!).
//
// Conventions:
// - activations feeding a GEMM are half, row-major [token][K], already
//   multiplied by the consumer tensor's q8_2f column field and by the
//   site's power-of-two guard 2^-s (the GEMM epilogue multiplies back);
// - weights are the file's q8_2f/q8_row int8 bytes read in place (no
//   planes): int8 is exact in half, the per-row scale is applied in the
//   f32 epilogue;
// - every simdgroup fragment element is addressed through `zi_fc`, the
//   Apple GPU 8x8 fragment layout (MLX BaseMMAFrag), checked at start-up
//   by `zi_fragprobe`.
#include <metal_stdlib>
using namespace metal;

// (row, col) of this lane's two consecutive elements in an 8x8 fragment.
static inline short2 zi_fc(ushort lane) {
    short qid = (short)(lane / 4);
    return short2((qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2);
}

kernel void zi_fragprobe(
    device float* out [[buffer(0)]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float src[64];
    for (ushort i = lane; i < 64; i += 32) src[i] = (float)i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 m;
    simdgroup_load(m, src, 8);
    short2 fc = zi_fc(lane);
    out[lane * 4 + 0] = m.thread_elements()[0];
    out[lane * 4 + 1] = m.thread_elements()[1];
    out[lane * 4 + 2] = (float)(fc.x * 8 + fc.y);
    out[lane * 4 + 3] = (float)(fc.x * 8 + fc.y + 1);
}

// ─────────────────────────── GEMM ───────────────────────────
// Y[t][o] = mul · rs[o] · Σ_k X[t][k] · Q[o][k]    (z selects one of up to
// three weight tensors, its activation slot and its output offset)
struct ZMm {
    uint n;       // token rows of this dispatch (stores skipped past n)
    uint rows;    // output features (multiple of 64)
    uint K;       // multiple of 32
    uint ldx;     // X row stride (halves)
    uint ldy;     // Y row stride (elements)
    uint epi;     // 0 = f32 store, 1 = half store
    float mul;
    uint pad0;
    uint x_off[4];   // per z, elements
    uint y_off[4];   // per z, elements
};

// 64 features × 64 tokens × 32 k per 128-thread group; 2×2 simdgroups,
// each 32 tokens × 32 features (4×4 fragments). Both operand tiles are
// packed as dense 8×8 blocks. WT: the W tile is stored [o][k] (vector
// stores) and loaded transposed; otherwise it is scattered into [k][o].
template <bool WT>
static inline void zi_mm_body(
    device const char* W, device const float* rs, device const half* X,
    device float* Y, constant ZMm& p, uint z,
    threadgroup half* sw, threadgroup half* sx,
    uint tid, ushort sg, ushort lane, uint2 tg)
{
    const uint t0 = tg.x * 64u, o0 = tg.y * 64u, K = p.K;
    const uint r = tid >> 1, kh = (tid & 1u) * 16u;
    device const uint4* wp = (device const uint4*)(W + (ulong)(o0 + r) * K + kh);
    device const uint4* xp = (device const uint4*)(X + p.x_off[z] + (ulong)(t0 + r) * p.ldx + kh);
    const ushort sgo = sg & 1, sgt = sg >> 1;
    simdgroup_float8x8 acc[4][4];
    for (ushort i = 0; i < 4; ++i)
        for (ushort j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint4 wq = wp[0];
    uint4 xa = xp[0], xb = xp[1];
    // staging addresses (block-packed)
    threadgroup uint4* xd = (threadgroup uint4*)(sx + ((r / 8u) * 4u + kh / 8u) * 64u + (r % 8u) * 8u);
    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            half4 h0 = half4(as_type<char4>(wq.x));
            half4 h1 = half4(as_type<char4>(wq.y));
            half4 h2 = half4(as_type<char4>(wq.z));
            half4 h3 = half4(as_type<char4>(wq.w));
            if (WT) {
                threadgroup half4* d0 = (threadgroup half4*)(sw + ((r / 8u) * 4u + kh / 8u) * 64u + (r % 8u) * 8u);
                d0[0] = h0; d0[1] = h1;
                d0[16] = h2; d0[17] = h3;   // next k-block: +64 halves = +16 half4
            } else {
                const uint ob = r / 8u, oi = r % 8u;
                half hv[16] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w,
                               h2.x, h2.y, h2.z, h2.w, h3.x, h3.y, h3.z, h3.w};
                for (uint i = 0; i < 16u; ++i) {
                    uint kk = kh + i;
                    sw[((kk / 8u) * 8u + ob) * 64u + (kk % 8u) * 8u + oi] = hv[i];
                }
            }
            xd[0] = xa;
            xd[8] = xb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 32u < K) {
            wp += 2;
            xp += 4;
            wq = wp[0];
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
            for (ushort j = 0; j < 4; ++j) {
                if (WT)
                    simdgroup_load(b[j], sw + ((4u * sgo + j) * 4u + kb) * 64u, 8, ulong2(0, 0), true);
                else
                    simdgroup_load(b[j], sw + (kb * 8u + 4u * sgo + j) * 64u, 8);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
    if (t0 + 32u * sgt >= p.n) return;
    const short2 fc = zi_fc(lane);
    for (ushort j = 0; j < 4; ++j) {
        const uint o = o0 + 32u * sgo + 8u * j + (uint)fc.y;
        const float s0 = rs[o] * p.mul, s1 = rs[o + 1] * p.mul;
        for (ushort i = 0; i < 4; ++i) {
            const uint t = t0 + 32u * sgt + 8u * i + (uint)fc.x;
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

#define ZI_MM_ARGS \
    device const char* w0 [[buffer(0)]], \
    device const char* w1 [[buffer(1)]], \
    device const char* w2 [[buffer(2)]], \
    device const float* rs0 [[buffer(3)]], \
    device const float* rs1 [[buffer(4)]], \
    device const float* rs2 [[buffer(5)]], \
    device const half* X [[buffer(6)]], \
    device float* Y [[buffer(7)]], \
    constant ZMm& p [[buffer(8)]], \
    uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], \
    uint3 tg [[threadgroup_position_in_grid]]

kernel void zi_q8mm(ZI_MM_ARGS)
{
    threadgroup half sw[2048];
    threadgroup half sx[2048];
    const uint z = tg.z;
    device const char* W = z == 0u ? w0 : (z == 1u ? w1 : w2);
    device const float* rs = z == 0u ? rs0 : (z == 1u ? rs1 : rs2);
    zi_mm_body<false>(W, rs, X, Y, p, z, sw, sx, tid, sg, lane, tg.xy);
}

kernel void zi_q8mm_wt(ZI_MM_ARGS)
{
    threadgroup half sw[2048];
    threadgroup half sx[2048];
    const uint z = tg.z;
    device const char* W = z == 0u ? w0 : (z == 1u ? w1 : w2);
    device const float* rs = z == 0u ? rs0 : (z == 1u ? rs1 : rs2);
    zi_mm_body<true>(W, rs, X, Y, p, z, sw, sx, tid, sg, lane, tg.xy);
}

// ───────────────────── flash attention (hd 128) ─────────────────────
// The fused qkv panel [row][3H] half (q, k normalised+roped, q already
// carries 1/√hd·log2 e); one item's keys are its image rows then its
// caption rows (two row ranges of the panel). Output [row][H] half,
// multiplied by the O projection's column field and `oscale`.
struct ZFa {
    uint img_off;
    uint n_img;
    uint cap_off;
    uint n_cap;
    uint ldp;
    uint H;
    uint ldo;
    float oscale;
};

static inline uint zfa_row(constant ZFa& p, uint j) {
    return j < p.n_img ? p.img_off + j : p.cap_off + (j - p.n_img);
}

// Flash body. NSG simdgroups × 8 queries per group (32 or 64 queries);
// keys in blocks of 32, K/V staged row-major in threadgroup memory (Kᵀ
// fragments load transposed). PF: the next block's K/V are prefetched
// into registers before this block's math. Measured (M4, n 1056/4224,
// 1.34-1.51 TF/s, all within noise of each other): the Q·Kᵀ loop order
// and a transposed K staging (`[d][key]`) — kept out.
template <ushort NSG, bool PF, ushort SKIP>
static inline void zi_flash_body(
    device const half* P, device half* O, device const float* colo, constant ZFa& p,
    threadgroup half* sk, threadgroup half* sv,
    uint tid, ushort sg, ushort lane, uint2 tg)
{
    constexpr uint NT = 32u * NSG, CH = 512u / NT;   // uint4 chunks of K (and of V) per thread
    const uint h = tg.y;
    const uint ntot = p.n_img + p.n_cap;
    const uint q0 = tg.x * (8u * NSG) + 8u * sg;
    const uint qr = zfa_row(p, q0);
    const uint hq = h * 128u;
    simdgroup_half8x8 qf[16];
    for (ushort d = 0; d < 16; ++d)
        simdgroup_load(qf[d], P + (ulong)qr * p.ldp + hq + 8u * d, p.ldp);
    simdgroup_float8x8 of[16];
    for (ushort d = 0; d < 16; ++d) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    const short2 fc = zi_fc(lane);
    uint4 kq[CH], vq[CH];
    uint ckey[CH], cpart[CH];
    for (uint c = 0; c < CH; ++c) {
        const uint ch = tid + c * NT;
        ckey[c] = ch / 16u;
        cpart[c] = (ch % 16u) * 8u;
    }
    if (PF) {
        for (uint c = 0; c < CH; ++c) {
            const uint kr = zfa_row(p, ckey[c]);
            kq[c] = *(device const uint4*)(P + (ulong)kr * p.ldp + p.H + hq + cpart[c]);
            vq[c] = *(device const uint4*)(P + (ulong)kr * p.ldp + 2u * p.H + hq + cpart[c]);
        }
    }
    for (uint kb0 = 0; kb0 < ntot; kb0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!PF) {
            for (uint c = 0; c < CH; ++c) {
                const uint kr = zfa_row(p, kb0 + ckey[c]);
                kq[c] = *(device const uint4*)(P + (ulong)kr * p.ldp + p.H + hq + cpart[c]);
                vq[c] = *(device const uint4*)(P + (ulong)kr * p.ldp + 2u * p.H + hq + cpart[c]);
            }
        }
        for (uint c = 0; c < CH; ++c) {
            *(threadgroup uint4*)(sk + ckey[c] * 128u + cpart[c]) = kq[c];
            *(threadgroup uint4*)(sv + ckey[c] * 128u + cpart[c]) = vq[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && kb0 + 32u < ntot) {
            for (uint c = 0; c < CH; ++c) {
                const uint kr = zfa_row(p, kb0 + 32u + ckey[c]);
                kq[c] = *(device const uint4*)(P + (ulong)kr * p.ldp + p.H + hq + cpart[c]);
                vq[c] = *(device const uint4*)(P + (ulong)kr * p.ldp + 2u * p.H + hq + cpart[c]);
            }
        }
        simdgroup_float8x8 s[4];
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            s[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            if (SKIP != 2) {
                #pragma clang loop unroll(full)
                for (ushort d = 0; d < 16; ++d) {
                    simdgroup_half8x8 kf;
                    simdgroup_load(kf, sk + 8u * c * 128u + 8u * d, 128, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[c], qf[d], kf, s[c]);
                }
            }
        }
        float mx = -INFINITY;
        for (ushort c = 0; c < 4; ++c)
            mx = max(mx, max(s[c].thread_elements()[0], s[c].thread_elements()[1]));
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float mn = max(m, mx);
        const float alpha = exp2(m - mn);
        float rsum = 0.0f;
        simdgroup_half8x8 pf[4];
        for (ushort c = 0; c < 4; ++c) {
            float p0 = exp2(s[c].thread_elements()[0] - mn);
            float p1 = exp2(s[c].thread_elements()[1] - mn);
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
        if (SKIP != 1) {
            #pragma clang loop unroll(full)
            for (ushort c = 0; c < 4; ++c) {
                #pragma clang loop unroll(full)
                for (ushort d = 0; d < 16; ++d) {
                    simdgroup_half8x8 vf;
                    simdgroup_load(vf, sv + 8u * c * 128u + 8u * d, 128);
                    simdgroup_multiply_accumulate(of[d], pf[c], vf, of[d]);
                }
            }
        } else {
            for (ushort d = 0; d < 16; ++d) of[d].thread_elements()[0] += (float)pf[d % 4].thread_elements()[0];
        }
    }
    if (q0 >= ntot) return;   // the tail group of a 64-query tile
    const float inv = 1.0f / l;
    const uint orow = qr + (uint)fc.x;
    for (ushort d = 0; d < 16; ++d) {
        const uint col = hq + 8u * d + (uint)fc.y;
        half2 v = half2(of[d].thread_elements()[0] * inv * colo[col] * p.oscale,
                        of[d].thread_elements()[1] * inv * colo[col + 1] * p.oscale);
        *(device half2*)(O + (ulong)orow * p.ldo + col) = v;
    }
}

#define ZI_FLASH(NAME, NSG, PF, SKIP) \
kernel void NAME( \
    device const half* P [[buffer(0)]], \
    device half* O [[buffer(1)]], \
    device const float* colo [[buffer(2)]], \
    constant ZFa& p [[buffer(3)]], \
    uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], \
    uint2 tg [[threadgroup_position_in_grid]]) \
{ \
    threadgroup half sk[32 * 128]; \
    threadgroup half sv[32 * 128]; \
    zi_flash_body<NSG, PF, SKIP>(P, O, colo, p, sk, sv, tid, sg, lane, tg); \
}

ZI_FLASH(zi_flash_q64pf, 8, true, 0)
ZI_FLASH(zi_flash_q32, 4, false, 0)
ZI_FLASH(zi_flash_nopv, 8, true, 1)
ZI_FLASH(zi_flash_noqk, 8, true, 2)


// ───────── qk RMSNorm + interleaved RoPE, in place on the panel ─────────
struct ZQk {
    uint H;
    uint nh;
    uint ldp;
    uint row0;
    float eps;
    float qmul;
};

kernel void zi_qkrope(
    device half* P [[buffer(0)]],
    device const float* nq [[buffer(1)]],
    device const float* nk [[buffer(2)]],
    device const float* cs [[buffer(3)]],
    device const float* sn [[buffer(4)]],
    constant ZQk& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = p.row0 + tg.x;
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
    const float2 c = *(device const float2*)(cs + (ulong)row * 64u + j);
    const float2 s = *(device const float2*)(sn + (ulong)row * 64u + j);
    float4 r;
    r.x = f.x * c.x - f.y * s.x;
    r.y = f.x * s.x + f.y * c.x;
    r.z = f.z * c.y - f.w * s.y;
    r.w = f.z * s.y + f.w * c.y;
    if (!isk) r *= p.qmul;
    *ptr = half4(r);
}

// ───────────── row op: gated post-norm residual + next pre-norm ─────────────
// mode bit 0: x += (has_g ? tanh(g) : 1) ⊙ rms(y)·wpost
// mode bit 1: o_j = half(rms(x)·wpre·(has_s ? 1+s : 1) · c_j · oscale), j < nout
struct ZRow {
    uint H;
    uint row0;
    uint mode;
    uint nout;
    uint has_g;
    uint has_s;
    float eps1;
    float eps2;
    float oscale;
    uint pad0;
};

static inline float zi_block_sum(float v, threadgroup float* red, ushort sg, ushort lane) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (ushort i = 0; i < 8; ++i) t += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void zi_rowop(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device half* o0 [[buffer(2)]],
    device half* o1 [[buffer(3)]],
    device half* o2 [[buffer(4)]],
    device const float* wpost [[buffer(5)]],
    device const float* g [[buffer(6)]],
    device const float* wpre [[buffer(7)]],
    device const float* s [[buffer(8)]],
    device const float* c0 [[buffer(9)]],
    device const float* c1 [[buffer(10)]],
    device const float* c2 [[buffer(11)]],
    constant ZRow& p [[buffer(12)]],
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
        float ss = 0.0f;
        for (uint k = 0; k < cnt; ++k) { float t = yr[tid + 256u * k]; v[k] = t; ss += t * t; }
        const float inv = rsqrt(zi_block_sum(ss, red, sg, lane) / (float)H + p.eps1);
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            const float gg = p.has_g != 0u ? precise::tanh(g[i]) : 1.0f;
            const float xv = xr[i] + gg * (v[k] * inv * wpost[i]);
            xr[i] = xv;
            v[k] = xv;
        }
    } else {
        for (uint k = 0; k < cnt; ++k) v[k] = xr[tid + 256u * k];
    }
    if ((p.mode & 2u) != 0u) {
        float ss = 0.0f;
        for (uint k = 0; k < cnt; ++k) ss += v[k] * v[k];
        const float inv = rsqrt(zi_block_sum(ss, red, sg, lane) / (float)H + p.eps2);
        const ulong ob = (ulong)row * H;
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            float xn = v[k] * inv * wpre[i];
            if (p.has_s != 0u) xn *= 1.0f + s[i];
            xn *= p.oscale;
            o0[ob + i] = (half)(xn * c0[i]);
            if (p.nout > 1u) o1[ob + i] = (half)(xn * c1[i]);
            if (p.nout > 2u) o2[ob + i] = (half)(xn * c2[i]);
        }
    }
}

// ───────────── SwiGLU: h = silu(g)·u·col·oscale → half ─────────────
struct ZSw {
    uint n4;      // elements / 4
    uint inter;
    uint g_off;   // floats
    uint u_off;   // floats
    uint h_off;   // halves
    float oscale;
};

kernel void zi_swiglu(
    device const float* gu [[buffer(0)]],
    device half* h [[buffer(1)]],
    device const float* col [[buffer(2)]],
    constant ZSw& p [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n4) return;
    const float4 g = *(device const float4*)(gu + p.g_off + 4u * i);
    const float4 u = *(device const float4*)(gu + p.u_off + 4u * i);
    const float4 c = *(device const float4*)(col + (4u * i) % p.inter);
    const float4 sl = g / (1.0f + exp(-g));
    *(device half4*)(h + p.h_off + 4u * i) = half4(sl * u * c * p.oscale);
}

// ───────────── x_embedder (64 → H) + x_pad rows, per item ─────────────
struct ZEm {
    uint H;
    uint n_img;
    uint n_img_p;
    uint nitems;
};

kernel void zi_embed(
    device const float* xt [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device const float* xpad [[buffer(3)]],
    device float* x [[buffer(4)]],
    constant ZEm& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, t = gid.y;
    if (c >= p.H || t >= p.n_img_p) return;
    float v;
    if (t < p.n_img) {
        device const float4* w4 = (device const float4*)(W + (ulong)c * 64u);
        device const float4* x4 = (device const float4*)(xt + (ulong)t * 64u);
        float acc = 0.0f;
        for (ushort k = 0; k < 16; ++k) acc += dot(w4[k], x4[k]);
        v = acc + b[c];
    } else {
        v = xpad[c];
    }
    for (uint it = 0; it < p.nitems; ++it)
        x[((ulong)it * p.n_img_p + t) * p.H + c] = v;
}

// ─────── final layer: LayerNorm(eps, no affine)·scale → Linear(H → 64) + b ───────
struct ZFi {
    uint H;
    uint row0;
    uint out0;
    float eps;
};

kernel void zi_final(
    device const float* x [[buffer(0)]],
    device const float* fs [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant ZFi& p [[buffer(5)]],
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
    const float mean = zi_block_sum(s, red, sg, lane) / (float)H;
    float q = 0.0f;
    for (uint k = 0; k < cnt; ++k) { float d = v[k] - mean; q += d * d; }
    const float inv = rsqrt(zi_block_sum(q, red, sg, lane) / (float)H + p.eps);
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
        if (lane == 0) out[(ulong)(p.out0 + tgi) * 64u + o] = acc + b[o];
    }
}

// ───────────── plain float4 copy (caption rows into x) ─────────────
kernel void zi_copy4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant uint& n4 [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n4) dst[i] = src[i];
}

// ───────────── probe: pure simdgroup MMA issue rate (no loads) ─────────────
template <typename T>
static inline void zi_peak_body(device float* out, uint iters, ushort lane, uint gid) {
    simdgroup_matrix<T, 8, 8> a[4], b[4];
    for (ushort i = 0; i < 4; ++i) {
        a[i] = make_filled_simdgroup_matrix<T, 8, 8>((T)(0.001f * (lane + i)));
        b[i] = make_filled_simdgroup_matrix<T, 8, 8>((T)(0.002f * (lane + 2 * i)));
    }
    simdgroup_float8x8 acc[4][4];
    for (ushort i = 0; i < 4; ++i)
        for (ushort j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint it = 0; it < iters; ++it) {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 4; ++i)
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    float s = 0.0f;
    for (ushort i = 0; i < 4; ++i)
        for (ushort j = 0; j < 4; ++j) s += acc[i][j].thread_elements()[0];
    if (s == 12345.678f) out[gid] = s;
}

kernel void zi_peak_h(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
    ushort lane [[thread_index_in_simdgroup]], uint gid [[thread_position_in_grid]]) {
    zi_peak_body<half>(out, iters, lane, gid);
}
kernel void zi_peak_f(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
    ushort lane [[thread_index_in_simdgroup]], uint gid [[thread_position_in_grid]]) {
    zi_peak_body<float>(out, iters, lane, gid);
}

// ═══════════════════════════ resident Flux VAE ═══════════════════════════
// Activations NHWC; the residual stream f32, conv inputs half. Every conv
// is an implicit GEMM on the DiT GEMM's tile (64 pixels × 64 out-channels
// × 32 K, half weights [oc][tap][ic], f32 accumulation); the nearest-2×
// upsample is folded into the gather.
struct ZVc {
    uint n;        // output pixels (GEMM rows)
    uint rows;     // output channels (multiple of 64)
    uint K;        // taps · ic (plain mode: the row length)
    uint ic;       // input channels (multiple of 32)
    uint H;        // output height
    uint W;        // output width
    uint taps;     // 9 = 3×3 pad 1, 1 = 1×1, 0 = plain rows (X [n][ldx])
    uint up;       // 1: the source image is (H/2, W/2), nearest-upsampled
    uint epi;      // 0 f32 [n][ldy], 1 half [n][ldy], 2 half transposed [o][ldy]
    uint ldy;
    uint has_bias;
    uint has_res;  // f32 residual R[n][ldy] added (may alias Y)
    float mul;
    uint ldx;
    uint pad0;
    uint pad1;
};

static inline uint4 zv_gather(device const half* X, constant ZVc& p, uint pix, uint k) {
    if (pix >= p.n) return uint4(0);
    if (p.taps == 0u) return *(device const uint4*)(X + (ulong)pix * p.ldx + k);
    const uint tap = k / p.ic, c = k - tap * p.ic;
    int dy = 0, dx = 0;
    if (p.taps == 9u) { dy = (int)(tap / 3u) - 1; dx = (int)(tap % 3u) - 1; }
    const int y = (int)(pix / p.W) + dy, x = (int)(pix % p.W) + dx;
    if (y < 0 || x < 0 || y >= (int)p.H || x >= (int)p.W) return uint4(0);
    uint src;
    if (p.up != 0u) src = ((uint)y >> 1) * (p.W >> 1) + ((uint)x >> 1);
    else src = (uint)y * p.W + (uint)x;
    return *(device const uint4*)(X + (ulong)src * p.ic + c);
}

kernel void zv_conv(
    device const half* Wt [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const half* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const float* R [[buffer(4)]],
    constant ZVc& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    threadgroup half sw[2048];
    threadgroup half sx[2048];
    const uint t0 = tg.x * 64u, o0 = tg.y * 64u, K = p.K;
    const uint r = tid >> 1, kh = (tid & 1u) * 16u;
    device const uint4* wp = (device const uint4*)(Wt + (ulong)(o0 + r) * K + kh);
    const ushort sgo = sg & 1, sgt = sg >> 1;
    simdgroup_float8x8 acc[4][4];
    for (ushort i = 0; i < 4; ++i)
        for (ushort j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint4 wa = wp[0], wb = wp[1];
    uint4 xa = zv_gather(X, p, t0 + r, kh), xb = zv_gather(X, p, t0 + r, kh + 8u);
    threadgroup uint4* xd = (threadgroup uint4*)(sx + ((r / 8u) * 4u + kh / 8u) * 64u + (r % 8u) * 8u);
    threadgroup uint4* wd = (threadgroup uint4*)(sw + ((r / 8u) * 4u + kh / 8u) * 64u + (r % 8u) * 8u);
    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        wd[0] = wa; wd[8] = wb;
        xd[0] = xa; xd[8] = xb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 32u < K) {
            wp += 4;
            wa = wp[0];
            wb = wp[1];
            xa = zv_gather(X, p, t0 + r, k0 + 32u + kh);
            xb = zv_gather(X, p, t0 + r, k0 + 32u + kh + 8u);
        }
        #pragma clang loop unroll(full)
        for (ushort kb = 0; kb < 4; ++kb) {
            simdgroup_half8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                simdgroup_load(a[i], sx + ((4u * sgt + i) * 4u + kb) * 64u, 8);
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < 4; ++j)
                simdgroup_load(b[j], sw + ((4u * sgo + j) * 4u + kb) * 64u, 8, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
    if (t0 + 32u * sgt >= p.n) return;
    const short2 fc = zi_fc(lane);
    for (ushort j = 0; j < 4; ++j) {
        const uint o = o0 + 32u * sgo + 8u * j + (uint)fc.y;
        const float b0 = p.has_bias != 0u ? bias[o] : 0.0f;
        const float b1 = p.has_bias != 0u ? bias[o + 1] : 0.0f;
        for (ushort i = 0; i < 4; ++i) {
            const uint t = t0 + 32u * sgt + 8u * i + (uint)fc.x;
            float v0 = acc[i][j].thread_elements()[0] * p.mul + b0;
            float v1 = acc[i][j].thread_elements()[1] * p.mul + b1;
            if (p.epi == 2u) {
                device half* Yh = (device half*)Y;
                Yh[(ulong)o * p.ldy + t] = (half)v0;
                Yh[(ulong)(o + 1) * p.ldy + t] = (half)v1;
                continue;
            }
            const ulong yi = (ulong)t * p.ldy + o;
            if (p.has_res != 0u) {
                const float2 rr = *(device const float2*)(R + yi);
                v0 += rr.x;
                v1 += rr.y;
            }
            if (p.epi == 0u) *(device float2*)(Y + yi) = float2(v0, v1);
            else *(device half2*)((device half*)Y + yi) = half2(v0, v1);
        }
    }
}

// GroupNorm statistics, two exact passes: pass 0 sums x, pass 1 sums
// (x - mean)². Grid (groups, blocks); each group reads its cg = C/groups
// contiguous channels of every pixel of the block.
struct ZGn {
    uint n;        // pixels
    uint C;
    uint groups;
    uint nblk;     // blocks per group (grid.y)
    uint pass;     // 0 = sum, 1 = centred sum of squares
    float eps;
    uint silu;     // apply: SiLU after the affine
    uint half_out; // apply: 1 = half output, 0 = f32
};

kernel void zv_gn_part(
    device const float* x [[buffer(0)]],
    device float* part [[buffer(1)]],      // [groups][nblk]
    device const float* stat [[buffer(2)]],// [groups][2] mean, inv (pass 1 reads mean)
    constant ZGn& p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    const uint g = tg.x, b = tg.y;
    const uint cg = p.C / p.groups;
    const uint per = (p.n + p.nblk - 1) / p.nblk;
    const uint p0 = b * per, p1 = min(p.n, p0 + per);
    const float mean = p.pass == 1u ? stat[2u * g] : 0.0f;
    float acc = 0.0f;
    for (uint px = p0 + tid; px < p1; px += 256u) {
        device const float4* v4 = (device const float4*)(x + (ulong)px * p.C + g * cg);
        for (uint c = 0; c < cg / 4u; ++c) {
            float4 v = v4[c];
            if (p.pass == 1u) { v -= mean; acc += dot(v, v); }
            else acc += v.x + v.y + v.z + v.w;
        }
    }
    acc = zi_block_sum(acc, red, sg, lane);
    if (tid == 0) part[g * p.nblk + b] = acc;
}

// Reduce the partials of one pass: pass 0 → mean, pass 1 → inv std.
kernel void zv_gn_fin(
    device const float* part [[buffer(0)]],
    device float* stat [[buffer(1)]],
    constant ZGn& p [[buffer(2)]],
    uint g [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    float s = 0.0f;
    for (uint b = lane; b < p.nblk; b += 32u) s += part[g * p.nblk + b];
    s = simd_sum(s);
    if (lane == 0) {
        const float cnt = (float)p.n * (float)(p.C / p.groups);
        if (p.pass == 0u) stat[2u * g] = s / cnt;
        else stat[2u * g + 1u] = rsqrt(s / cnt + p.eps);
    }
}

// y = affine(norm(x)) [· SiLU] → half or f32, NHWC, float4 at a time.
kernel void zv_gn_apply(
    device const float* x [[buffer(0)]],
    device void* y [[buffer(1)]],
    device const float* stat [[buffer(2)]],
    device const float* w [[buffer(3)]],
    device const float* bb [[buffer(4)]],
    constant ZGn& p [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    const uint n4 = p.n * p.C / 4u;
    if (i >= n4) return;
    const uint c = (4u * i) % p.C;
    const uint g = c / (p.C / p.groups);
    const float mean = stat[2u * g], inv = stat[2u * g + 1u];
    float4 v = (((device const float4*)x)[i] - mean) * inv * *(device const float4*)(w + c)
             + *(device const float4*)(bb + c);
    if (p.silu != 0u) v = v / (1.0f + exp(-v));
    if (p.half_out != 0u) ((device half4*)y)[i] = half4(v);
    else ((device float4*)y)[i] = v;
}

// f32 → half, float4 at a time.
kernel void zv_cvt(
    device const float4* x [[buffer(0)]],
    device half4* y [[buffer(1)]],
    constant uint& n4 [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n4) y[i] = half4(x[i]);
}

// Row softmax of f32 scores [rows][n] → half P (one 256-thread group per row).
kernel void zv_softmax(
    device const float* s [[buffer(0)]],
    device half* pr [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device const float* sr = s + (ulong)row * n;
    float mx = -INFINITY;
    for (uint j = tid; j < n; j += 256u) mx = max(mx, sr[j]);
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = red[0];
    for (ushort i = 1; i < 8; ++i) m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < n; j += 256u) sum += exp(sr[j] - m);
    const float inv = 1.0f / zi_block_sum(sum, red, sg, lane);
    device half* pw = pr + (ulong)row * n;
    for (uint j = tid; j < n; j += 256u) pw[j] = (half)(exp(sr[j] - m) * inv);
}

// conv_out [n][64] f32 (3 real channels) → RGB planes [3][n].
kernel void zv_rgb(
    device const float* y [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    for (uint c = 0; c < 3u; ++c) out[(ulong)c * n + i] = y[(ulong)i * 64u + c];
}

// ───────────── debug: max|x| of a half range into slot (float bits) ─────────────
kernel void zi_amax(
    device const half* a [[buffer(0)]],
    device atomic_uint* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& slot [[buffer(3)]],
    uint i [[thread_position_in_grid]],
    uint gs [[threads_per_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    float m = 0.0f;
    for (uint j = i; j < n; j += gs) {
        float v = fabs((float)a[j]);
        m = isnan(v) ? INFINITY : max(m, v);
    }
    m = simd_max(m);
    if (lane == 0) atomic_fetch_max_explicit(&out[slot], as_type<uint>(m), memory_order_relaxed);
}
