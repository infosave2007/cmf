// EmbeddingGemma 2 text encoder on Metal: the module-local MSL library,
// compiled once per process by `gpu_metal/egemma2.rs`.
//
// Everything is f32 — operands, accumulators and every row op — so the
// device forward is the CPU forward up to summation order (the model's
// residual stream overflows f16, and the exact files promise f32 maths).
//
// GEMMs take an item table (one item per grid z): each item names its own
// shapes, strides and offsets, so one dispatch runs the q/k/v projections of
// a layer (three weights) or every (sequence, head) block of an attention.
#include <metal_stdlib>
using namespace metal;

// (row, col) of this lane's two consecutive elements in an 8x8 fragment
// (the Apple GPU layout; see gpu_metal/zimage_msl.metal `zi_fc`).
static inline short2 eg_fc(ushort lane) {
    short qid = (short)(lane / 4);
    return short2((qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2);
}

kernel void eg_fragprobe(
    device float* out [[buffer(0)]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float src[64];
    for (ushort i = lane; i < 64; i += 32) src[i] = (float)i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 m;
    simdgroup_load(m, src, 8);
    short2 fc = eg_fc(lane);
    out[lane * 4 + 0] = m.thread_elements()[0];
    out[lane * 4 + 1] = m.thread_elements()[1];
    out[lane * 4 + 2] = (float)(fc.x * 8 + fc.y);
    out[lane * 4 + 3] = (float)(fc.x * 8 + fc.y + 1);
}

// ─────────────────────────── GEMM ───────────────────────────
// Y[t][o] = alpha · Σ_k X[t][k] · W(k, o)
//   NT: W(k, o) = W[o][k]   (a weight matrix [m][K], or attention keys)
//   NN: W(k, o) = W[k][o]   (attention values)
// Rows past `n` and features past `m` load a clamped (valid) row and are
// never stored. K is a multiple of 32 for NT; for NN the item's K is
// rounded up to 32 by the host and X carries zeros in the padding, where
// the W row index is clamped to the item's last valid row (`wrows - 1`).
struct EgMm {
    uint n;
    uint m;
    uint K;
    uint ldx;
    uint ldw;
    uint ldy;
    uint x_off;
    uint w_off;
    uint y_off;
    float alpha;
    uint wrows;   // NN: valid W rows (the clamp); NT: unused
    uint pad1;
};

// 64 rows × 64 features × 32 k per 128-thread group; 2×2 simdgroups, each
// 32 rows × 32 features (4×4 fragments). Operand tiles are packed as dense
// 8×8 blocks in threadgroup memory; the next tile is prefetched into
// registers while the current one multiplies.
template <bool NN>
static inline void eg_mm_body(
    device const float* X, device const float* W, device float* Y,
    device const EgMm& p,
    threadgroup float* sw, threadgroup float* sx,
    uint tid, ushort sg, ushort lane, uint2 tg)
{
    const uint t0 = tg.x * 64u, o0 = tg.y * 64u;
    if (t0 >= p.n || o0 >= p.m) return;
    const uint K = p.K;
    // X tile: row r, k range [kh, kh + 16)
    const uint r = tid >> 1, kh = (tid & 1u) * 16u;
    device const float4* xp =
        (device const float4*)(X + p.x_off + (ulong)min(t0 + r, p.n - 1u) * p.ldx + kh);
    // W tile
    device const float* wbase;
    uint wr, wc;
    if (NN) {
        wr = tid >> 2;            // k row within the tile, 0..31
        wc = (tid & 3u) * 16u;    // o range [wc, wc + 16)
        wbase = W + p.w_off + o0 + wc;
    } else {
        wr = r;
        wc = kh;
        wbase = W + p.w_off + (ulong)min(o0 + r, p.m - 1u) * p.ldw + kh;
    }
    const ushort sgo = sg & 1, sgt = sg >> 1;
    simdgroup_float8x8 acc[4][4];
    for (ushort i = 0; i < 4; ++i)
        for (ushort j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    float4 xa = xp[0], xb = xp[1], xc = xp[2], xd = xp[3];
    float4 wa, wb, wcc, wd;
    {
        device const float4* wp = NN
            ? (device const float4*)(wbase + (ulong)min(wr, p.wrows - 1u) * p.ldw)
            : (device const float4*)wbase;
        wa = wp[0]; wb = wp[1]; wcc = wp[2]; wd = wp[3];
    }
    threadgroup float4* xs = (threadgroup float4*)(sx + ((r / 8u) * 4u + kh / 8u) * 64u + (r % 8u) * 8u);
    threadgroup float4* ws = NN
        ? (threadgroup float4*)(sw + ((wr / 8u) * 8u + wc / 8u) * 64u + (wr % 8u) * 8u)
        : (threadgroup float4*)(sw + ((wr / 8u) * 4u + wc / 8u) * 64u + (wr % 8u) * 8u);
    for (uint k0 = 0; k0 < K; k0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // one block row is 8 floats (2 float4); the next 8x8 block is +64
        // floats (+16 float4)
        xs[0] = xa; xs[1] = xb; xs[16] = xc; xs[17] = xd;
        ws[0] = wa; ws[1] = wb; ws[16] = wcc; ws[17] = wd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 32u < K) {
            xp += 8;
            xa = xp[0]; xb = xp[1]; xc = xp[2]; xd = xp[3];
            device const float4* wp = NN
                ? (device const float4*)(wbase + (ulong)min(k0 + 32u + wr, p.wrows - 1u) * p.ldw)
                : (device const float4*)(wbase + k0 + 32u);
            wa = wp[0]; wb = wp[1]; wcc = wp[2]; wd = wp[3];
        }
        #pragma clang loop unroll(full)
        for (ushort kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                simdgroup_load(a[i], sx + ((4u * sgt + i) * 4u + kb) * 64u, 8);
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < 4; ++j) {
                if (NN)
                    simdgroup_load(b[j], sw + (kb * 8u + 4u * sgo + j) * 64u, 8);
                else
                    simdgroup_load(b[j], sw + ((4u * sgo + j) * 4u + kb) * 64u, 8, ulong2(0, 0), true);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4; ++i)
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
    if (t0 + 32u * sgt >= p.n) return;
    const short2 fc = eg_fc(lane);
    const float al = p.alpha;
    for (ushort j = 0; j < 4; ++j) {
        const uint o = o0 + 32u * sgo + 8u * j + (uint)fc.y;
        if (o >= p.m) continue;
        const bool both = o + 1u < p.m;
        for (ushort i = 0; i < 4; ++i) {
            const uint t = t0 + 32u * sgt + 8u * i + (uint)fc.x;
            if (t >= p.n) continue;
            device float* yp = Y + p.y_off + (ulong)t * p.ldy + o;
            yp[0] = acc[i][j].thread_elements()[0] * al;
            if (both) yp[1] = acc[i][j].thread_elements()[1] * al;
        }
    }
}

#define EG_MM_ARGS \
    device const float* X [[buffer(0)]], \
    device const float* w0 [[buffer(1)]], \
    device const float* w1 [[buffer(2)]], \
    device const float* w2 [[buffer(3)]], \
    device float* Y [[buffer(4)]], \
    device const EgMm* items [[buffer(5)]], \
    uint tid [[thread_index_in_threadgroup]], \
    ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]], \
    uint3 tg [[threadgroup_position_in_grid]]

kernel void eg_mm_nt(EG_MM_ARGS)
{
    threadgroup float sw[2048];
    threadgroup float sx[2048];
    const uint z = tg.z;
    device const float* W = z == 0u ? w0 : (z == 1u ? w1 : w2);
    eg_mm_body<false>(X, W, Y, items[z], sw, sx, tid, sg, lane, tg.xy);
}

kernel void eg_mm_nn(EG_MM_ARGS)
{
    threadgroup float sw[2048];
    threadgroup float sx[2048];
    const uint z = tg.z;
    device const float* W = z == 0u ? w0 : (z == 1u ? w1 : w2);
    eg_mm_body<true>(X, W, Y, items[z], sw, sx, tid, sg, lane, tg.xy);
}

// ─────────────────────────── row ops ───────────────────────────
// One simdgroup per row (4 rows per 128-thread group), float4 lanes; widths
// are multiples of 4. Sums of squares reduce in f32 (the CPU sums in f64:
// the two agree to ~1e-7 relative).

struct EgRow {
    uint d;
    uint ldx;
    uint ldy;
    float eps;
    float pre;     // x is scaled by this first (rms of pre·x)
    float post;    // add_norm: h = (h + rms(y)·w) · post
    uint has_w;    // rms: weight given; add_norm: second norm given
    uint n;        // rows
};

// Y = rms(pre·X) · w
kernel void eg_rms(
    device const float* X [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* Y [[buffer(2)]],
    constant EgRow& p [[buffer(3)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.n) return;
    device const float* x = X + (ulong)row * p.ldx;
    device float* y = Y + (ulong)row * p.ldy;
    float ss = 0.0f;
    for (uint i = lane * 4u; i < p.d; i += 128u) {
        float4 v = *(device const float4*)(x + i) * p.pre;
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)p.d + p.eps);
    for (uint i = lane * 4u; i < p.d; i += 128u) {
        float4 v = (*(device const float4*)(x + i) * p.pre) * inv;
        if (p.has_w != 0u) v *= *(device const float4*)(w + i);
        *(device float4*)(y + i) = v;
    }
}

// H = (H + rms(Yin) · w) · post, then (has_w) A = rms(H) · w2 — the
// post-norm residual add fused with the next block's pre-norm.
kernel void eg_add_norm(
    device float* H [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* Yin [[buffer(2)]],
    device float* A [[buffer(3)]],
    device const float* w2 [[buffer(4)]],
    constant EgRow& p [[buffer(5)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.n) return;
    device const float* y = Yin + (ulong)row * p.ldx;
    device float* h = H + (ulong)row * p.ldy;
    float ss = 0.0f;
    for (uint i = lane * 4u; i < p.d; i += 128u) {
        float4 v = *(device const float4*)(y + i);
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)p.d + p.eps);
    float ss2 = 0.0f;
    for (uint i = lane * 4u; i < p.d; i += 128u) {
        float4 v = *(device const float4*)(h + i)
                 + (*(device const float4*)(y + i) * inv) * *(device const float4*)(w + i);
        if (p.post != 1.0f) v *= p.post;
        *(device float4*)(h + i) = v;
        ss2 += dot(v, v);
    }
    if (p.has_w == 0u) return;
    ss2 = simd_sum(ss2);
    const float inv2 = rsqrt(ss2 / (float)p.d + p.eps);
    device float* a = A + (ulong)row * p.ldy;
    for (uint i = lane * 4u; i < p.d; i += 128u) {
        *(device float4*)(a + i) = (*(device const float4*)(h + i) * inv2) * *(device const float4*)(w2 + i);
    }
}

// O = gelu_tanh(G) · U, elementwise over [n][d] (strides per operand)
struct EgAct {
    uint d;
    uint ldg;
    uint ldu;
    uint ldo;
};

kernel void eg_gelu_mul(
    device const float* G [[buffer(0)]],
    device const float* U [[buffer(1)]],
    device float* O [[buffer(2)]],
    constant EgAct& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x * 4u, row = gid.y;
    if (i >= p.d) return;
    float4 g = *(device const float4*)(G + (ulong)row * p.ldg + i);
    float4 u = *(device const float4*)(U + (ulong)row * p.ldu + i);
    // 0.5·g·(1 + tanh z) = g / (1 + exp(-2z)),  z = √(2/π)(g + 0.044715 g³)
    float4 z = 0.7978845608f * (g + 0.044715f * g * g * g);
    float4 r = g / (1.0f + precise::exp(-2.0f * z));
    *(device float4*)(O + (ulong)row * p.ldo + i) = r * u;
}

// Per-head q/k RMS-norm with weight + RoPE, v RMS-norm (no weight), in
// place on a [n][ld] q|k|v row: one row per group, a head per simdgroup
// (the group's simdgroups stride over the heads). RoPE is `rotate_half`
// within each `sec` channels of a head (the text model: one section of the
// whole head; the vision tower: the column half, then the row half), the
// angles of row `pos[row]` of the [*, hd/2] cos/sin tables, section s
// reading entries [s·sec/2, (s+1)·sec/2).
struct EgQkv {
    uint ld;
    uint hd;
    uint nq;
    uint nkv;
    uint q_off;
    uint k_off;
    uint v_off;
    float eps;
    uint sec;
    uint pad0;
};

kernel void eg_qkv_prep(
    device float* QKV [[buffer(0)]],
    device const float* qn [[buffer(1)]],
    device const float* kn [[buffer(2)]],
    device const float* rcos [[buffer(3)]],
    device const float* rsin [[buffer(4)]],
    device const uint* pos [[buffer(5)]],
    constant EgQkv& p [[buffer(6)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    const uint hd = p.hd, half_d = hd / 2u, hs = p.sec / 2u;
    device float* base = QKV + (ulong)row * p.ld;
    device const float* c = rcos + (ulong)pos[row] * half_d;
    device const float* s = rsin + (ulong)pos[row] * half_d;
    const uint heads = p.nq + 2u * p.nkv;
    for (uint h = sg; h < heads; h += nsg) {
        device float* x;
        device const float* w = nullptr;
        if (h < p.nq) {
            x = base + p.q_off + h * hd; w = qn;
        } else if (h < p.nq + p.nkv) {
            x = base + p.k_off + (h - p.nq) * hd; w = kn;
        } else {
            x = base + p.v_off + (h - p.nq - p.nkv) * hd;
        }
        float ss = 0.0f;
        for (uint i = lane * 4u; i < hd; i += 128u) {
            float4 v = *(device const float4*)(x + i);
            ss += dot(v, v);
        }
        ss = simd_sum(ss);
        const float inv = rsqrt(ss / (float)hd + p.eps);
        if (w != nullptr) {
            for (uint s0 = 0; s0 < hd; s0 += p.sec) {
                device float* xs = x + s0;
                device const float* ws = w + s0;
                device const float* cs = c + s0 / 2u;
                device const float* sn = s + s0 / 2u;
                for (uint i = lane * 4u; i < hs; i += 128u) {
                    float4 a = (*(device const float4*)(xs + i) * inv) * *(device const float4*)(ws + i);
                    float4 b = (*(device const float4*)(xs + i + hs) * inv)
                             * *(device const float4*)(ws + i + hs);
                    float4 cc = *(device const float4*)(cs + i), si = *(device const float4*)(sn + i);
                    *(device float4*)(xs + i) = a * cc - b * si;
                    *(device float4*)(xs + i + hs) = b * cc + a * si;
                }
            }
        } else {
            for (uint i = lane * 4u; i < hd; i += 128u) {
                *(device float4*)(x + i) = *(device const float4*)(x + i) * inv;
            }
        }
    }
}

// Flash attention over whole sequences (no window), f32 throughout, for
// small heads (the vision tower's 64): no score matrix in memory. Grid:
// x = 32-query tile, y = q head, z = sequence ([start, len] in `segs`);
// four simdgroups of 8 query rows each; keys in blocks of 32, the online
// softmax per row. Q/K/V are read in place from the q|k|v rows (stride
// `ld`); a tail block reads up to 31 rows past the sequence (another
// sequence's, or the buffer's zeroed padding — the host allocates 32 rows
// more) and masks them to p = 0. The output goes to att[t][h·HD..].
struct EgFlash {
    uint ld;
    uint q_off;
    uint k_off;
    uint v_off;
    uint ldo;
    uint nq;     // q heads
    uint group;  // q heads per kv head
    float scale;
};

template <uint HD>
static inline void eg_flash_body(
    device const float* QKV, device float* out, device const uint* segs,
    constant EgFlash& p, uint3 tg, ushort sg, ushort lane,
    threadgroup float* ss, threadgroup float* sd, threadgroup float* sm, threadgroup float* sl)
{
    const uint s0 = segs[2u * tg.z], len = segs[2u * tg.z + 1u];
    const uint qbase = tg.x * 32u;
    if (qbase >= len) return;
    const uint h = tg.y, kvh = h / p.group;
    device const float* qrow = QKV + (ulong)(s0 + qbase + 8u * sg) * p.ld + p.q_off + h * HD;
    device const float* kbase = QKV + (ulong)s0 * p.ld + p.k_off + kvh * HD;
    device const float* vbase = QKV + (ulong)s0 * p.ld + p.v_off + kvh * HD;
    const uint srow = lane / 4u, schunk = lane % 4u;
    if (lane < 8u) {
        sm[lane] = -INFINITY;
        sl[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NOB = HD / 8u;
    simdgroup_float8x8 qf[NOB];
    for (uint kb = 0; kb < NOB; ++kb)
        simdgroup_load(qf[kb], qrow + kb * 8u, p.ld, ulong2(0, 0), false);
    simdgroup_float8x8 o8[NOB];
    for (uint i = 0; i < NOB; ++i) o8[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kb0 = 0; kb0 < len; kb0 += 32u) {
        // S[8×32] = Q·Kᵀ
        for (uint cb = 0; cb < 4u; ++cb) {
            simdgroup_float8x8 s8 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            device const float* krow = kbase + (ulong)(kb0 + cb * 8u) * p.ld;
            for (uint kb = 0; kb < NOB; ++kb) {
                simdgroup_float8x8 b8;
                simdgroup_load(b8, krow + kb * 8u, p.ld, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s8, qf[kb], b8, s8);
            }
            simdgroup_store(s8, ss + cb * 8u, 32u, ulong2(0, 0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: 4 lanes per row, 8 columns each
        const float mprev = sm[srow], lprev = sl[srow];
        float lmax = -INFINITY;
        for (uint j = 0; j < 8u; ++j) {
            const uint col = schunk * 8u + j;
            if (kb0 + col < len) lmax = max(lmax, ss[srow * 32u + col] * p.scale);
        }
        lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
        lmax = max(lmax, simd_shuffle_xor(lmax, 2u));
        const float mnew = max(mprev, lmax);
        const float alpha = precise::exp(mprev - mnew);
        float psum = 0.0f;
        for (uint j = 0; j < 8u; ++j) {
            const uint col = schunk * 8u + j;
            const float e = (kb0 + col < len) ? precise::exp(ss[srow * 32u + col] * p.scale - mnew) : 0.0f;
            ss[srow * 32u + col] = e;
            psum += e;
        }
        psum += simd_shuffle_xor(psum, 1u);
        psum += simd_shuffle_xor(psum, 2u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (schunk == 0u) {
            sm[srow] = mnew;
            sl[srow] = alpha * lprev + psum;
        }
        // O = diag(alpha)·O when a row max moved
        if (simd_any(alpha != 1.0f)) {
            for (uint i = lane; i < 64u; i += 32u) sd[i] = 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (schunk == 0u) sd[srow * 8u + srow] = alpha;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 d8;
            simdgroup_load(d8, sd, 8u, ulong2(0, 0), false);
            for (uint i = 0; i < NOB; ++i) {
                simdgroup_float8x8 t8;
                simdgroup_multiply(t8, d8, o8[i]);
                o8[i] = t8;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // O += P·V
        for (uint kb = 0; kb < 4u; ++kb) {
            simdgroup_float8x8 a8;
            simdgroup_load(a8, ss + kb * 8u, 32u, ulong2(0, 0), false);
            device const float* vrow = vbase + (ulong)(kb0 + kb * 8u) * p.ld;
            for (uint i = 0; i < NOB; ++i) {
                simdgroup_float8x8 b8;
                simdgroup_load(b8, vrow + i * 8u, p.ld, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(o8[i], a8, b8, o8[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // O /= l, store the rows inside the sequence
    for (uint i = lane; i < 64u; i += 32u) sd[i] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (schunk == 0u) sd[srow * 8u + srow] = 1.0f / sl[srow];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 d8;
    simdgroup_load(d8, sd, 8u, ulong2(0, 0), false);
    const uint row0 = qbase + 8u * sg;
    if (row0 >= len) return;
    device float* obase = out + (ulong)(s0 + row0) * p.ldo + h * HD;
    for (uint i = 0; i < NOB; ++i) {
        simdgroup_float8x8 t8;
        simdgroup_multiply(t8, d8, o8[i]);
        if (row0 + 8u <= len) {
            simdgroup_store(t8, obase + i * 8u, p.ldo, ulong2(0, 0), false);
        } else {
            simdgroup_store(t8, sd, 8u, ulong2(0, 0), false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint r = e / 8u;
                if (row0 + r < len) obase[(ulong)r * p.ldo + i * 8u + e % 8u] = sd[e];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) sd[e] = 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (schunk == 0u) sd[srow * 8u + srow] = 1.0f / sl[srow];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(d8, sd, 8u, ulong2(0, 0), false);
        }
    }
}

kernel void eg_flash64(
    device const float* QKV [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const uint* segs [[buffer(2)]],
    constant EgFlash& p [[buffer(3)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]])
{
    threadgroup float ssm[4 * 8 * 32];
    threadgroup float sdm[4 * 64];
    threadgroup float smm[4 * 8];
    threadgroup float slm[4 * 8];
    eg_flash_body<64>(QKV, out, segs, p, tg, sg, lane,
                      ssm + sg * 256u, sdm + sg * 64u, smm + sg * 8u, slm + sg * 8u);
}

// Softmax of each attention row over its allowed keys, in place; zeros
// elsewhere up to the padded row width. One simdgroup per row.
struct EgAtt {
    uint s_off;    // this item's S block (elements)
    uint nq;       // query rows
    uint nk;       // keys (columns used)
    uint lds;      // S row stride (nk rounded up to 32)
    uint q0;       // first query's position in its sequence
    uint k0;       // first key's position in its sequence
    uint len;      // sequence length
    uint window;   // 0 = full attention, else |i - j| <= window
};

kernel void eg_softmax(
    device float* S [[buffer(0)]],
    device const EgAtt* items [[buffer(1)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    device const EgAtt& it = items[tg.y];
    const uint i = tg.x * 4u + sg;
    if (i >= it.nq) return;
    device float* row = S + it.s_off + (ulong)i * it.lds;
    const uint qi = it.q0 + i;
    uint lo = 0u, hi = it.len;
    if (it.window != 0u) {
        lo = qi > it.window ? qi - it.window : 0u;
        hi = min(it.len, qi + it.window + 1u);
    }
    // to columns of this block
    lo = lo > it.k0 ? lo - it.k0 : 0u;
    hi = hi > it.k0 ? min(hi - it.k0, it.nk) : 0u;
    float mx = -INFINITY;
    for (uint j = lo + lane; j < hi; j += 32u) mx = max(mx, row[j]);
    mx = simd_max(mx);
    float sum = 0.0f;
    for (uint j = lo + lane; j < hi; j += 32u) {
        float e = precise::exp(row[j] - mx);
        row[j] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    const float inv = 1.0f / sum;
    for (uint j = lane; j < it.lds; j += 32u) {
        row[j] = (j >= lo && j < hi) ? row[j] * inv : 0.0f;
    }
}

// Final RMS-norm (with weight) and mean over each sequence's rows:
// out[s][c] = mean_t rms(H[t]) · w. Two passes: per-row 1/rms, then a
// column sum per (sequence, 128-column slice).
kernel void eg_row_inv(
    device const float* H [[buffer(0)]],
    device float* inv [[buffer(1)]],
    constant EgRow& p [[buffer(2)]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    const uint row = tg * 4u + sg;
    if (row >= p.n) return;
    device const float* x = H + (ulong)row * p.ldx;
    float ss = 0.0f;
    for (uint i = lane * 4u; i < p.d; i += 128u) {
        float4 v = *(device const float4*)(x + i);
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    if (lane == 0) inv[row] = rsqrt(ss / (float)p.d + p.eps);
}

kernel void eg_pool(
    device const float* H [[buffer(0)]],
    device const float* inv [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device const uint* segs [[buffer(3)]],   // [start, len] per sequence
    device float* out [[buffer(4)]],
    constant EgRow& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, s = gid.y;
    if (c >= p.d) return;
    const uint s0 = segs[2u * s], l = segs[2u * s + 1u];
    float acc = 0.0f;
    for (uint t = 0; t < l; ++t) {
        acc += (H[(ulong)(s0 + t) * p.ldx + c] * inv[s0 + t]) * w[c];
    }
    out[(ulong)s * p.d + c] = acc / (float)l;
}
