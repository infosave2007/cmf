// Kandinsky S2V on Metal: the kernels the qi21 and Z-Image libraries do not
// have. Compiled after both (`gpu_metal/s2v.rs`).

// ───────────── row op ─────────────
// mode 1: x += g·(y + b)   (2: g from the buffer, else 1)
// mode 8: o = half((LN(x)·(1 + sc) + sh)·oscale)
// mode 16: o = half((LN(x)·sc + sh)·oscale)   (a learned LayerNorm)
// mode 32: o = half(x·oscale)
struct SRow {
    uint H;
    uint row0;    // first row of x
    uint orow0;   // first row of y and o
    uint mode;
    float eps;
    float oscale;
    uint pad0;
    uint pad1;
};

kernel void s2v_row(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device half* o [[buffer(2)]],
    device const float* g [[buffer(3)]],
    device const float* sh [[buffer(4)]],
    device const float* sc [[buffer(5)]],
    device const float* b [[buffer(6)]],
    constant SRow& p [[buffer(7)]],
    uint tgi [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    const uint H = p.H, cnt = H / 256u;
    device float* xr = x + (ulong)(p.row0 + tgi) * H;
    float v[16];
    if ((p.mode & 1u) != 0u) {
        device const float* yr = y + (ulong)(p.orow0 + tgi) * H;
        for (uint k = 0; k < cnt; ++k) {
            const uint i = tid + 256u * k;
            const float gv = (p.mode & 2u) != 0u ? g[i] : 1.0f;
            const float xv = xr[i] + gv * (yr[i] + b[i]);
            xr[i] = xv;
            v[k] = xv;
        }
    } else {
        for (uint k = 0; k < cnt; ++k) v[k] = xr[tid + 256u * k];
    }
    if ((p.mode & 56u) == 0u) return;
    float mean = 0.0f, inv = 1.0f;
    if ((p.mode & 24u) != 0u) {
        float sm = 0.0f;
        for (uint k = 0; k < cnt; ++k) sm += v[k];
        mean = qi_block_sum(sm, red, sg, lane) / (float)H;
        float q = 0.0f;
        for (uint k = 0; k < cnt; ++k) { const float d = v[k] - mean; q += d * d; }
        inv = rsqrt(qi_block_sum(q, red, sg, lane) / (float)H + p.eps);
    }
    const ulong ob = (ulong)(p.orow0 + tgi) * H;
    for (uint k = 0; k < cnt; ++k) {
        const uint i = tid + 256u * k;
        float r = v[k];
        if ((p.mode & 8u) != 0u) r = (v[k] - mean) * inv * (1.0f + sc[i]) + sh[i];
        else if ((p.mode & 16u) != 0u) r = (v[k] - mean) * inv * sc[i] + sh[i];
        o[ob + i] = (half)(r * p.oscale);
    }
}

// ───────── bias + per-head RMSNorm + RoPE, one panel column in place ─────────
struct SNorm {
    uint ld;
    uint col;
    uint nh;
    uint flags;   // 1 norm + weight, 2 rotate
    float eps;
    float bscale;
    float qmul;
    uint row0;
};

kernel void s2v_norm(
    device half* P [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* cs [[buffer(3)]],
    device const float* sn [[buffer(4)]],
    constant SNorm& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = p.row0 + tg.x;
    const uint head = tg.y * 4u + sg;
    if (head >= p.nh) return;
    device half4* ptr = (device half4*)(P + (ulong)row * p.ld + p.col + head * 128u + lane * 4u);
    float4 f = float4(*ptr) + *(device const float4*)(bias + head * 128u + lane * 4u) * p.bscale;
    if ((p.flags & 1u) != 0u) {
        const float ss = simd_sum(dot(f, f));
        f = f * rsqrt(ss / 128.0f + p.eps) * *(device const float4*)(w + lane * 4u);
    }
    if ((p.flags & 2u) != 0u) {
        const uint j = lane * 2u;
        const float2 c = *(device const float2*)(cs + (ulong)row * 64u + j);
        const float2 s = *(device const float2*)(sn + (ulong)row * 64u + j);
        float4 r;
        r.x = f.x * c.x - f.y * s.x;
        r.y = f.x * s.x + f.y * c.x;
        r.z = f.z * c.y - f.w * s.y;
        r.w = f.z * s.y + f.w * c.y;
        f = r;
    }
    *ptr = half4(f * p.qmul);
}

// ───────────── GELU (erf form) in place on a half panel ─────────────
struct SGelu {
    uint n4;
    float scale;   // the stored guard undone before, and put back after
    float iscale;
    uint pad0;
};

kernel void s2v_gelu(
    device half4* h [[buffer(0)]],
    constant SGelu& p [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.n4) return;
    const float4 v = float4(h[i]) * p.scale;
    const float4 a = fabs(v) * 0.70710678f;
    const float4 t = 1.0f / (1.0f + 0.3275911f * a);
    const float4 e = 1.0f - (((((1.0614054f * t - 1.4531521f) * t) + 1.4214138f) * t - 0.28449672f) * t + 0.2548296f) * t * exp(-a * a);
    const float4 erf = select(e, -e, v < 0.0f);
    h[i] = half4(0.5f * v * (1.0f + erf) * p.iscale);
}

// ───────────── add two row vectors to rows of x ─────────────
struct SAdd {
    uint H;
    uint n;
    uint pad0;
    uint pad1;
};

kernel void s2v_addv(
    device float* x [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    constant SAdd& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x, t = gid.y;
    if (c >= p.H || t >= p.n) return;
    x[(ulong)t * p.H + c] += a[c] + b[c];
}

// ───────────── head: LN(x)·(1 + sc) + sh → Linear(H → 320) + b ─────────────
struct SHead {
    uint H;
    uint row0;
    float eps;
    uint pad0;
};

kernel void s2v_head(
    device const float* x [[buffer(0)]],
    device const float* sh [[buffer(1)]],
    device const float* sc [[buffer(2)]],
    device const float* W [[buffer(3)]],
    device const float* bi [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant SHead& p [[buffer(6)]],
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
    for (uint k = 0; k < cnt; ++k) { const float d = v[k] - mean; q += d * d; }
    const float inv = rsqrt(qi_block_sum(q, red, sg, lane) / (float)H + p.eps);
    for (uint k = 0; k < cnt; ++k) {
        const uint i = tid + 256u * k;
        yr[i] = (v[k] - mean) * inv * (1.0f + sc[i]) + sh[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0; round < 5u; ++round) {
        for (uint o = round * 64u + sg * 8u; o < round * 64u + sg * 8u + 8u; ++o) {
            device const float* wr = W + (ulong)o * H;
            float acc = 0.0f;
            for (uint i = lane; i < H; i += 32u) acc += yr[i] * wr[i];
            acc = simd_sum(acc);
            if (lane == 0) out[(ulong)tgi * 320u + o] = acc + bi[o];
        }
    }
}

// ───────────── VAE: 3-D causal conv as an implicit GEMM ─────────────
struct SVc {
    uint n;        // output positions (GEMM rows)
    uint rows;     // output channels
    uint K;
    uint ic;       // input channels (multiple of 32)
    uint F;        // output frames
    uint H;
    uint W;
    uint taps;     // 27 = causal 3×3×3, 1 = 1×1×1, 0 = plain rows
    uint up_s;     // 1: the source is half size in space, 2: stride 2 (twice the size)
    uint up_t;     // first frame + 2× in time
    uint epi;      // 0 f32, 1 half, 2 half transposed
    uint ldy;
    uint has_bias;
    uint has_res;
    float mul;
    uint ldx;
};

static inline uint4 s2v_gather(device const half* X, constant SVc& p, uint pos, uint k) {
    if (pos >= p.n) return uint4(0);
    if (p.taps == 0u) return *(device const uint4*)(X + (ulong)pos * p.ldx + k);
    if (p.taps == 1u) return *(device const uint4*)(X + (ulong)pos * p.ic + k);
    const uint tap = k / p.ic, c = k - tap * p.ic;
    const uint per = p.H * p.W;
    const uint fr = pos / per, rest = pos % per;
    const int y = (int)(rest / p.W), x = (int)(rest % p.W);
    const int sd = p.up_s == 2u ? 2 : 1;
    const int sf = clamp((int)fr + (int)(tap / 9u) - 2, 0, (int)p.F - 1);
    const int sy = clamp(y * sd + (int)((tap / 3u) % 3u) - 1, 0, (int)p.H * sd - 1);
    const int sx = clamp(x * sd + (int)(tap % 3u) - 1, 0, (int)p.W * sd - 1);
    uint tf = (uint)sf;
    if (p.up_t != 0u && tf > 0u) tf = (tf - 1u) / 2u + 1u;
    const uint us = p.up_s == 1u ? 1u : 0u;
    const uint sh = (p.H * (uint)sd) >> us, sw = (p.W * (uint)sd) >> us;
    const uint src = (tf * sh + ((uint)sy >> us)) * sw + ((uint)sx >> us);
    return *(device const uint4*)(X + (ulong)src * p.ic + c);
}

kernel void s2v_vconv(
    device const half* Wt [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device const half* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const float* R [[buffer(4)]],
    constant SVc& p [[buffer(5)]],
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
    uint4 xa = s2v_gather(X, p, t0 + r, kh), xb = s2v_gather(X, p, t0 + r, kh + 8u);
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
            xa = s2v_gather(X, p, t0 + r, k0 + 32u + kh);
            xb = s2v_gather(X, p, t0 + r, k0 + 32u + kh + 8u);
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
    const short2 fc = qi_fc(lane);
    for (ushort j = 0; j < 4; ++j) {
        const uint o = o0 + 32u * sgo + 8u * j + (uint)fc.y;
        if (p.epi != 2u && o + 1u >= p.rows) continue;
        const float b0 = p.has_bias != 0u ? bias[o] : 0.0f;
        const float b1 = p.has_bias != 0u ? bias[o + 1] : 0.0f;
        for (ushort i = 0; i < 4; ++i) {
            const uint t = t0 + 32u * sgt + 8u * i + (uint)fc.x;
            if (t >= p.n) continue;
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

// Row softmax over the first n of ld columns, the rest written as zero.
kernel void s2v_softmax(
    device const float* s [[buffer(0)]],
    device half* pr [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& ld [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device const float* sr = s + (ulong)row * ld;
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
    const float inv = 1.0f / qi_block_sum(sum, red, sg, lane);
    device half* pw = pr + (ulong)row * ld;
    for (uint j = tid; j < ld; j += 256u) pw[j] = j < n ? (half)(exp(sr[j] - m) * inv) : (half)0;
}

// conv_out [n][64] f32 (3 real channels) → RGB planes [3][n].
kernel void s2v_rgb(
    device const float* y [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    for (uint c = 0; c < 3u; ++c) out[(ulong)c * n + i] = y[(ulong)i * 64u + c];
}

// ───────────────── self-attention with f32 weights ─────────────────
// `qi_flash` holds its weights in half, so it renormalises each query against its
// running maximum on every key block. The branch alone costs a third of the kernel's
// rate, and half of the rate goes with the rescaling behind it. A few keys stand far
// above the rest and scores pass 100 in some blocks, so half cannot do without that.
// With f32 weights one scale per query is enough: the largest score over 16 key blocks
// spread across the keys, which leaves 100 bits above it. The f32 product is about as
// fast as the half one, and the kernel runs 1.8 times as fast at the size of a video.
// Every key is visible, because the row count is a multiple of 32, and a row's sum is
// gathered across its lanes at the end.
kernel void s2v_flash(
    device const half* P [[buffer(0)]],
    device half* O [[buffer(1)]],
    constant QFa& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]])
{
    constexpr uint NT = 256u, TOT = 32u * 16u, SAMPLES = 16u;
    threadgroup half sk[32 * 128];
    threadgroup float sv[32 * 128];
    const uint hq = (tg.y + p.head0) * 128u;
    const uint ntot = p.n_own;
    const uint q0 = tg.x * 64u + 8u * sg;
    const uint qi = min(q0, p.nq - 8u);
    simdgroup_half8x8 qf[16];
    for (ushort d = 0; d < 16; ++d)
        simdgroup_load(qf[d], P + (ulong)qi * p.ldp + hq + 8u * d, p.ldp);
    simdgroup_float8x8 of[16];
    for (ushort d = 0; d < 16; ++d) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float l = 0.0f, m = -INFINITY;
    const short2 fc = qi_fc(lane);
    const uint blocks = ntot / 32u;
    for (uint i = 0; i < SAMPLES; ++i) {
        const uint kb0 = (blocks * i / SAMPLES) * 32u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ch = tid; ch < TOT; ch += NT) {
            const uint key = ch / 16u, part = (ch % 16u) * 8u;
            device const half* kp = P + (ulong)(kb0 + key) * p.ldp + p.H + hq;
            *(threadgroup uint4*)(sk + key * 128u + part) = *(device const uint4*)(kp + part);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, sk + 8u * c * 128u + 8u * d, 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qf[d], kf, s);
            }
            m = max(m, max(s.thread_elements()[0], s.thread_elements()[1]));
        }
    }
    m = max(m, simd_shuffle_xor(m, 1));
    m = max(m, simd_shuffle_xor(m, 8));
    for (uint kb0 = 0; kb0 < ntot; kb0 += 32u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ch = tid; ch < TOT; ch += NT) {
            const uint key = ch / 16u, part = (ch % 16u) * 8u;
            device const half* kp = P + (ulong)(kb0 + key) * p.ldp + p.H + hq;
            const uint off = key * 128u + part;
            *(threadgroup uint4*)(sk + off) = *(device const uint4*)(kp + part);
            device const half4* vq = (device const half4*)(kp + p.H + part);
            *(threadgroup float4*)(sv + off) = (float4)vq[0];
            *(threadgroup float4*)(sv + off + 4u) = (float4)vq[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 pf[4];
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, sk + 8u * c * 128u + 8u * d, 128, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qf[d], kf, s);
            }
            const float p0 = exp2(min(s.thread_elements()[0] - m, 100.0f));
            const float p1 = exp2(min(s.thread_elements()[1] - m, 100.0f));
            l += p0 + p1;
            pf[c].thread_elements()[0] = p0;
            pf[c].thread_elements()[1] = p1;
        }
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 4; ++c) {
            #pragma clang loop unroll(full)
            for (ushort d = 0; d < 16; ++d) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, sv + 8u * c * 128u + 8u * d, 128);
                simdgroup_multiply_accumulate(of[d], pf[c], vf, of[d]);
            }
        }
    }
    l += simd_shuffle_xor(l, 1);
    l += simd_shuffle_xor(l, 8);
    if (q0 >= p.nq) return;
    const float inv = 1.0f / l;
    const uint orow = qi + (uint)fc.x;
    for (ushort d = 0; d < 16; ++d) {
        const uint col = hq + 8u * d + (uint)fc.y;
        const half2 v = half2(of[d].thread_elements()[0] * inv, of[d].thread_elements()[1] * inv);
        *(device half2*)(O + (ulong)orow * p.ldo + col) = v;
    }
}
