// Candidate GDN scans for the small path: a lane owns a state COLUMN
// (value head h, value dim j), so a warp reads 32 consecutive floats of a
// key row. Variant A: one warp holds all 128 key rows of its 32 columns.
// Variant B: a block of DKW warps, each 128 / DKW key rows, the dot products
// summed across warps in a fixed order through shared memory.

// A: one warp a (head, 32 columns).
extern "C" __global__ void __launch_bounds__(32) gdn_scan_a(const float* __restrict__ cq, const float* __restrict__ gb,
                                                           float* S, float* __restrict__ o, int n, int c_dim, int nv,
                                                           int nk, int dv, float* __restrict__ ssave)
{
    const int lane = threadIdx.x;
    const int wpr = dv >> 5;
    const int h = blockIdx.x / wpr, j = (blockIdx.x - h * wpr) * 32 + lane;
    const int ko = h / (nv / nk), kd = nk * 128;
    float* sp = S + (size_t)h * 128 * dv + j;
    float s[128];
#pragma unroll
    for (int i = 0; i < 128; i++) s[i] = sp[(size_t)i * dv];
    if (ssave) {
        float* wp = ssave + (size_t)h * 128 * dv + j;
#pragma unroll
        for (int i = 0; i < 128; i++) wp[(size_t)i * dv] = s[i];
    }
    __shared__ float ks[128], qs[128];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (size_t)t * c_dim;
        *(float4*)(ks + lane * 4) = *(const float4*)(row + kd + (size_t)ko * 128 + lane * 4);
        *(float4*)(qs + lane * 4) = *(const float4*)(row + (size_t)ko * 128 + lane * 4);
        __syncwarp();
        const float g = gb[(size_t)t * 2 * nv + h], beta = gb[(size_t)t * 2 * nv + nv + h];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
        for (int i = 0; i < 128; i += 4) {
            a0 = fmaf(s[i], ks[i], a0);
            a1 = fmaf(s[i + 1], ks[i + 1], a1);
            a2 = fmaf(s[i + 2], ks[i + 2], a2);
            a3 = fmaf(s[i + 3], ks[i + 3], a3);
        }
        const float kv = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
        const float delta = __fmul_rn(__fsub_rn(row[2 * kd + (size_t)h * dv + j], __fmul_rn(g, kv)), beta);
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
        for (int i = 0; i < 128; i += 4) {
            s[i] = fmaf(ks[i], delta, __fmul_rn(g, s[i]));
            s[i + 1] = fmaf(ks[i + 1], delta, __fmul_rn(g, s[i + 1]));
            s[i + 2] = fmaf(ks[i + 2], delta, __fmul_rn(g, s[i + 2]));
            s[i + 3] = fmaf(ks[i + 3], delta, __fmul_rn(g, s[i + 3]));
            b0 = fmaf(qs[i], s[i], b0);
            b1 = fmaf(qs[i + 1], s[i + 1], b1);
            b2 = fmaf(qs[i + 2], s[i + 2], b2);
            b3 = fmaf(qs[i + 3], s[i + 3], b3);
        }
        if (o) o[(size_t)t * nv * dv + (size_t)h * dv + j] = __fadd_rn(__fadd_rn(b0, b1), __fadd_rn(b2, b3));
        __syncwarp();
    }
#pragma unroll
    for (int i = 0; i < 128; i++) sp[(size_t)i * dv] = s[i];
}

// B: a block of DKW warps a (head, 32 columns); warp w holds key rows
// [w R, (w + 1) R), R = 128 / DKW.
template <int DKW>
__device__ __forceinline__ void gdn_scan_b_body(const float* __restrict__ cq, const float* __restrict__ gb, float* S,
                                                float* __restrict__ o, int n, int c_dim, int nv, int nk, int dv,
                                                float* __restrict__ ssave)
{
    constexpr int R = 128 / DKW;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int wpr = dv >> 5;
    const int h = blockIdx.x / wpr, j = (blockIdx.x - h * wpr) * 32 + lane;
    const int ko = h / (nv / nk), kd = nk * 128;
    const size_t base = ((size_t)h * 128 + w * R) * dv + j;
    float* sp = S + base;
    float s[R];
#pragma unroll
    for (int i = 0; i < R; i++) s[i] = sp[(size_t)i * dv];
    if (ssave) {
#pragma unroll
        for (int i = 0; i < R; i++) ssave[base + (size_t)i * dv] = s[i];
    }
    __shared__ float ks[128], qs[128];
    __shared__ float part[2][DKW][32];
    for (int t = 0; t < n; t++) {
        const float* row = cq + (size_t)t * c_dim;
        if (threadIdx.x < 32) *(float4*)(ks + lane * 4) = *(const float4*)(row + kd + (size_t)ko * 128 + lane * 4);
        else if (threadIdx.x < 64) *(float4*)(qs + lane * 4) = *(const float4*)(row + (size_t)ko * 128 + lane * 4);
        __syncthreads();
        const float g = gb[(size_t)t * 2 * nv + h], beta = gb[(size_t)t * 2 * nv + nv + h];
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int i = 0; i < R; i += 2) {
            a0 = fmaf(s[i], ks[w * R + i], a0);
            a1 = fmaf(s[i + 1], ks[w * R + i + 1], a1);
        }
        part[0][w][lane] = __fadd_rn(a0, a1);
        __syncthreads();
        float kv = part[0][0][lane];
#pragma unroll
        for (int q = 1; q < DKW; q++) kv = __fadd_rn(kv, part[0][q][lane]);
        const float delta = __fmul_rn(__fsub_rn(row[2 * kd + (size_t)h * dv + j], __fmul_rn(g, kv)), beta);
        float b0 = 0.0f, b1 = 0.0f;
#pragma unroll
        for (int i = 0; i < R; i += 2) {
            s[i] = fmaf(ks[w * R + i], delta, __fmul_rn(g, s[i]));
            s[i + 1] = fmaf(ks[w * R + i + 1], delta, __fmul_rn(g, s[i + 1]));
            b0 = fmaf(qs[w * R + i], s[i], b0);
            b1 = fmaf(qs[w * R + i + 1], s[i + 1], b1);
        }
        part[1][w][lane] = __fadd_rn(b0, b1);
        __syncthreads();
        if (w == 0 && o) {
            float ov = part[1][0][lane];
#pragma unroll
            for (int q = 1; q < DKW; q++) ov = __fadd_rn(ov, part[1][q][lane]);
            o[(size_t)t * nv * dv + (size_t)h * dv + j] = ov;
        }
        // part and ks/qs are rewritten next row only after this barrier
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < R; i++) sp[(size_t)i * dv] = s[i];
}

extern "C" __global__ void __launch_bounds__(64) gdn_scan_b2(const float* cq, const float* gb, float* S, float* o, int n,
                                                            int c_dim, int nv, int nk, int dv, float* ssave)
{
    gdn_scan_b_body<2>(cq, gb, S, o, n, c_dim, nv, nk, dv, ssave);
}
extern "C" __global__ void __launch_bounds__(128) gdn_scan_b4(const float* cq, const float* gb, float* S, float* o,
                                                             int n, int c_dim, int nv, int nk, int dv, float* ssave)
{
    gdn_scan_b_body<4>(cq, gb, S, o, n, c_dim, nv, nk, dv, ssave);
}
extern "C" __global__ void __launch_bounds__(256) gdn_scan_b8(const float* cq, const float* gb, float* S, float* o,
                                                             int n, int c_dim, int nv, int nk, int dv, float* ssave)
{
    gdn_scan_b_body<8>(cq, gb, S, o, n, c_dim, nv, nk, dv, ssave);
}

// The gates a block a projection row: blocks [0, nv) the a rows, [nv, 2 nv)
// the b rows; float4 loads of the weight row and the normed input.
extern "C" __global__ void gdn_gates2(const float* __restrict__ xn, int hidden, const float* __restrict__ wa,
                                      const float* __restrict__ wb, const float* __restrict__ a_log,
                                      const float* __restrict__ dt_bias, float* __restrict__ gb, int nv)
{
    __shared__ float red[32];
    const int r = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int NW = blockDim.x >> 5;
    const bool isb = r >= nv;
    const int h = isb ? r - nv : r;
    const float4* w4 = (const float4*)((isb ? wb : wa) + (size_t)h * hidden);
    const float4* x4 = (const float4*)(xn + (size_t)t * hidden);
    float s = 0.0f;
    for (int c = tid; c < (hidden >> 2); c += blockDim.x) {
        const float4 a = w4[c], x = x4[c];
        s = fmaf(a.w, x.w, fmaf(a.z, x.z, fmaf(a.y, x.y, fmaf(a.x, x.x, s))));
    }
    for (int q = 16; q > 0; q >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, q));
    if (lane == 0) red[warp] = s;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        for (int q = 0; q < NW; q++) v = __fadd_rn(v, red[q]);
        if (isb) {
            gb[(size_t)t * 2 * nv + nv + h] = 1.0f / (1.0f + expf(-v));
        } else {
            const float x2 = __fadd_rn(v, dt_bias[h]);
            const float sp = x2 > 20.0f ? x2 : log1pf(expf(x2));
            gb[(size_t)t * 2 * nv + h] = expf(-__fmul_rn(expf(a_log[h]), sp));
        }
    }
}
