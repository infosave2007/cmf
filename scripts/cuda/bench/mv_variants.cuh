// Experimental variants of the q4tp row loop for mv_bench (-DVARIANT=n).
//   1  compute only: every row reads row 0's bytes (no DRAM traffic)
//   2  no scale lookup (scale 1)
//   3  two accumulators per word pair
//   4  x as int16 fixed point, two products per __dp2a
//   5  lane owns two groups per iteration (more bytes in flight)
//   6  no shared staging: transposed x read from global memory
__device__ __forceinline__ float nib8b(u32 w, const float* p, int st, float& b, float a) {
    a += (float)(w & 15u) * p[0];
    b += (float)((w >> 4) & 15u) * p[st];
    a += (float)((w >> 8) & 15u) * p[2 * st];
    b += (float)((w >> 12) & 15u) * p[3 * st];
    a += (float)((w >> 16) & 15u) * p[4 * st];
    b += (float)((w >> 20) & 15u) * p[5 * st];
    a += (float)((w >> 24) & 15u) * p[6 * st];
    b += (float)(w >> 28) * p[7 * st];
    return a;
}

__global__ void mv_var(const MvJob* __restrict__ jobs, const float* __restrict__ x, int cols)
{
    extern __shared__ float sh[];
    const int T = blockDim.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = T >> 5;
    const MvJob jb = jobs[0];
    const int cpad = (cols + 31) & ~31;
    float* xs = sh + 16;
    float* gs = xs + cpad + 64;
    const int gpr = cols >> 5;
    const int GP = gpr | 1;
#if VARIANT == 4
    // int16 fixed point of x, pairs packed in one int: xi[k2 * GP + g] = (x[2k2+1] << 16) | x[2k2]
    int* xi = (int*)xs;
    float mx = 0.0f;
    for (int i = tid; i < cols; i += T) mx = fmaxf(mx, fabsf(x[i]));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) sh[warp] = mx;
    __syncthreads();
    mx = 0.0f;
    for (int w = 0; w < NW; w++) mx = fmaxf(mx, sh[w]);
    __syncthreads();
    const float q = 32767.0f / mx, iq = mx / 32767.0f;
    for (int i = tid; i < cols / 2; i += T) {
        const int a = (int)rintf(x[2 * i] * q), b = (int)rintf(x[2 * i + 1] * q);
        xi[(i & 15) * GP + (i >> 4)] = (b << 16) | (a & 0xffff);
    }
    __syncthreads();
    int* gi = (int*)gs;
    for (int g = tid; g < gpr; g += T) {
        int s = 0;
        for (int k = 0; k < 16; k++) { int v = xi[k * GP + g]; s += (short)(v & 0xffff) + (v >> 16); }
        gi[g] = 8 * s;
    }
    __syncthreads();
#else
    for (int i = tid; i < cols; i += T) xs[(i & 31) * GP + (i >> 5)] = x[i];
    __syncthreads();
    for (int g = tid; g < gpr; g += T) {
        const float* p = xs + g;
        float s = 0.0f;
        for (int k = 0; k < 32; k++) s += p[k * GP];
        gs[g] = 8.0f * s;
    }
    __syncthreads();
#endif
    const int row0 = (int)blockIdx.x * jb.rpb;
    const int row1 = min(row0 + jb.rpb, jb.rows);
    float* y = (float*)jb.y;
    for (int r = row0 + warp; r < row1; r += NW) {
        float acc = 0.0f;
#if VARIANT == 1
        const uint4* W = (const uint4*)jb.w;
#else
        const uint4* W = (const uint4*)jb.w + (size_t)r * gpr;
#endif
        const u8* C = (const u8*)jb.aux0 + (size_t)r * gpr;
        const float* L = (const float*)jb.aux1 + (size_t)r * 32;
#if VARIANT == 9
        // variant 7 with the staging stride a compile-time constant (GPC)
        const float lad = L[lane];
        const int iters = (gpr + 31) >> 5;
        if (GP == GPC) {
            for (int it = 0; it < iters; it++) {
                const int g = lane + (it << 5);
                float a = 0.0f, s0 = 0.0f;
                int c0 = 0;
                if (g < gpr) {
                    const uint4 w = W[g];
                    c0 = C[g];
                    const float* p = xs + g;
                    a = nib8(w.x, p, GPC, a);
                    a = nib8(w.y, p + 8 * GPC, GPC, a);
                    a = nib8(w.z, p + 16 * GPC, GPC, a);
                    a = nib8(w.w, p + 24 * GPC, GPC, a);
                    s0 = gs[g];
                }
                acc += __shfl_sync(0xffffffffu, lad, c0) * (a - s0);
            }
        }
#elif VARIANT == 7 || VARIANT == 8
        // The row's ladder in the lanes' registers: a scale is a shuffle, not
        // a second dependent load. Every lane runs every iteration (a shuffle
        // needs the whole warp at one program point); a lane past the row's
        // last group does no loads and adds nothing.
        const float lad = L[lane];
        const int iters = (gpr + 31) >> 5;
#if VARIANT == 8
        int it = 0;
        for (; it + 1 < iters; it += 2) {
            const int g0 = lane + (it << 5), g1 = g0 + 32;
            const bool ok0 = g0 < gpr, ok1 = g1 < gpr;
            float a = 0.0f, b = 0.0f, s0 = 0.0f, s1 = 0.0f;
            int c0 = 0, c1 = 0;
            if (ok0) {
                const uint4 w = W[g0];
                c0 = C[g0];
                const float* p = xs + g0;
                a = nib8(w.x, p, GP, a);
                a = nib8(w.y, p + 8 * GP, GP, a);
                a = nib8(w.z, p + 16 * GP, GP, a);
                a = nib8(w.w, p + 24 * GP, GP, a);
                s0 = gs[g0];
            }
            if (ok1) {
                const uint4 w = W[g1];
                c1 = C[g1];
                const float* p = xs + g1;
                b = nib8(w.x, p, GP, b);
                b = nib8(w.y, p + 8 * GP, GP, b);
                b = nib8(w.z, p + 16 * GP, GP, b);
                b = nib8(w.w, p + 24 * GP, GP, b);
                s1 = gs[g1];
            }
            const float sc0 = __shfl_sync(0xffffffffu, lad, c0);
            const float sc1 = __shfl_sync(0xffffffffu, lad, c1);
            acc += sc0 * (a - s0) + sc1 * (b - s1);
        }
        for (; it < iters; it++) {
#else
        for (int it = 0; it < iters; it++) {
#endif
            const int g = lane + (it << 5);
            const bool ok = g < gpr;
            float a = 0.0f, s0 = 0.0f;
            int c0 = 0;
            if (ok) {
                const uint4 w = W[g];
                c0 = C[g];
                const float* p = xs + g;
                a = nib8(w.x, p, GP, a);
                a = nib8(w.y, p + 8 * GP, GP, a);
                a = nib8(w.z, p + 16 * GP, GP, a);
                a = nib8(w.w, p + 24 * GP, GP, a);
                s0 = gs[g];
            }
            const float sc = __shfl_sync(0xffffffffu, lad, c0);
            acc += sc * (a - s0);
        }
#elif VARIANT == 5
        for (int g = 2 * lane; g < gpr; g += 64) {
            const uint4 w = W[g];
            const uint4 w2 = W[g + 1];
            const float* p = xs + g;
            float a = 0.0f, b = 0.0f;
            a = nib8(w.x, p, GP, a);
            a = nib8(w.y, p + 8 * GP, GP, a);
            a = nib8(w.z, p + 16 * GP, GP, a);
            a = nib8(w.w, p + 24 * GP, GP, a);
            b = nib8(w2.x, p + 1, GP, b);
            b = nib8(w2.y, p + 1 + 8 * GP, GP, b);
            b = nib8(w2.z, p + 1 + 16 * GP, GP, b);
            b = nib8(w2.w, p + 1 + 24 * GP, GP, b);
            acc += L[C[g]] * (a - gs[g]) + L[C[g + 1]] * (b - gs[g + 1]);
        }
#else
        for (int g = lane; g < gpr; g += 32) {
            const uint4 w = W[g];
#if VARIANT == 4
            const int* p = xi + g;
            int s = 0;
            const u32 ws[4] = {w.x, w.y, w.z, w.w};
            for (int q4 = 0; q4 < 4; q4++) {
                const u32 lo = ws[q4] & 0x0f0f0f0fu;         // nibbles 0,2,4,6 as bytes
                const u32 hi = (ws[q4] >> 4) & 0x0f0f0f0fu;  // nibbles 1,3,5,7
                // pair j of the word: weights (2j, 2j+1) = (lo byte j, hi byte j)
                for (int jj = 0; jj < 4; jj++) {
                    const u32 pair = ((lo >> (8 * jj)) & 0xffu) | (((hi >> (8 * jj)) & 0xffu) << 8);
                    s = __dp2a_lo(p[(q4 * 4 + jj) * GP], (int)pair, s);
                }
            }
            acc += L[C[g]] * ((float)(s - gi[g]) * iq);
#elif VARIANT == 3
            const float* p = xs + g;
            float a = 0.0f, b = 0.0f;
            a = nib8b(w.x, p, GP, b, a);
            a = nib8b(w.y, p + 8 * GP, GP, b, a);
            a = nib8b(w.z, p + 16 * GP, GP, b, a);
            a = nib8b(w.w, p + 24 * GP, GP, b, a);
            acc += L[C[g]] * ((a + b) - gs[g]);
#else
            const float* p = xs + g;
            float a = 0.0f;
            a = nib8(w.x, p, GP, a);
            a = nib8(w.y, p + 8 * GP, GP, a);
            a = nib8(w.z, p + 16 * GP, GP, a);
            a = nib8(w.w, p + 24 * GP, GP, a);
#if VARIANT == 2
            acc += a - gs[g];
#else
            acc += L[C[g]] * (a - gs[g]);
#endif
#endif
        }
#endif
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[r] = acc;
    }
}

static void mv_var_launch(int blocks, int threads, size_t shm, const MvJob* dj, const float* dx, int cols)
{
    static bool once = false;
    if (!once) { cudaFuncSetAttribute(mv_var, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 << 10); once = true; }
    mv_var<<<blocks, threads, shm>>>(dj, dx, cols);
}
