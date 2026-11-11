// The f32 scalar small-path matvec measured before the tensor-core kernel
// replaced it (kept for the microbenchmarks in this directory).
// ------------------------------------------------------------ small matvec
//
// y_j = (res_j +) W x_j for the rows j < N of x, several matrices a launch.
// A warp owns R rows of a matrix, a lane one 32-weight group of the slab:
// the input is staged into shared memory 1024 columns at a time (q4tp:
// element-major with an odd stride, so the 32 lanes of a load hit 32 banks)
// with each group's sum, and a lane takes the slab's x values into registers
// once for all R rows. Per (matrix row, input row) the sums run as one row of
// `mv_multi` runs them: the group's 32 products in order, minus 8 x the
// group's input sum, times the group's rung, groups in order, then the warp.

struct HyJob {
    u64 w;     // weight body
    u64 aux0;  // q4tp: codes
    u64 aux1;  // q4tp: ladder
    u64 y;     // output, row j at y + j * ystride
    u64 res;   // 0, or the rows added to the product
    int rows;
    int kind;  // 6 = q4tp, 4 = f32; bits 16.. rows a warp owns (q4tp)
    int blk0;  // first block of this job in the launch
    int ystride;
};

#define XSP 1056  // floats of one staged q4tp slab row: 32 x 33

// Stage slab `c0` of the N input rows: xs[j][k * 33 + g] = x[j][c0 + 32 g + k],
// every thread's float4 loads issued before any store.
template <int N>
__device__ __forceinline__ void stage_slab(float* __restrict__ xs, const float* __restrict__ x, int xstride, int c0,
                                           int ncol)
{
    const int tid = threadIdx.x;
    const int cc = (tid & 255) * 4;  // 256 threads x 4 columns = the slab
    float4 v[N];
#pragma unroll
    for (int j = 0; j < N; j++)
        v[j] = cc < ncol ? *(const float4*)(x + (size_t)j * xstride + c0 + cc) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int k = cc & 31, g = cc >> 5;
#pragma unroll
    for (int j = 0; j < N; j++) {
        float* d = xs + j * XSP + g;
        d[k * 33] = v[j].x;
        d[(k + 1) * 33] = v[j].y;
        d[(k + 2) * 33] = v[j].z;
        d[(k + 3) * 33] = v[j].w;
    }
}

template <int N, int R>
__device__ __forceinline__ void mvn_body(const HyJob* __restrict__ jobs, int njobs, const float* __restrict__ x,
                                         int xstride, int cols)
{
    extern __shared__ float sh[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int jx = 0;
    while (jx + 1 < njobs && (int)blockIdx.x >= jobs[jx + 1].blk0) jx++;
    const HyJob jb = jobs[jx];
    const int kind = jb.kind & 0xffff;
    // rows a warp owns for this matrix (<= R)
    const int rpw = min(R, max(1, jb.kind >> 16));
    float acc[R][N];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int j = 0; j < N; j++) acc[r][j] = 0.0f;
    float* y = (float*)jb.y;
    const float* res = (const float*)jb.res;
    const int row0 = ((int)blockIdx.x - jb.blk0) * (8 * rpw) + warp * rpw;
    float* xs = sh;  // [N][32 * 33]
    if (kind == 6) {
        const int gpr = cols >> 5;
        float lad[R];
        bool ok[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rr = row0 + r;
            ok[r] = r < rpw && rr < jb.rows;
            lad[r] = ok[r] ? ((const float*)jb.aux1)[(size_t)rr * 32 + lane] : 0.0f;
        }
        const int iters = (gpr + 31) >> 5;
        for (int it = 0; it < iters; it++) {
            const int c0 = it << 10;
            const int ncol = min(1024, cols - c0);
            const int g = (it << 5) + lane;
            const bool live = g < gpr;
            // the slab's weights first: their loads fly while the input stages
            uint4 w[R];
            int code[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (live && ok[r]) {
                    w[r] = ((const uint4*)jb.w)[(size_t)(row0 + r) * gpr + g];
                    code[r] = ((const u8*)jb.aux0)[(size_t)(row0 + r) * gpr + g];
                } else {
                    w[r] = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
                    code[r] = 0;
                }
            }
            __syncthreads();
            stage_slab<N>(xs, x, xstride, c0, ncol);
            __syncthreads();
            float sc[R];
#pragma unroll
            for (int r = 0; r < R; r++) sc[r] = __shfl_sync(FULL, lad[r], code[r]);
            const float* xp = xs + lane;
#pragma unroll
            for (int q = 0; q < 4; q++) {
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int k = q * 8 + e;
                    float xv[N];
#pragma unroll
                    for (int j = 0; j < N; j++) xv[j] = xp[j * XSP + k * 33];
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        const u32 wd = q == 0 ? w[r].x : q == 1 ? w[r].y : q == 2 ? w[r].z : w[r].w;
                        // (nibble - 8) * rung: 2^23 + nib as a float's bits,
                        // minus 2^23 + 8, exactly; then the row's scale
                        const float wv = __fmul_rn(
                            __fsub_rn(__uint_as_float(((wd >> (4 * e)) & 15u) | 0x4B000000u), 8388616.0f), sc[r]);
#pragma unroll
                        for (int j = 0; j < N; j++) acc[r][j] = fmaf(wv, xv[j], acc[r][j]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rr = row0 + r;
#pragma unroll
            for (int j = 0; j < N; j++) {
                float v = acc[r][j];
                for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
                if (lane == 0 && ok[r]) {
                    const size_t at = (size_t)j * jb.ystride + rr;
                    y[at] = res ? __fadd_rn(res[at], v) : v;
                }
            }
        }
    } else {
        // f32 (a small matrix: GDN gates): a row a warp, the slab staged as
        // the q4tp one (element k of group g at k * 33 + g)
        const int rr = row0;
        const int iters = (cols + 1023) >> 10;
        for (int it = 0; it < iters; it++) {
            const int c0 = it << 10;
            const int ncol = min(1024, cols - c0);
            __syncthreads();
            stage_slab<N>(xs, x, xstride, c0, ncol);
            __syncthreads();
            if (rr < jb.rows) {
                const float* W = (const float*)jb.w + (size_t)rr * cols + c0;
                for (int c = lane; c < ncol; c += 32) {
                    const float wv = W[c];
                    const int off = (c & 31) * 33 + (c >> 5);
#pragma unroll
                    for (int j = 0; j < N; j++) acc[0][j] = fmaf(wv, xs[j * XSP + off], acc[0][j]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < N; j++) {
            float v = acc[0][j];
            for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
            if (lane == 0 && rr < jb.rows) {
                const size_t at = (size_t)j * jb.ystride + rr;
                y[at] = res ? __fadd_rn(res[at], v) : v;
            }
        }
    }
}

#define MVN(NN, RR)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) hy_mvn_##NN(const HyJob* __restrict__ jobs, int njobs, \
                                                                 const float* __restrict__ x, int xstride,   \
                                                                 int cols)                                   \
    {                                                                                                        \
        mvn_body<NN, RR>(jobs, njobs, x, xstride, cols);                                                     \
    }
MVN(1, 8)
MVN(2, 8)
MVN(3, 8)
MVN(4, 8)
MVN(5, 8)
MVN(6, 8)
MVN(7, 8)
MVN(8, 8)
MVN(12, 4)
MVN(16, 4)

