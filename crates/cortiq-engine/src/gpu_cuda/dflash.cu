// The DFlash 2 block on CUDA (`gpu_cuda/hyb/dflash.rs`): the stages the
// tensor-core matvec and cuBLAS do not do — the row convolution, the q / k
// head norm with the rotation, and the attention of a block row over its
// layer's context and the block's own rows. f32, the host block's
// arithmetic (`dflash::Drafter::block`), order aside.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

#define FULL 0xffffffffu
#define INF __int_as_float(0x7f800000)

// y[r][i] (+)= sum over o < min(kn, r + 1) of (base[stage][o][i] + d[r][stage][o][i / gs]) * x[r - o][i];
// `add`: onto what y holds (the residual stream), else in place of it.
extern "C" __global__ void df_conv(const float* __restrict__ x, const float* __restrict__ d,
                                   const float* __restrict__ base, float* __restrict__ y, int hs, int kn, int gs,
                                   int stage, int rows, int add, int pr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= hs || r >= rows) return;
    const int groups = hs / gs, g = i / gs;
    float acc = 0.0f;
    const int omax = min(kn, r + 1);
    for (int o = 0; o < omax; o++) {
        const int so = stage * kn + o;
        const float k = __fadd_rn(base[(size_t)so * hs + i], d[(size_t)r * pr + so * groups + g]);
        acc = fmaf(k, x[(size_t)(r - o) * hs + i], acc);
    }
    const size_t idx = (size_t)r * hs + i;
    y[idx] = add ? __fadd_rn(y[idx], acc) : acc;
}

// Rows [n][heads][hd] in place: the RMS over a head times w, then the NeoX
// rotation of position pos0 + row (when `rotary`). A block a (head, row),
// hd threads (hd <= 256).
extern "C" __global__ void df_qk(float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ inv,
                                 int heads, int hd, int pos0, int rotary, float eps)
{
    __shared__ float red[8];
    __shared__ float buf[256];
    const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float* p = x + ((size_t)r * heads + h) * hd;
    const float v = t < hd ? p[t] : 0.0f;
    float s = v * v;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < (int)(blockDim.x >> 5); k++) tot += red[k];
    const float sc = 1.0f / sqrtf(tot / (float)hd + eps);
    if (t < hd) buf[t] = v * sc * w[t];
    __syncthreads();
    if (t >= hd) return;
    if (!rotary) {
        p[t] = buf[t];
        return;
    }
    const int half = hd >> 1, i = t < half ? t : t - half;
    const float a = (float)(pos0 + r) * inv[i];
    const float c = cosf(a), sn = sinf(a);
    p[t] = t < half ? buf[t] * c - buf[t + half] * sn : buf[t] * c + buf[t - half] * sn;
}

// y [rows][nh][hd]: the query of (head h, row r) over the context rows
// [lo, lo + n) of ck / cv ([position - first][kv][hd]) and over the block's
// own rows kb / vb ([rows][kv][hd], every one of them). Row r sees the
// context from position anchor + r + 1 - window on (lo_abs is the position
// of context row lo; window 0: all of it). 128 threads a block; the scores
// in dynamic shared memory (n + rows floats).
extern "C" __global__ void df_attn(const float* __restrict__ q, const float* __restrict__ kb,
                                   const float* __restrict__ vb, const float* __restrict__ ck,
                                   const float* __restrict__ cv, float* __restrict__ y, int rows, int nh, int kv,
                                   int hd, int lo, int n, int lo_abs, int anchor, int window, float scale)
{
    extern __shared__ float sm[];
    __shared__ float qs[256];
    __shared__ float red[8];
    const int h = blockIdx.x, r = blockIdx.y, g = h / (nh / kv);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, NW = blockDim.x >> 5;
    for (int i = tid; i < hd; i += blockDim.x) qs[i] = q[((size_t)r * nh + h) * hd + i];
    __syncthreads();
    int m = 0;
    if (window > 0) {
        const int from = max(0, anchor + r + 1 - window);
        m = min(n, max(0, from - lo_abs));
    }
    const int total = n + rows;
    for (int j = warp; j < total; j += NW) {
        const float* kr = j < n ? ck + ((size_t)(lo + j) * kv + g) * hd : kb + ((size_t)(j - n) * kv + g) * hd;
        float s = 0.0f;
        for (int i = lane; i < hd; i += 32) s = fmaf(qs[i], kr[i], s);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
        if (lane == 0) sm[j] = j < m ? -INF : s * scale;
    }
    __syncthreads();
    float mx = -INF;
    for (int j = tid; j < total; j += blockDim.x) mx = fmaxf(mx, sm[j]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = red[0];
    for (int k = 1; k < NW; k++) mx = fmaxf(mx, red[k]);
    __syncthreads();
    float sum = 0.0f;
    for (int j = tid; j < total; j += blockDim.x) {
        const float e = sm[j] == -INF ? 0.0f : expf(sm[j] - mx);
        sm[j] = e;
        sum += e;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(FULL, sum, o);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    sum = 0.0f;
    for (int k = 0; k < NW; k++) sum += red[k];
    const float inv = 1.0f / sum;
    for (int i = tid; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < total; j++) {
            const float* vr = j < n ? cv + ((size_t)(lo + j) * kv + g) * hd : vb + ((size_t)(j - n) * kv + g) * hd;
            acc = fmaf(sm[j], vr[i], acc);
        }
        y[((size_t)r * nh + h) * hd + i] = acc * inv;
    }
}
