// The GigaAM encoder on CUDA (`gpu::gigaam_encode`): the kernels between
// cuBLAS products — biases and activations, the row norms, the rotary split
// into head planes, the softmax, the gates, the convolution module's tail.
// f32 throughout.

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// y += b a row; act 1: SiLU after.
extern "C" __global__ void ga_bias_act(float* __restrict__ y, const float* __restrict__ b, unsigned long long total, int cols,
                                       int has_bias, int act)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = y[i];
    if (has_bias) v += b[i % cols];
    if (act == 1) v = v * sigm(v);
    y[i] = v;
}

// dst = a * src (dst may be src).
extern "C" __global__ void ga_scale(float* dst, const float* src, unsigned long long total, float a)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = a * src[i];
}

// dst += a * src.
extern "C" __global__ void ga_axpy(float* __restrict__ dst, const float* __restrict__ src, unsigned long long total, float a)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = fmaf(a, src[i], dst[i]);
}

// dst = src for the first `valid` elements, zero after (a fold of frame pairs
// with an odd last frame, and the copy back).
extern "C" __global__ void ga_copy_pad(float* __restrict__ dst, const float* __restrict__ src, unsigned long long total,
                                       unsigned long long valid)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = i < valid ? src[i] : 0.0f;
}

// dst = (res + mix * (s - res)) * rsqrt(mean(.^2) + eps) * w, a block of 256
// a row; mix 1 with res == s is the plain row norm. `scalar_w`: w[0] for
// every lane (the output norm).
extern "C" __global__ void ga_rms(const float* __restrict__ res, const float* __restrict__ s, const float* __restrict__ w,
                                  float* __restrict__ dst, int rows, int cols, float eps, float mix, int scalar_w)
{
    __shared__ float red[256];
    const int row = blockIdx.x + blockIdx.y * 65535, t = threadIdx.x;
    if (row >= rows) return;
    const float* r = res + (size_t)row * cols;
    const float* x = s + (size_t)row * cols;
    float acc = 0.0f;
    for (int i = t; i < cols; i += 256) {
        const float v = r[i] + mix * (x[i] - r[i]);
        acc = fmaf(v, v, acc);
    }
    red[t] = acc;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)cols + eps);
    float* y = dst + (size_t)row * cols;
    for (int i = t; i < cols; i += 256) {
        const float v = r[i] + mix * (x[i] - r[i]);
        y[i] = v * inv * (scalar_w ? w[0] : w[i]);
    }
}

// qk rows [n][2 nh hd] and vg rows [n][3 nh hd] into head-major planes
// [nh][n][hd]: q and k rotated in neighbour pairs by cs / sn [n][hd / 2],
// v multiplied by the sigmoid of its input gate (the second third of vg).
extern "C" __global__ void ga_split(const float* __restrict__ qk, const float* __restrict__ vg, const float* __restrict__ cs,
                                    const float* __restrict__ sn, float* __restrict__ q, float* __restrict__ k,
                                    float* __restrict__ v, int n, int nh, int hd)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int half = hd / 2, hw = nh * hd;
    if (i >= (unsigned long long)nh * n * half) return;
    const int p = (int)(i % half);
    const int row = (int)((i / half) % n);
    const int h = (int)(i / ((unsigned long long)half * n));
    const float c = cs[(size_t)row * half + p], s = sn[(size_t)row * half + p];
    const size_t dst = ((size_t)h * n + row) * hd + 2 * p;
    const float* qr = qk + (size_t)row * 2 * hw + h * hd + 2 * p;
    const float* kr = qr + hw;
    q[dst] = qr[0] * c - qr[1] * s;
    q[dst + 1] = qr[0] * s + qr[1] * c;
    k[dst] = kr[0] * c - kr[1] * s;
    k[dst + 1] = kr[0] * s + kr[1] * c;
    const float* vr = vg + (size_t)row * 3 * hw + h * hd + 2 * p;
    v[dst] = vr[0] * sigm(vr[hw]);
    v[dst + 1] = vr[1] * sigm(vr[hw + 1]);
}

// Softmax over each row of [rows][n] in place, a block of 256 a row.
extern "C" __global__ void ga_softmax(float* __restrict__ s, int rows, int n)
{
    __shared__ float red[256];
    const int row = blockIdx.x + blockIdx.y * 65535, t = threadIdx.x;
    if (row >= rows) return;
    float* x = s + (size_t)row * n;
    float m = -3.0e38f;
    for (int i = t; i < n; i += 256) m = fmaxf(m, x[i]);
    red[t] = m;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] = fmaxf(red[t], red[t + st]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();
    float acc = 0.0f;
    for (int i = t; i < n; i += 256) {
        const float e = expf(x[i] - mx);
        x[i] = e;
        acc += e;
    }
    red[t] = acc;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int i = t; i < n; i += 256) x[i] *= inv;
}

// The attention context from head-major planes [nh][n][hd] to rows
// [n][nh hd], times the sigmoid of the output gate (the last third of vg).
extern "C" __global__ void ga_gate(const float* __restrict__ o, const float* __restrict__ vg, float* __restrict__ ctx, int n,
                                   int nh, int hd)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int hw = nh * hd;
    if (i >= (unsigned long long)n * hw) return;
    const int col = (int)(i % hw), row = (int)(i / hw);
    const int h = col / hd, e = col % hd;
    ctx[i] = o[((size_t)h * n + row) * hd + e] * sigm(vg[(size_t)row * 3 * hw + 2 * hw + col]);
}

// mean[c] = mean_w[c] / n * sum over frames of cg[frame][c] (cg rows are
// 2 d wide: the stream, then its gate), a block of 256 a channel.
extern "C" __global__ void ga_mean(const float* __restrict__ cg, const float* __restrict__ mean_w, float* __restrict__ mean,
                                   int n, int d)
{
    __shared__ float red[256];
    const int c = blockIdx.x, t = threadIdx.x;
    if (c >= d) return;
    float acc = 0.0f;
    for (int i = t; i < n; i += 256) acc += cg[(size_t)i * 2 * d + c];
    red[t] = acc;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    if (t == 0) mean[c] = red[0] / (float)n * mean_w[c];
}

// z[frame][c] = silu((conv + mean[c]) * sigmoid(gate)): the depthwise
// convolution over time with taps [k][d] and (k - 1) / 2 zero frames on both
// sides, of the stream half of cg.
extern "C" __global__ void ga_conv(const float* __restrict__ cg, const float* __restrict__ taps, const float* __restrict__ bias,
                                   const float* __restrict__ mean, float* __restrict__ z, int n, int d, int k)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)n * d) return;
    const int c = (int)(i % d), row = (int)(i / d);
    const int half = (k - 1) / 2;
    float acc = bias[c];
    for (int j = 0; j < k; j++) {
        const int src = row + j - half;
        if (src >= 0 && src < n) acc = fmaf(taps[(size_t)j * d + c], cg[(size_t)src * 2 * d + c], acc);
    }
    const float v = (acc + mean[c]) * sigm(cg[(size_t)row * 2 * d + d + c]);
    z[i] = v * sigm(v);
}
