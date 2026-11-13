// A vision tower's encoder on CUDA (`gpu::vit_encode`, `gpu::sam_encode`):
// `gpu_wgpu/vit.rs`'s kernels, f32 throughout; the products are cuBLAS's.

typedef unsigned int u32;

// LayerNorm with bias, a block of 256 a row (two passes, as `vt_ln`).
extern "C" __global__ void vt_ln(const float* __restrict__ src, const float* __restrict__ w, const float* __restrict__ b,
                                 float* __restrict__ dst, int rows, int cols, float eps)
{
    __shared__ float red[256];
    const int row = blockIdx.x + blockIdx.y * 65535, t = threadIdx.x;
    if (row >= rows) return;
    const float* x = src + (size_t)row * cols;
    float s = 0.0f;
    for (int i = t; i < cols; i += 256) s += x[i];
    red[t] = s;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float mean = red[0] / (float)cols;
    __syncthreads();
    float v = 0.0f;
    for (int i = t; i < cols; i += 256) {
        const float d = x[i] - mean;
        v += d * d;
    }
    red[t] = v;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)cols + eps);
    float* y = dst + (size_t)row * cols;
    for (int i = t; i < cols; i += 256) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// erf by Abramowitz–Stegun 7.1.26 (|error| <= 1.5e-7), as `vt_bias_act`
__device__ __forceinline__ float erf_as(float x) {
    const float s = x < 0.0f ? -1.0f : 1.0f, a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return s * (1.0f - poly * expf(-a * a));
}

// y = act(y + b), act 0 none, 1 erf GELU, 2 tanh GELU, 3 quick GELU.
extern "C" __global__ void vt_bias_act(float* __restrict__ y, const float* __restrict__ b, unsigned long long total, int cols,
                                       int act)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = y[i] + b[i % cols];
    if (act == 1) v = 0.5f * v * (1.0f + erf_as(v * 0.70710678118654752f));
    else if (act == 2) {
        const float u = fminf(fmaxf(0.7978845608f * (v + 0.044715f * v * v * v), -15.0f), 15.0f);
        v = 0.5f * v * (1.0f + tanhf(u));
    } else if (act == 3) v = v / (1.0f + expf(-1.702f * v));
    y[i] = v;
}

// x += y + b (a residual add).
extern "C" __global__ void vt_add(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b,
                                  unsigned long long total, int cols)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    x[i] = x[i] + y[i] + b[i % cols];
}

// qkv rows [n][3 nh hd] into head-major planes [nh][n][hd], q and k rotated by
// cs / sn [n][hd] (none: as they are): `halves` pairs lane j with j +- hd/2,
// else with its neighbour.
extern "C" __global__ void vt_split(const float* __restrict__ qkv, const float* __restrict__ cs, const float* __restrict__ sn,
                                    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v, int n, int nh, int hd,
                                    int halves)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)nh * n * hd) return;
    const int j = (int)(i % hd);
    const int pp = (int)((i / hd) % n);
    const int h = (int)(i / ((unsigned long long)n * hd));
    const int d = nh * hd;
    const size_t base = (size_t)pp * 3 * d + (size_t)h * hd;
    if (!cs) {
        q[i] = qkv[base + j];
        k[i] = qkv[base + d + j];
        v[i] = qkv[base + 2 * d + j];
        return;
    }
    int pj;
    float sg = 1.0f;
    if (!halves) {
        pj = j ^ 1;
        if ((j & 1) == 0) sg = -1.0f;
    } else {
        const int half = hd / 2;
        if (j < half) {
            pj = j + half;
            sg = -1.0f;
        } else pj = j - half;
    }
    const float c = cs[(size_t)pp * hd + j], s = sn[(size_t)pp * hd + j];
    q[i] = qkv[base + j] * c + sg * qkv[base + pj] * s;
    k[i] = qkv[base + d + j] * c + sg * qkv[base + d + pj] * s;
    v[i] = qkv[base + 2 * d + j];
}

// SAM's window partition: `rows` window rows from the grid g x g, zero padded.
extern "C" __global__ void sam_part(const float* __restrict__ px, float* __restrict__ py, int rows, int d, int g, int w)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)rows * d) return;
    const int r = (int)(i / d), e = (int)(i % d);
    const int wn = (g + w - 1) / w;
    const int win = r / (w * w), local = r % (w * w);
    const int y = (win / wn) * w + local / w, x = (win % wn) * w + local % w;
    py[i] = (y >= g || x >= g) ? 0.0f : px[((size_t)y * g + x) * d + e];
}

// The inverse, added to the grid with a bias.
extern "C" __global__ void sam_unpart(float* __restrict__ ux, const float* __restrict__ uy, const float* __restrict__ ub, int n,
                                      int d, int g, int w)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned long long)n * d) return;
    const int r = (int)(i / d), e = (int)(i % d);
    const int wn = (g + w - 1) / w;
    const int y = r / g, x = r % g;
    const int win = (y / w) * wn + x / w;
    const int src = win * w * w + (y % w) * w + x % w;
    ux[i] = ux[i] + uy[(size_t)src * d + e] + ub[e];
}

// Windowed attention with decomposed relative positions: a thread a (head,
// window row); head-major planes [nh][rows][hd] in, [rows][nh hd] out.
// Windows of at most 16 x 16 rows.
extern "C" __global__ void sam_win(const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv,
                                   const float* __restrict__ wrel, float* __restrict__ wo, int rows, int g, int s, int nh, int hd,
                                   float scale)
{
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int h = (int)(idx / rows), r = (int)(idx % rows);
    if (h >= nh) return;
    const int n = s * s;
    const int win = r / n, qi = r % n;
    const int wn = (g + s - 1) / s;
    if ((win / wn) * s + qi / s >= g || (win % wn) * s + qi % s >= g) return;
    const float* q = wq + ((size_t)h * rows + r) * hd;
    const float* kb = wk + ((size_t)h * rows + (size_t)win * n) * hd;
    const float* vb = wv + ((size_t)h * rows + (size_t)win * n) * hd;
    float rh[32], rw[32];
    const int qy = qi / s, qx = qi % s, rl = 2 * s - 1;
    for (int k = 0; k < s; k++) {
        const float* ry = wrel + (size_t)(qy + s - 1 - k) * hd;
        const float* rx = wrel + (size_t)(rl + qx + s - 1 - k) * hd;
        float a = 0.0f, b = 0.0f;
        for (int e = 0; e < hd; e++) {
            a = fmaf(q[e], ry[e], a);
            b = fmaf(q[e], rx[e], b);
        }
        rh[k] = a;
        rw[k] = b;
    }
    float sc[256];
    float mx = -1.0e30f;
    for (int k = 0; k < n; k++) {
        const float* kr = kb + (size_t)k * hd;
        float a = 0.0f;
        for (int e = 0; e < hd; e++) a = fmaf(q[e], kr[e], a);
        const float v = a * scale + rh[k / s] + rw[k % s];
        sc[k] = v;
        mx = fmaxf(mx, v);
    }
    float den = 0.0f;
    for (int k = 0; k < n; k++) {
        sc[k] = expf(sc[k] - mx);
        den += sc[k];
    }
    const float inv = 1.0f / den;
    float* o = wo + (size_t)r * nh * hd + (size_t)h * hd;
    for (int e = 0; e < hd; e++) {
        float a = 0.0f;
        for (int k = 0; k < n; k++) a = fmaf(sc[k], vb[(size_t)k * hd + e], a);
        o[e] = a * inv;
    }
}

// One head's score rows S [n][n] into probabilities in place, a block a row;
// with `rel` (SAM's global blocks, [2s - 1 h; 2s - 1 w][hd], s x s = n) the
// decomposed relative bias of the row's query (q: the head's plane [n][hd]).
extern "C" __global__ void vt_softmax(float* __restrict__ S, int n, const float* __restrict__ q, const float* __restrict__ rel,
                                      int s, int hd)
{
    __shared__ float red[256];
    __shared__ float rh[64], rw[64];
    const int row = blockIdx.x + blockIdx.y * 65535, t = threadIdx.x;
    if (row >= n) return;
    float* x = S + (size_t)row * n;
    if (rel) {
        const int qy = row / s, qx = row % s, rl = 2 * s - 1;
        const float* qr = q + (size_t)row * hd;
        for (int kk = t; kk < 2 * s; kk += 256) {
            const int k = kk % s;
            const float* rt = kk < s ? rel + (size_t)(qy + s - 1 - k) * hd : rel + (size_t)(rl + qx + s - 1 - k) * hd;
            float a = 0.0f;
            for (int e = 0; e < hd; e++) a = fmaf(qr[e], rt[e], a);
            if (kk < s) rh[k] = a;
            else rw[k] = a;
        }
        __syncthreads();
        for (int i = t; i < n; i += 256) x[i] += rh[i / s] + rw[i % s];
        __syncthreads();
    }
    float m = -1.0e30f;
    for (int i = t; i < n; i += 256) m = fmaxf(m, x[i]);
    red[t] = m;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] = fmaxf(red[t], red[t + st]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = t; i < n; i += 256) {
        const float e = expf(x[i] - mx);
        x[i] = e;
        sum += e;
    }
    red[t] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int i = t; i < n; i += 256) x[i] *= inv;
}

// A bias-free conv over an HWC grid g x g, weights [co][k k][ci]: a thread an
// output (cell, channel).
extern "C" __global__ void sam_conv(const float* __restrict__ cw, const float* __restrict__ cx, float* __restrict__ cy, int g,
                                    int ci, int co, int k, int stride, int pad, int g2)
{
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned long long)g2 * g2 * co) return;
    const int cell = (int)(idx / co), c = (int)(idx % co);
    const int oy = cell / g2, ox = cell % g2;
    float acc = 0.0f;
    for (int ky = 0; ky < k; ky++) {
        const int iy = oy * stride + ky - pad;
        if (iy < 0 || iy >= g) continue;
        for (int kx = 0; kx < k; kx++) {
            const int ix = ox * stride + kx - pad;
            if (ix < 0 || ix >= g) continue;
            const float* xr = cx + ((size_t)iy * g + ix) * ci;
            const float* wr = cw + ((size_t)c * k * k + ky * k + kx) * ci;
            for (int e = 0; e < ci; e++) acc = fmaf(wr[e], xr[e], acc);
        }
    }
    cy[idx] = acc;
}
