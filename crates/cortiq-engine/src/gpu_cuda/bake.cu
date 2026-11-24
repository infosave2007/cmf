// The skill bake's GDN layers on the card (`fcd.rs` gdn_attn_fwd/bwd): the
// frozen operator of `fcd_ops` (gdn_conv_*, gdn_group_*) in f32. Rows are
// sequences of `t` positions, `[b][t]`; a block runs one (sequence, v-head),
// thread j one column j of the delta-rule state S [DK][DV] (S[i][j] for every
// key dimension i in its registers): the decay, the delta update and o = Sᵀq̂
// are local to a column, the norms and the key/query gradients reduce across
// the block.

#ifndef DK
#define DK 128
#endif
#ifndef DV
#define DV 128
#endif

__device__ __forceinline__ float bk_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float bk_silu_bwd(float x) {
    float s = 1.f / (1.f + __expf(-x));
    return s * (1.f + x * (1.f - s));
}
__device__ __forceinline__ float bk_softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }
__device__ __forceinline__ float bk_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

// Σ v over the block (blockDim a multiple of 32); every thread gets it.
__device__ __forceinline__ float bk_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
    const int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; i++) s += red[i];
    return s;
}

// Depthwise causal conv + SiLU: pre = Σ_j raw[ti-(kk-1)+j]·tap[c][j], cq = silu(pre).
extern "C" __global__ void bk_conv_fwd(const float* raw, const float* taps, float* pre, float* cq, int n, int t,
                                       int cdim, int kk) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * cdim) return;
    const int r = (int)(idx / cdim), c = (int)(idx % cdim), ti = r % t, base = r - ti;
    float acc = 0.f;
    for (int j = 0; j < kk; j++) {
        const int p = ti - (kk - 1) + j;
        if (p >= 0) acc += raw[(long)(base + p) * cdim + c] * taps[c * kk + j];
    }
    pre[idx] = acc;
    cq[idx] = bk_silu(acc);
}

// Conv + SiLU backward: draw[p][c] = Σ_j dcq[p+(kk-1)-j][c]·silu'(pre)·tap[c][j].
extern "C" __global__ void bk_conv_bwd(const float* pre, const float* dcq, const float* taps, float* draw, int n,
                                       int t, int cdim, int kk) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * cdim) return;
    const int r = (int)(idx / cdim), c = (int)(idx % cdim), p = r % t, base = r - p;
    float acc = 0.f;
    for (int j = 0; j < kk; j++) {
        const int ti = p + (kk - 1) - j;
        if (ti < t) {
            const long q = (long)(base + ti) * cdim + c;
            acc += dcq[q] * bk_silu_bwd(pre[q]) * taps[c * kk + j];
        }
    }
    draw[idx] = acc;
}

// The per-position scalars of head h at row `row`: q̂/k̂ scales and the gates.
struct BkStep {
    float invq, invk, nq2, nk2, g, beta, arg;
};

__device__ __forceinline__ BkStep bk_step(const float* cr, const float* a, const float* bb, long row, int h, int nv,
                                          int ko, int kd, float ea, float dtb, float* qs, float* ks, float* red) {
    const int j = threadIdx.x;
    for (int i = j; i < DK; i += DV) {
        qs[i] = cr[ko * DK + i];
        ks[i] = cr[kd + ko * DK + i];
    }
    __syncthreads();
    float pq = 0.f, pk = 0.f;
    for (int i = j; i < DK; i += DV) {
        pq += qs[i] * qs[i];
        pk += ks[i] * ks[i];
    }
    BkStep s;
    s.nq2 = bk_block_sum(pq, red) + 1e-6f;
    s.nk2 = bk_block_sum(pk, red) + 1e-6f;
    s.invq = 1.f / (sqrtf(s.nq2) * sqrtf((float)DK));
    s.invk = 1.f / sqrtf(s.nk2);
    s.arg = a[row * nv + h] + dtb;
    s.g = __expf(-ea * bk_softplus(s.arg));
    s.beta = bk_sigmoid(bb[row * nv + h]);
    return s;
}

// Forward: grid (nv, sequences), block DV. out[row][h·DV + j] = rms(o)·w·silu(z).
extern "C" __global__ void bk_gdn_fwd(const float* cq, const float* z, const float* a, const float* bb,
                                      const float* a_log, const float* dt_bias, const float* normw, float* out, int t,
                                      int nv, int nk, float eps, int seq0) {
    const int h = blockIdx.x, bi = seq0 + blockIdx.y, j = threadIdx.x;
    const int rep = nv / nk, ko = h / rep, kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    __shared__ float qs[DK], ks[DK], red[32];
    float S[DK];
#pragma unroll
    for (int i = 0; i < DK; i++) S[i] = 0.f;
    const float ea = __expf(a_log[h]), dtb = dt_bias[h], wn = normw[j];
    for (int ti = 0; ti < t; ti++) {
        const long row = (long)bi * t + ti;
        const float* cr = cq + row * cdim;
        const BkStep s = bk_step(cr, a, bb, row, h, nv, ko, kd, ea, dtb, qs, ks, red);
        const float vj = cr[2 * kd + h * DV + j];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < DK; i++) {
            S[i] *= s.g;
            kv += S[i] * ks[i] * s.invk;
        }
        const float u = s.beta * (vj - kv);
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < DK; i++) {
            S[i] += ks[i] * s.invk * u;
            o += qs[i] * s.invq * S[i];
        }
        const float ss = bk_block_sum(o * o, red);
        const float inv = rsqrtf(ss / DV + eps);
        out[row * vd + h * DV + j] = o * inv * wn * bk_silu(z[row * vd + h * DV + j]);
        __syncthreads();
    }
}

// Backward (BPTT): grid (nv, sequences of this launch), block DV. The replay
// keeps S after every position in `hist` ([unit][t+1][DK][DV], unit = local
// sequence · nv + h), kv and o per position, the gates; the reverse sweep carries
// dS [DK] per column. dcq must be zero on entry (q/k channels of a group are
// shared by its v-heads: atomic adds); dz, da, db, dcq's v channels are stored.
extern "C" __global__ void bk_gdn_bwd(const float* cq, const float* z, const float* a, const float* bb,
                                      const float* a_log, const float* dt_bias, const float* normw, const float* dout,
                                      float* dcq, float* dz, float* da, float* db, float* hist, float* kvh, float* oh,
                                      int t, int nv, int nk, float eps, int seq0) {
    const int h = blockIdx.x, lb = blockIdx.y, bi = seq0 + lb, j = threadIdx.x;
    const int rep = nv / nk, ko = h / rep, kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    __shared__ float qs[DK], ks[DK], red[32], accq[DK], acck[DK];
    const long unit = (long)lb * nv + h;
    float* H = hist + unit * (long)(t + 1) * DK * DV;
    float* KV = kvh + unit * (long)t * DV;
    float* O = oh + unit * (long)t * DV;
    const float ea = __expf(a_log[h]), dtb = dt_bias[h], wn = normw[j];

    // replay
    float S[DK];
#pragma unroll
    for (int i = 0; i < DK; i++) {
        S[i] = 0.f;
        H[(long)i * DV + j] = 0.f;
    }
    for (int ti = 0; ti < t; ti++) {
        const long row = (long)bi * t + ti;
        const float* cr = cq + row * cdim;
        const BkStep s = bk_step(cr, a, bb, row, h, nv, ko, kd, ea, dtb, qs, ks, red);
        const float vj = cr[2 * kd + h * DV + j];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < DK; i++) {
            S[i] *= s.g;
            kv += S[i] * ks[i] * s.invk;
        }
        const float u = s.beta * (vj - kv);
        float o = 0.f;
        float* Hn = H + (long)(ti + 1) * DK * DV;
#pragma unroll
        for (int i = 0; i < DK; i++) {
            S[i] += ks[i] * s.invk * u;
            o += qs[i] * s.invq * S[i];
            Hn[(long)i * DV + j] = S[i];
        }
        KV[(long)ti * DV + j] = kv;
        O[(long)ti * DV + j] = o;
        __syncthreads();
    }

    // reverse sweep
    float dS[DK];
#pragma unroll
    for (int i = 0; i < DK; i++) dS[i] = 0.f;
    const int lane = j & 31;
    for (int ti = t - 1; ti >= 0; ti--) {
        const long row = (long)bi * t + ti;
        const float* cr = cq + row * cdim;
        for (int i = j; i < DK; i += DV) {
            accq[i] = 0.f;
            acck[i] = 0.f;
        }
        const BkStep s = bk_step(cr, a, bb, row, h, nv, ko, kd, ea, dtb, qs, ks, red);
        const float vj = cr[2 * kd + h * DV + j];
        const float kvj = KV[(long)ti * DV + j], oj = O[(long)ti * DV + j];
        const float* St = H + (long)(ti + 1) * DK * DV;
        const float* Sp = H + (long)ti * DK * DV;
        // 1. gated RMSNorm
        const float ss = bk_block_sum(oj * oj, red);
        const float inv = rsqrtf(ss / DV + eps);
        const float zv = z[row * vd + h * DV + j];
        const float weff = wn * bk_silu(zv);
        const float dof = dout[row * vd + h * DV + j];
        const float sdot = bk_block_sum(dof * weff * oj, red);
        dz[row * vd + h * DV + j] = dof * oj * inv * wn * bk_silu_bwd(zv);
        const float doj = inv * weff * dof - oj * (sdot / DV) * inv * inv * inv;
        // 2./3. dS += q̂ ⊗ do; dq̂ = S_t·do (across columns); du = dSᵀk̂ (local)
        float du = 0.f;
#pragma unroll
        for (int i = 0; i < DK; i++) {
            dS[i] += qs[i] * s.invq * doj;
            du += dS[i] * ks[i] * s.invk;
            float v = St[(long)i * DV + j] * doj;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) atomicAdd(&accq[i], v);
        }
        const float vmk = vj - kvj;
        const float dbeta = bk_block_sum(du * vmk, red);
        dcq[row * cdim + 2 * kd + h * DV + j] = s.beta * du;
        const float dkv = -(s.beta * du);
        // 3.-5. dk̂ = dS·u + S_pre·dkv (across columns); dg = ⟨dS_pre, S_{t-1}⟩; dS ← g·dS_pre
        float dgp = 0.f;
#pragma unroll
        for (int i = 0; i < DK; i++) {
            const float sp = Sp[(long)i * DV + j];
            float v = dS[i] * vmk * s.beta + sp * s.g * dkv;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) atomicAdd(&acck[i], v);
            const float dspre = dS[i] + ks[i] * s.invk * dkv;
            dgp += dspre * sp;
            dS[i] = s.g * dspre;
        }
        const float dg = bk_block_sum(dgp, red);
        if (j == 0) {
            da[row * nv + h] = dg * s.g * (-ea) * bk_sigmoid(s.arg);
            db[row * nv + h] = dbeta * s.beta * (1.f - s.beta);
        }
        // 7. l2-norm through-grads into the group's shared q/k channels
        float pq = 0.f, pk = 0.f;
        for (int i = j; i < DK; i += DV) {
            pq += accq[i] * qs[i];
            pk += acck[i] * ks[i];
        }
        const float qdot = bk_block_sum(pq, red), kdot = bk_block_sum(pk, red);
        for (int i = j; i < DK; i += DV) {
            atomicAdd(&dcq[row * cdim + ko * DK + i], s.invq * accq[i] - qs[i] * qdot * s.invq / s.nq2);
            atomicAdd(&dcq[row * cdim + kd + ko * DK + i], s.invk * acck[i] - ks[i] * kdot * s.invk / s.nk2);
        }
        __syncthreads();
    }
}

// ── the bake's frozen FFN (fcd.rs layer_forward_scaled / skillbake.rs backward) ──

__device__ __forceinline__ float bk_act(float x, int gelu) {
    return gelu ? 0.5f * x * (1.f + erff(x * 0.70710678118654752f)) : bk_silu(x);
}
__device__ __forceinline__ float bk_act_bwd(float x, int gelu) {
    if (gelu) {
        const float cdf = 0.5f * (1.f + erff(x * 0.70710678118654752f));
        const float pdf = __expf(-0.5f * x * x) * 0.3989422804014327f;
        return cdf + x * pdf;
    }
    return bk_silu_bwd(x);
}

// act[r][c] = act(g)·u·scale[c] over both[r] = (gate | up), 2·inter wide.
extern "C" __global__ void bk_ffn_act(const float* both, const float* scale, float* act, int n, int inter, int gelu,
                                      int has_scale) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * inter) return;
    const int r = (int)(idx / inter), c = (int)(idx % inter);
    const long i = (long)r * 2 * inter;
    float a = bk_act(both[i + c], gelu) * both[i + inter + c];
    if (has_scale) a *= scale[c];
    act[idx] = a;
}

// dact2 [n][inter] (gradient of act·scale) → dgu [n][2·inter] = (dg | du) and
// dm[c] += Σ_r dact2·act(g)·u (the pre-scale act: the mask's gradient before
// σ'). Grid (inter / 256, row chunks of `rows`): a thread one column.
extern "C" __global__ void bk_ffn_act_bwd(const float* both, const float* dact2, const float* scale, float* dgu,
                                          float* dm, int n, int inter, int gelu, int has_scale, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= inter) return;
    const int r0 = blockIdx.y * rows, r1 = min(n, r0 + rows);
    const float sc = has_scale ? scale[c] : 1.f;
    float acc = 0.f;
    for (int r = r0; r < r1; r++) {
        const long i = (long)r * 2 * inter;
        const float g = both[i + c], u = both[i + inter + c], d2 = dact2[(long)r * inter + c];
        const float ag = bk_act(g, gelu);
        acc += d2 * ag * u;
        const float da = d2 * sc;
        dgu[i + c] = da * u * bk_act_bwd(g, gelu);
        dgu[i + inter + c] = da * ag;
    }
    atomicAdd(&dm[c], acc);
}

// ── the bake's exact attention core (fcd_ops::attn_head_{fwd,bwd}_window) ──

// dst[(bi·nh + h)][p][c] = src[(bi·t + p)·(sh·hd) + (h / rep)·hd + c]: a q, or
// a k/v plane repeated over its group's q-heads.
extern "C" __global__ void bk_attn_perm(const float* src, float* dst, int b, int t, int nh, int sh, int hd, int rep) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)b * nh * t * hd;
    if (idx >= tot) return;
    const int c = (int)(idx % hd);
    const long r = idx / hd;
    const int p = (int)(r % t);
    const long u = r / t;
    const int h = (int)(u % nh), bi = (int)(u / nh);
    dst[idx] = src[((long)bi * t + p) * sh * hd + (long)(h / rep) * hd + c];
}

// dst[(bi·t + p)·(dh·hd) + g·hd + c] = Σ over the rep q-heads h of group g of src[(bi·nh + h)][p][c]
// (rep 1: the q layout back).
extern "C" __global__ void bk_attn_unperm(const float* src, float* dst, int b, int t, int nh, int dh, int hd, int rep) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long tot = (long)b * t * dh * hd;
    if (idx >= tot) return;
    const int c = (int)(idx % hd);
    const long r = idx / hd;
    const int g = (int)(r % dh);
    const long row = r / dh;
    const int p = (int)(row % t), bi = (int)(row / t);
    float acc = 0.f;
    for (int k = 0; k < rep; k++) {
        const int h = g * rep + k;
        acc += src[(((long)bi * nh + h) * t + p) * hd + c];
    }
    dst[idx] = acc;
}

// Causal (windowed) softmax of S's rows [batch][p][q] → P, and Pᵀ ([batch][q][p]).
extern "C" __global__ void bk_softmax_fwd(const float* S, float* P, float* PT, int t, int window) {
    const long row = blockIdx.x;
    const long u = row / t;
    const int p = (int)(row % t);
    const int first = window > 0 ? max(0, p + 1 - window) : 0;
    __shared__ float red[32];
    const float* s = S + row * t;
    float mx = -3.402823466e38f;
    for (int q = first + threadIdx.x; q <= p; q += blockDim.x) mx = fmaxf(mx, s[q]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = -3.402823466e38f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float den = 0.f;
    for (int q = first + threadIdx.x; q <= p; q += blockDim.x) den += __expf(s[q] - mx);
    for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = den;
    __syncthreads();
    den = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) den += red[i];
    const float inv = 1.f / den;
    for (int q = threadIdx.x; q < t; q += blockDim.x) {
        const float v = (q >= first && q <= p) ? __expf(s[q] - mx) * inv : 0.f;
        P[row * t + q] = v;
        PT[(u * t + q) * t + p] = v;
    }
}

// dS = P ⊙ (dP − Σ_q P·dP) per row, and dSᵀ.
extern "C" __global__ void bk_softmax_bwd(const float* P, const float* dP, float* dS, float* dST, int t) {
    const long row = blockIdx.x;
    const long u = row / t;
    const int p = (int)(row % t);
    __shared__ float red[32];
    float acc = 0.f;
    for (int q = threadIdx.x; q < t; q += blockDim.x) acc += P[row * t + q] * dP[row * t + q];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float dot = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) dot += red[i];
    for (int q = threadIdx.x; q < t; q += blockDim.x) {
        const float v = P[row * t + q] * (dP[row * t + q] - dot);
        dS[row * t + q] = v;
        dST[(u * t + q) * t + p] = v;
    }
}

// ── the bake's LM head loss (skillbake.rs: CE over the scored rows) ──

// Row r of logits [rows][vocab]: nll[r] = logsumexp − logit[target], arg[r] =
// argmax; with grad, the row becomes (softmax − onehot)·scale in place.
extern "C" __global__ void bk_ce_rows(float* logits, const int* targets, float* nll, int* arg, int vocab, float scale,
                                      int grad) {
    const long r = blockIdx.x;
    float* row = logits + r * vocab;
    __shared__ float rv[32];
    __shared__ int ri[32];
    float mx = -3.402823466e38f;
    int am = 0;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        const float v = row[j];
        if (v > mx) {
            mx = v;
            am = j;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, mx, o);
        const int oi = __shfl_xor_sync(0xffffffffu, am, o);
        if (ov > mx || (ov == mx && oi < am)) {
            mx = ov;
            am = oi;
        }
    }
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        rv[w] = mx;
        ri[w] = am;
    }
    __syncthreads();
    mx = rv[0];
    am = ri[0];
    for (int i = 1; i < nw; i++) {
        if (rv[i] > mx || (rv[i] == mx && ri[i] < am)) {
            mx = rv[i];
            am = ri[i];
        }
    }
    __syncthreads();
    float s = 0.f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) s += __expf(row[j] - mx);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) rv[w] = s;
    __syncthreads();
    s = 0.f;
    for (int i = 0; i < nw; i++) s += rv[i];
    const int tg = targets[r];
    if (threadIdx.x == 0) {
        nll[r] = mx + logf(s) - row[tg];
        arg[r] = am;
    }
    __syncthreads();
    if (grad) {
        const float inv = 1.f / s;
        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            float g = __expf(row[j] - mx) * inv * scale;
            if (j == tg) g -= scale;
            row[j] = g;
        }
    }
}

// ── the resident Phase-A step (skillbake.rs Pass::run on the card) ──

// h[r] = embed[ids[r]]
extern "C" __global__ void bk_embed(const float* embed, const int* ids, float* h, int n, int hsz) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * hsz) return;
    const long r = idx / hsz;
    h[idx] = embed[(long)ids[r] * hsz + (idx % hsz)];
}

// y = x·inv·w' (w' = 1 + w under the Gemma style), inv = 1/√(mean x² + eps); a block a row.
extern "C" __global__ void bk_rmsnorm(const float* x, const float* w, float* y, float* inv_out, int d, float eps,
                                      int gemma) {
    const long r = blockIdx.x;
    const float* xr = x + r * d;
    __shared__ float red[32];
    float ss = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) ss += xr[j] * xr[j];
    ss = bk_block_sum(ss, red);
    const float inv = rsqrtf(ss / d + eps);
    if (threadIdx.x == 0 && inv_out) inv_out[r] = inv;
    for (int j = threadIdx.x; j < d; j += blockDim.x) y[r * d + j] = xr[j] * inv * (gemma ? 1.f + w[j] : w[j]);
}

// dx (+)= inv·w'·dy − x·inv³/d·Σ dy·w'·x (`acc` adds into dx, else stores).
extern "C" __global__ void bk_rmsnorm_bwd(const float* x, const float* w, const float* inv_in, const float* dy,
                                          float* dx, int d, int gemma, int acc) {
    const long r = blockIdx.x;
    const float* xr = x + r * d;
    const float* dr = dy + r * d;
    __shared__ float red[32];
    const float inv = inv_in[r];
    float s = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s += dr[j] * (gemma ? 1.f + w[j] : w[j]) * xr[j];
    s = bk_block_sum(s, red);
    const float k = s * inv * inv * inv / d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        const float g = inv * (gemma ? 1.f + w[j] : w[j]) * dr[j] - xr[j] * k;
        dx[r * d + j] = acc ? dx[r * d + j] + g : g;
    }
}

// f32 -> bf16 over n values, rounded to nearest even (NaN kept quiet): the
// activation operand of a bf16 weight GEMM in the resident step.
extern "C" __global__ void bk_to_bf16(const float* __restrict__ x, unsigned short* __restrict__ y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned int u = __float_as_uint(x[i]);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        y[i] = (unsigned short)((u >> 16) | 0x40u);
        return;
    }
    y[i] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// y = a + b (n elements)
extern "C" __global__ void bk_add(const float* a, const float* b, float* y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// The full-attention layer's projections → rotated q, k, the v plane and the
// output-gate pre-activations: per (row, head) one block of hd threads.
// qkv row: [q | k | v] or, with the output gate, [q_h(hd) g_h(hd)]·nh | k | v.
// q/k get their per-head RMSNorm (when the weights exist) then RoPE over the
// first 2·half dims (pairs i, i + half) from `rope` [t][half][2] = (cos, sin).
extern "C" __global__ void bk_attn_prep(const float* qkv, const float* qn, const float* kn, const float* rope,
                                        float* qrot, float* krot, float* vp, float* gate, int t, int nh, int nkv,
                                        int hd, int half, int gated, float eps, int gemma, int has_qn, int has_kn) {
    const int r = blockIdx.x, hh = blockIdx.y, c = threadIdx.x;
    const int qdim = nh * hd, kvdim = nkv * hd, qrows = gated ? 2 * qdim : qdim, fused = qrows + 2 * kvdim;
    const float* row = qkv + (long)r * fused;
    const int pos = r % t;
    __shared__ float buf[512];
    __shared__ float red[32];
    // hh < nh: a q head; nh ≤ hh < nh + nkv: a k head (v copied beside it)
    const bool isq = hh < nh;
    const int head = isq ? hh : hh - nh;
    const float* src = isq ? row + (gated ? 2 * head * hd : head * hd) : row + qrows + head * hd;
    const float* nw = isq ? qn : kn;
    const int has = isq ? has_qn : has_kn;
    for (int j = c; j < hd; j += blockDim.x) buf[j] = src[j];
    __syncthreads();
    if (has) {
        float ss = 0.f;
        for (int j = c; j < hd; j += blockDim.x) ss += buf[j] * buf[j];
        ss = bk_block_sum(ss, red);
        const float inv = rsqrtf(ss / hd + eps);
        __syncthreads();
        for (int j = c; j < hd; j += blockDim.x) buf[j] = buf[j] * inv * (gemma ? 1.f + nw[j] : nw[j]);
        __syncthreads();
    }
    float* dst = isq ? qrot + (long)r * qdim + head * hd : krot + (long)r * kvdim + head * hd;
    for (int j = c; j < hd; j += blockDim.x) {
        float v = buf[j];
        if (j < 2 * half) {
            const int i = j < half ? j : j - half;
            const float cs = rope[((long)pos * half + i) * 2], sn = rope[((long)pos * half + i) * 2 + 1];
            v = j < half ? buf[i] * cs - buf[i + half] * sn : buf[i] * sn + buf[i + half] * cs;
        }
        dst[j] = v;
    }
    if (isq && gated)
        for (int j = c; j < hd; j += blockDim.x) gate[(long)r * qdim + head * hd + j] = row[2 * head * hd + hd + j];
    if (!isq)
        for (int j = c; j < hd; j += blockDim.x) vp[(long)r * kvdim + head * hd + j] = row[qrows + kvdim + head * hd + j];
}

// bk_attn_prep's backward: dqrot, dkrot (rotated, normed space), dv and dgate
// back to dqkv [n][fused] in the projection layout (overwritten). RoPE is a
// rotation (its transpose undoes it); the per-head RMSNorm's backward is
// dx = inv·w'·dy − x·inv³/hd·Σ dy·w'·x, x the raw projection read from qkv.
extern "C" __global__ void bk_attn_prep_bwd(const float* qkv, const float* qn, const float* kn, const float* rope,
                                            const float* dqrot, const float* dkrot, const float* dv, const float* dgate,
                                            float* dqkv, int t, int nh, int nkv, int hd, int half, int gated, float eps,
                                            int gemma, int has_qn, int has_kn) {
    const int r = blockIdx.x, hh = blockIdx.y, c = threadIdx.x;
    const int qdim = nh * hd, kvdim = nkv * hd, qrows = gated ? 2 * qdim : qdim, fused = qrows + 2 * kvdim;
    const float* row = qkv + (long)r * fused;
    float* drow = dqkv + (long)r * fused;
    const int pos = r % t;
    __shared__ float x[512];
    __shared__ float dy[512];
    __shared__ float red[32];
    const bool isq = hh < nh;
    const int head = isq ? hh : hh - nh;
    const int off = isq ? (gated ? 2 * head * hd : head * hd) : qrows + head * hd;
    const float* nw = isq ? qn : kn;
    const int has = isq ? has_qn : has_kn;
    const float* dsrc = isq ? dqrot + (long)r * qdim + head * hd : dkrot + (long)r * kvdim + head * hd;
    for (int j = c; j < hd; j += blockDim.x) x[j] = row[off + j];
    // RoPE transposed: (dy_i, dy_{i+h}) = (do_i·cs + do_{i+h}·sn, −do_i·sn + do_{i+h}·cs)
    for (int j = c; j < hd; j += blockDim.x) {
        float v = dsrc[j];
        if (j < 2 * half) {
            const int i = j < half ? j : j - half;
            const float cs = rope[((long)pos * half + i) * 2], sn = rope[((long)pos * half + i) * 2 + 1];
            v = j < half ? dsrc[i] * cs + dsrc[i + half] * sn : -dsrc[i] * sn + dsrc[i + half] * cs;
        }
        dy[j] = v;
    }
    __syncthreads();
    if (has) {
        float ss = 0.f, sd = 0.f;
        for (int j = c; j < hd; j += blockDim.x) {
            ss += x[j] * x[j];
            sd += dy[j] * (gemma ? 1.f + nw[j] : nw[j]) * x[j];
        }
        ss = bk_block_sum(ss, red);
        __syncthreads();
        sd = bk_block_sum(sd, red);
        const float inv = rsqrtf(ss / hd + eps);
        const float k = sd * inv * inv * inv / hd;
        for (int j = c; j < hd; j += blockDim.x) drow[off + j] = inv * (gemma ? 1.f + nw[j] : nw[j]) * dy[j] - x[j] * k;
    } else {
        for (int j = c; j < hd; j += blockDim.x) drow[off + j] = dy[j];
    }
    if (isq && gated)
        for (int j = c; j < hd; j += blockDim.x) drow[2 * head * hd + hd + j] = dgate[(long)r * qdim + head * hd + j];
    if (!isq)
        for (int j = c; j < hd; j += blockDim.x) drow[qrows + kvdim + head * hd + j] = dv[(long)r * kvdim + head * hd + j];
}

// ao ← ao·σ(gate) (n elements)
extern "C" __global__ void bk_gate_mul(float* ao, const float* gate, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ao[i] *= bk_sigmoid(gate[i]);
}

// bk_gate_mul's backward: d ← d·σ(g) (in place, the pre-gate ao's gradient),
// dgate = d·ao·σ(g)(1 − σ(g)), from ao before the gate.
extern "C" __global__ void bk_gate_bwd(float* d, const float* ao, const float* gate, float* dgate, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = bk_sigmoid(gate[i]);
    dgate[i] = d[i] * ao[i] * s * (1.f - s);
    d[i] *= s;
}

// dst[k] = src[rows[k]] (row width d)
extern "C" __global__ void bk_gather(const float* src, const int* rows, float* dst, int k, int d) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)k * d) return;
    dst[idx] = src[(long)rows[idx / d] * d + (idx % d)];
}

// dst[rows[k]] += src[k] (rows distinct)
extern "C" __global__ void bk_scatter_add(const float* src, const int* rows, float* dst, int k, int d) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)k * d) return;
    dst[(long)rows[idx / d] * d + (idx % d)] += src[idx];
}
