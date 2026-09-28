// Cortiq Embryo — training kernels (Metal, Apple Silicon).
//
// Everything the birth/growth trainer needs on the device, hand-written:
// no MPS, no framework. The GEMM is the workhorse (all three orientations
// of a linear layer's forward/backward through the TA/TB function
// constants); the rest are the elementwise/reduction companions of the
// fixed graph. f32 storage, f32 accumulate — this is a TRAINER, the
// gradients want the precision (the runtime's inference kernels use
// half tiles; this is a deliberate difference).

#include <metal_stdlib>
using namespace metal;

// Keep the activation and its transpose identical to the established GDN
// operator (`linear_core::gdn_step`/`fcd_ops`).  In particular, depthwise
// convolution is followed by SiLU for all q/k/v streams (not only the output
// gate).  The backward recomputes the pre-activation from the raw stream so
// no additional activation-history buffer is needed.
inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

inline float silu_bwd_f32(float x) {
    const float s = 1.0f / (1.0f + exp(-x));
    return s * (1.0f + x * (1.0f - s));
}

// Deterministic scalar reduction used by the optional GDN correction lane's
// residual-gain gradient.  The lane has only one scalar per layer, so a
// serial thread is both simpler and numerically stable relative to an
// atomic tree reduction.
kernel void dot_accum_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    float s = 0.0f;
    for (uint i = 0u; i < n; ++i) s += a[i] * b[i];
    atomic_fetch_add_explicit(out, s, memory_order_relaxed);
}

kernel void slice_cols_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint4& a [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint rows = a.x, src_cols = a.y, cols = a.z;
    uint n = rows * cols;
    if (gid >= n) return;
    uint r = gid / cols, c = gid % cols;
    dst[gid] = src[r * src_cols + c];
}

kernel void pad_cols_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint4& a [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint rows = a.x, src_cols = a.y, cols = a.z;
    uint n = rows * cols;
    if (gid >= n) return;
    uint r = gid / cols, c = gid % cols;
    dst[gid] = c < src_cols ? src[r * src_cols + c] : 0.0f;
}

// ---------------------------------------------------------------------
// Exact ordinary one-head GatedDeltaNet correction lane (dk=dv=64).
// One serial thread per batch sequence keeps the causal state transition
// unambiguous; the surrounding projection/output GEMMs remain tiled.
// ---------------------------------------------------------------------

struct GdnFwdArgs { uint b; uint t; float eps; uint pad; };
struct GdnBwdArgs { uint b; uint t; uint pad0; uint pad1; };

kernel void gdn_fwd_f32(
    device const float* qraw   [[buffer(0)]],
    device const float* kraw   [[buffer(1)]],
    device const float* vraw   [[buffer(2)]],
    device const float* zraw   [[buffer(3)]],
    device const float* ab     [[buffer(4)]],
    device const float* convw  [[buffer(5)]], // [192,4], q/k/v depthwise
    device const float* norm   [[buffer(6)]],
    device const float* alogp  [[buffer(7)]],
    device const float* dtp    [[buffer(8)]],
    device float* qcv          [[buffer(9)]],
    device float* kcv          [[buffer(10)]],
    device float* vcv          [[buffer(11)]],
    device float* beta_out     [[buffer(12)]],
    device float* raw_o        [[buffer(13)]],
    device float* inv_o        [[buffer(14)]],
    device float* out          [[buffer(15)]],
    device float* states       [[buffer(16)]],
    constant GdnFwdArgs& args  [[buffer(17)]],
    uint bi [[threadgroup_position_in_grid]])
{
    const uint B = args.b, T = args.t;
    if (bi >= B) return;
    const float eps = args.eps;
    const float alog = alogp[0];
    const float dt_bias = dtp[0];
    const ulong state_stride = 64ul * 64ul;
    const ulong state_base = (ulong)bi * (T + 1u) * state_stride;
    for (uint i = 0u; i < 64u * 64u; ++i) states[state_base + i] = 0.0f;
    for (uint ti = 0u; ti < T; ++ti) {
        const ulong row = (ulong)bi * T + ti;
        float qx[64], kx[64], vx[64];
        for (uint c = 0u; c < 64u; ++c) {
            float qv = 0.0f, kv = 0.0f, vv = 0.0f;
            for (uint tap = 0u; tap < 4u; ++tap) {
                int si = (int)ti - (3 - (int)tap);
                if (si >= 0) {
                    ulong sr = (ulong)bi * T + (uint)si;
                    qv += qraw[sr * 64ul + c] * convw[c * 4u + tap];
                    kv += kraw[sr * 64ul + c] * convw[(64u + c) * 4u + tap];
                    vv += vraw[sr * 64ul + c] * convw[(128u + c) * 4u + tap];
                }
            }
            // Faithful GDN applies SiLU after the causal depthwise
            // convolution to each q/k/v channel before normalization and
            // the delta-rule update.
            qx[c] = silu_f32(qv);
            kx[c] = silu_f32(kv);
            vx[c] = silu_f32(vv);
            qcv[row * 64ul + c] = qx[c];
            kcv[row * 64ul + c] = kx[c];
            vcv[row * 64ul + c] = vx[c];
        }
        float qn = 0.0f, kn = 0.0f;
        for (uint c = 0u; c < 64u; ++c) { qn += qx[c] * qx[c]; kn += kx[c] * kx[c]; }
        const float iq = rsqrt(qn + 1.0e-6f) * rsqrt(64.0f);
        const float ik = rsqrt(kn + 1.0e-6f);
        const float sp = (ab[row * 64ul] + dt_bias > 20.0f)
            ? (ab[row * 64ul] + dt_bias)
            : log(1.0f + exp(ab[row * 64ul] + dt_bias));
        const float gg = exp(-exp(alog) * sp);
        const float bb = 1.0f / (1.0f + exp(-ab[row * 64ul + 1ul]));
        beta_out[row] = bb;
        const ulong prev = state_base + (ulong)ti * state_stride;
        const ulong cur = prev + state_stride;
        float kvv[64];
        for (uint j = 0u; j < 64u; ++j) kvv[j] = 0.0f;
        for (uint i = 0u; i < 64u; ++i) {
            const float kf = kx[i] * ik;
            for (uint j = 0u; j < 64u; ++j) {
                float s = states[prev + (ulong)i * 64ul + j] * gg;
                states[cur + (ulong)i * 64ul + j] = s;
                kvv[j] += s * kf;
            }
        }
        float oo[64];
        for (uint j = 0u; j < 64u; ++j) oo[j] = 0.0f;
        for (uint i = 0u; i < 64u; ++i) {
            const float kf = kx[i] * ik, qf = qx[i] * iq;
            for (uint j = 0u; j < 64u; ++j) {
                device float& s = states[cur + (ulong)i * 64ul + j];
                s += kf * (vx[j] - kvv[j]) * bb;
                oo[j] += qf * s;
            }
        }
        float ss = 0.0f;
        for (uint j = 0u; j < 64u; ++j) { raw_o[row * 64ul + j] = oo[j]; ss += oo[j] * oo[j]; }
        const float rinv = rsqrt(ss / 64.0f + eps);
        inv_o[row] = rinv;
        for (uint j = 0u; j < 64u; ++j) out[row * 64ul + j] = oo[j] * rinv * norm[j];
        (void)zraw; // z is consumed by the existing elementwise SiLU gate.
    }
}

kernel void gdn_bwd_f32(
    device const float* qraw   [[buffer(0)]],
    device const float* kraw   [[buffer(1)]],
    device const float* vraw   [[buffer(2)]],
    device const float* qcv    [[buffer(3)]],
    device const float* kcv    [[buffer(4)]],
    device const float* vcv    [[buffer(5)]],
    device const float* ab     [[buffer(6)]],
    device const float* beta_o [[buffer(7)]],
    device const float* raw_o  [[buffer(8)]],
    device const float* inv_o  [[buffer(9)]],
    device const float* norm   [[buffer(10)]],
    device const float* states [[buffer(11)]],
    device const float* dnorm  [[buffer(12)]],
    device const float* dz     [[buffer(13)]],
    device float* dqraw        [[buffer(14)]],
    device float* dkraw        [[buffer(15)]],
    device float* dvraw        [[buffer(16)]],
    device float* dab           [[buffer(17)]],
    device const float* convw  [[buffer(18)]],
    device const float* alogp  [[buffer(19)]],
    device const float* dtp    [[buffer(20)]],
    device atomic_float* gconv [[buffer(21)]],
    device atomic_float* gnorm [[buffer(22)]],
    device atomic_float* galog [[buffer(23)]],
    device atomic_float* gdt   [[buffer(24)]],
    constant GdnBwdArgs& args  [[buffer(25)]],
    uint bi [[threadgroup_position_in_grid]])
{
    const uint B = args.b, T = args.t;
    if (bi >= B) return;
    const ulong stride = 64ul * 64ul, seq = (ulong)bi * T;
    const float alog = alogp[0], dt_bias = dtp[0], ea = exp(alog);
    float dS[4096];
    for (uint i = 0u; i < 4096u; ++i) dS[i] = 0.0f;
    for (uint r = 0u; r < T; ++r) {
        for (uint c = 0u; c < 64u; ++c) {
            dqraw[(seq + r) * 64ul + c] = 0.0f;
            dkraw[(seq + r) * 64ul + c] = 0.0f;
            dvraw[(seq + r) * 64ul + c] = 0.0f;
            dab[(seq + r) * 64ul + c] = 0.0f;
        }
    }
    for (uint ti = T; ti-- > 0u;) {
        const ulong row = seq + ti;
        float oo[64], doo[64], qx[64], kx[64], vx[64];
        float qpre[64], kpre[64], vpre[64];
        float qn = 0.0f, kn = 0.0f;
        for (uint j = 0u; j < 64u; ++j) {
            oo[j] = raw_o[row * 64ul + j];
            float gj = dnorm[row * 64ul + j] * norm[j];
            doo[j] = gj * inv_o[row];
        }
        float dotn = 0.0f;
        for (uint j = 0u; j < 64u; ++j) dotn += dnorm[row * 64ul + j] * norm[j] * oo[j];
        const float rinv = inv_o[row];
        const float c_norm = rinv * rinv * rinv * dotn / 64.0f;
        for (uint j = 0u; j < 64u; ++j) {
            doo[j] -= c_norm * oo[j];
            atomic_fetch_add_explicit(gnorm + j, dnorm[row * 64ul + j] * oo[j] * rinv, memory_order_relaxed);
        }
        for (uint c = 0u; c < 64u; ++c) {
            // qcv/kcv/vcv hold post-SiLU values from forward.  Recompute the
            // causal-convolution pre-activations for the SiLU transpose and
            // convolution-weight/raw-stream gradients.
            float qp = 0.0f, kp = 0.0f, vp = 0.0f;
            for (uint tap = 0u; tap < 4u; ++tap) {
                int si = (int)ti - (3 - (int)tap);
                if (si >= 0) {
                    ulong sr = seq + (uint)si;
                    qp += qraw[sr * 64ul + c] * convw[c * 4u + tap];
                    kp += kraw[sr * 64ul + c] * convw[(64u + c) * 4u + tap];
                    vp += vraw[sr * 64ul + c] * convw[(128u + c) * 4u + tap];
                }
            }
            qpre[c] = qp; kpre[c] = kp; vpre[c] = vp;
            qx[c] = qcv[row * 64ul + c]; kx[c] = kcv[row * 64ul + c]; vx[c] = vcv[row * 64ul + c];
            qn += qx[c] * qx[c]; kn += kx[c] * kx[c];
        }
        const float iq = rsqrt(qn + 1.0e-6f) * rsqrt(64.0f), ik = rsqrt(kn + 1.0e-6f);
        const float bb = beta_o[row], aa = ab[row * 64ul];
        const float sp = (aa + dt_bias > 20.0f) ? (aa + dt_bias) : log(1.0f + exp(aa + dt_bias));
        const float gg = exp(-ea * sp);
        const ulong cur = ((ulong)bi * (T + 1u) + ti + 1u) * stride;
        const ulong prev = cur - stride;
        float dQf[64], dKf[64], dVf[64], du[64], dkv[64];
        for (uint c = 0u; c < 64u; ++c) {
            dQf[c] = 0.0f;
            dKf[c] = 0.0f;
            dVf[c] = 0.0f;
            du[c] = 0.0f;
            dkv[c] = 0.0f;
        }
        // o = S_tᵀ q̂: dS_t += q̂⊗do and dq̂ = S_t·do.
        for (uint i = 0u; i < 64u; ++i) {
            const float qf = qx[i] * iq;
            for (uint j = 0u; j < 64u; ++j) {
                dQf[i] += states[cur + (ulong)i * 64ul + j] * doo[j];
                dS[i * 64u + j] += qf * doo[j];
            }
        }
        // Rebuild kv = (g·S_prev)ᵀ k̂, exactly as in the forward update.
        float kvv[64];
        for (uint j = 0u; j < 64u; ++j) {
            kvv[j] = 0.0f;
            for (uint i = 0u; i < 64u; ++i) {
                kvv[j] += gg * states[prev + (ulong)i * 64ul + j] * (kx[i] * ik);
            }
        }
        // S_t = S_pre + k̂⊗u, u = β(v−kv).  First obtain du, then its
        // k/v/beta paths, and finally reverse S_pre = g·S_prev.
        for (uint i = 0u; i < 64u; ++i) {
            const float kf = kx[i] * ik;
            for (uint j = 0u; j < 64u; ++j) {
                du[j] += dS[i * 64u + j] * kf;
                dKf[i] += dS[i * 64u + j] * (vx[j] - kvv[j]) * bb;
                dVf[j] += dS[i * 64u + j] * kf * bb;
            }
        }
        float dbeta = 0.0f;
        for (uint j = 0u; j < 64u; ++j) {
            dbeta += du[j] * (vx[j] - kvv[j]);
            dkv[j] = -bb * du[j];
        }
        float dg = 0.0f;
        for (uint i = 0u; i < 64u; ++i) {
            const float kf = kx[i] * ik;
            for (uint j = 0u; j < 64u; ++j) {
                const float sprev = states[prev + (ulong)i * 64ul + j];
                const float dspre = dS[i * 64u + j] + kf * dkv[j];
                dKf[i] += (gg * sprev) * dkv[j];
                dg += dspre * sprev;
                dS[i * 64u + j] = gg * dspre;
            }
        }
        // q/k normalization backward, then causal depthwise-conv transpose.
        float dqdot = 0.0f, dkdot = 0.0f;
        // The normalization transpose needs d\hat{x}·x (the raw
        // post-convolution vector), not d\hat{x}·\hat{x}.  The latter adds
        // an extra inverse-norm factor and was the source of a large k/q
        // gradient error against fcd_ops.
        for (uint i = 0u; i < 64u; ++i) { dqdot += dQf[i] * qx[i]; dkdot += dKf[i] * kx[i]; }
        for (uint i = 0u; i < 64u; ++i) {
            float dqc = iq * dQf[i] - qx[i] * iq * iq * iq * 64.0f * dqdot;
            float dkc = ik * dKf[i] - kx[i] * ik * ik * ik * dkdot;
            const float dqp = dqc * silu_bwd_f32(qpre[i]);
            const float dkp = dkc * silu_bwd_f32(kpre[i]);
            const float dvp = dVf[i] * silu_bwd_f32(vpre[i]);
            for (uint tap = 0u; tap < 4u; ++tap) {
                int si = (int)ti - (3 - (int)tap);
                if (si >= 0) {
                    ulong sr = seq + (uint)si;
                    dqraw[sr * 64ul + i] += dqp * convw[i * 4u + tap];
                    dkraw[sr * 64ul + i] += dkp * convw[(64u + i) * 4u + tap];
                    dvraw[sr * 64ul + i] += dvp * convw[(128u + i) * 4u + tap];
                    atomic_fetch_add_explicit(gconv + i * 4u + tap, dqp * qraw[sr * 64ul + i], memory_order_relaxed);
                    atomic_fetch_add_explicit(gconv + (64u + i) * 4u + tap, dkp * kraw[sr * 64ul + i], memory_order_relaxed);
                    atomic_fetch_add_explicit(gconv + (128u + i) * 4u + tap, dvp * vraw[sr * 64ul + i], memory_order_relaxed);
                }
            }
        }
        const float d_a = dg * (-ea * (1.0f / (1.0f + exp(-(aa + dt_bias)))) * gg);
        // d g / d A_log = g · (-exp(A_log) · softplus(arg)); retain the
        // multiplicative decay `gg` (the D10 defect).
        const float d_alog = dg * (-ea * sp * gg);
        const float d_dt = d_a; // dt_bias enters only the decay softplus.
        dab[row * 64ul] = d_a;
        dab[row * 64ul + 1ul] = dbeta * bb * (1.0f - bb);
        atomic_fetch_add_explicit(galog, d_alog, memory_order_relaxed);
        atomic_fetch_add_explicit(gdt, d_dt, memory_order_relaxed);
        (void)dz; // z gradient is already the SiLU derivative output buffer.
    }
}

// ---------------------------------------------------------------------
// Parallel exact ordinary one-head GDN scans (dk=dv=64).
//
// A threadgroup owns one complete batch sequence.  The time loop is still
// strictly causal (and reverse-causal for the backward pass); within each
// token the 64 lanes split convolution channels, recurrent state rows,
// reductions, and output work.  The serial kernels above are deliberately
// retained as the oracle/fallback.
// ---------------------------------------------------------------------

kernel void gdn_fwd_par_f32(
    device const float* qraw   [[buffer(0)]],
    device const float* kraw   [[buffer(1)]],
    device const float* vraw   [[buffer(2)]],
    device const float* zraw   [[buffer(3)]],
    device const float* ab     [[buffer(4)]],
    device const float* convw  [[buffer(5)]],
    device const float* norm   [[buffer(6)]],
    device const float* alogp  [[buffer(7)]],
    device const float* dtp    [[buffer(8)]],
    device float* qcv          [[buffer(9)]],
    device float* kcv          [[buffer(10)]],
    device float* vcv          [[buffer(11)]],
    device float* beta_out     [[buffer(12)]],
    device float* raw_o        [[buffer(13)]],
    device float* inv_o        [[buffer(14)]],
    device float* out          [[buffer(15)]],
    device float* states       [[buffer(16)]],
    constant GdnFwdArgs& args  [[buffer(17)]],
    uint bi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint B = args.b, T = args.t;
    if (bi >= B || tid >= 64u) return;
    const float eps = args.eps;
    const float alog = alogp[0], dt_bias = dtp[0];
    const ulong state_stride = 64ul * 64ul;
    const ulong state_base = (ulong)bi * (T + 1u) * state_stride;
    threadgroup float qred[64];
    threadgroup float kred[64];
    threadgroup float kvred[64];
    threadgroup float ored[64];

    // Initial state is explicit so the parallel path is safe to reuse after
    // a prior command buffer (and matches the serial zero-state contract).
    for (uint j = 0u; j < 64u; ++j) {
        states[state_base + (ulong)tid * 64ul + j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint ti = 0u; ti < T; ++ti) {
        const ulong row = (ulong)bi * T + ti;
        const uint c = tid;
        float qv = 0.0f, kv = 0.0f, vv = 0.0f;
        for (uint tap = 0u; tap < 4u; ++tap) {
            int si = (int)ti - (3 - (int)tap);
            if (si >= 0) {
                ulong sr = (ulong)bi * T + (uint)si;
                qv += qraw[sr * 64ul + c] * convw[c * 4u + tap];
                kv += kraw[sr * 64ul + c] * convw[(64u + c) * 4u + tap];
                vv += vraw[sr * 64ul + c] * convw[(128u + c) * 4u + tap];
            }
        }
        const float qx = silu_f32(qv), kx = silu_f32(kv), vx = silu_f32(vv);
        qcv[row * 64ul + c] = qx;
        kcv[row * 64ul + c] = kx;
        vcv[row * 64ul + c] = vx;

        // Deterministic reductions (lane 0 sums in channel order) retain
        // the serial oracle's arithmetic while all channels run together.
        qred[tid] = qx * qx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float qn = 0.0f;
            for (uint i = 0u; i < 64u; ++i) qn += qred[i];
            qred[0] = qn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float iq = rsqrt(qred[0] + 1.0e-6f) * rsqrt(64.0f);
        kred[tid] = kx * kx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float kn = 0.0f;
            for (uint i = 0u; i < 64u; ++i) kn += kred[i];
            kred[0] = kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ik = rsqrt(kred[0] + 1.0e-6f);
        const float aa = ab[row * 64ul];
        const float sp = (aa + dt_bias > 20.0f)
            ? (aa + dt_bias)
            : log(1.0f + exp(aa + dt_bias));
        const float gg = exp(-exp(alog) * sp);
        if (tid == 0u) beta_out[row] = 1.0f / (1.0f + exp(-ab[row * 64ul + 1ul]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float bb = beta_out[row];
        const ulong prev = state_base + (ulong)ti * state_stride;
        const ulong cur = prev + state_stride;

        // Decay every state cell in parallel (one state row per lane).
        for (uint j = 0u; j < 64u; ++j) {
            states[cur + (ulong)tid * 64ul + j] = states[prev + (ulong)tid * 64ul + j] * gg;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // Each lane reduces one output column of Sᵀk̂.
        float kvj = 0.0f;
        for (uint i = 0u; i < 64u; ++i) {
            kvj += states[cur + (ulong)i * 64ul + tid] * (kcv[row * 64ul + i] * ik);
        }
        kvred[tid] = kvj;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = kvred[tid];

        // Delta update; each lane owns one state row.
        const float kf = kx * ik;
        for (uint j = 0u; j < 64u; ++j) {
            device float& s = states[cur + (ulong)tid * 64ul + j];
            s += kf * (vcv[row * 64ul + j] - kvred[j]) * bb;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // One output column per lane, followed by a deterministic norm sum.
        float oo = 0.0f;
        const float qf = qx * iq;
        for (uint i = 0u; i < 64u; ++i) {
            oo += (qcv[row * 64ul + i] * iq) * states[cur + (ulong)i * 64ul + tid];
        }
        raw_o[row * 64ul + tid] = oo;
        ored[tid] = oo * oo;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float ss = 0.0f;
            for (uint j = 0u; j < 64u; ++j) ss += ored[j];
            ored[0] = rsqrt(ss / 64.0f + eps);
            inv_o[row] = ored[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out[row * 64ul + tid] = oo * ored[0] * norm[tid];
        (void)zraw;
        (void)kv_col;
        (void)qf;
    }
}

kernel void gdn_bwd_par_f32(
    device const float* qraw   [[buffer(0)]],
    device const float* kraw   [[buffer(1)]],
    device const float* vraw   [[buffer(2)]],
    device const float* qcv    [[buffer(3)]],
    device const float* kcv    [[buffer(4)]],
    device const float* vcv    [[buffer(5)]],
    device const float* ab     [[buffer(6)]],
    device const float* beta_o [[buffer(7)]],
    device const float* raw_o  [[buffer(8)]],
    device const float* inv_o  [[buffer(9)]],
    device const float* norm   [[buffer(10)]],
    device const float* states [[buffer(11)]],
    device const float* dnorm  [[buffer(12)]],
    device const float* dz     [[buffer(13)]],
    device float* dqraw        [[buffer(14)]],
    device float* dkraw        [[buffer(15)]],
    device float* dvraw        [[buffer(16)]],
    device float* dab           [[buffer(17)]],
    device const float* convw  [[buffer(18)]],
    device const float* alogp  [[buffer(19)]],
    device const float* dtp    [[buffer(20)]],
    device atomic_float* gconv [[buffer(21)]],
    device atomic_float* gnorm [[buffer(22)]],
    device atomic_float* galog [[buffer(23)]],
    device atomic_float* gdt   [[buffer(24)]],
    constant GdnBwdArgs& args  [[buffer(25)]],
    uint bi [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint B = args.b, T = args.t;
    if (bi >= B || tid >= 64u) return;
    const ulong stride = 64ul * 64ul, seq = (ulong)bi * T;
    const float alog = alogp[0], dt_bias = dtp[0], ea = exp(alog);
    threadgroup float dS[4096];
    threadgroup float normred[64];
    threadgroup float qred[64];
    threadgroup float kred[64];
    threadgroup float kvred[64];
    threadgroup float betared[64];
    threadgroup float dgred[64];
    threadgroup float dqred[64];
    threadgroup float dkred[64];
    threadgroup float doo_sh[64];

    for (uint j = 0u; j < 64u; ++j) {
        dS[tid * 64u + j] = 0.0f;
        for (uint r = 0u; r < T; ++r) {
            const ulong row = seq + r;
            dqraw[row * 64ul + tid] = 0.0f;
            dkraw[row * 64ul + tid] = 0.0f;
            dvraw[row * 64ul + tid] = 0.0f;
            dab[row * 64ul + tid] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint ti = T; ti-- > 0u;) {
        const ulong row = seq + ti;
        const float oo = raw_o[row * 64ul + tid];
        const float dn = dnorm[row * 64ul + tid];
        const float no = norm[tid];
        const float inv = inv_o[row];
        const float gj = dn * no;
        const float doo = gj * inv;
        doo_sh[tid] = doo;
        normred[tid] = dn * no * oo;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float dotn = 0.0f;
            for (uint j = 0u; j < 64u; ++j) dotn += normred[j];
            normred[0] = dotn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float c_norm = inv * inv * inv * normred[0] / 64.0f;
        const float doo_n = doo - c_norm * oo;
        doo_sh[tid] = doo_n;
        atomic_fetch_add_explicit(gnorm + tid, dn * oo * inv, memory_order_relaxed);

        float qp = 0.0f, kp = 0.0f, vp = 0.0f;
        for (uint tap = 0u; tap < 4u; ++tap) {
            int si = (int)ti - (3 - (int)tap);
            if (si >= 0) {
                ulong sr = seq + (uint)si;
                qp += qraw[sr * 64ul + tid] * convw[tid * 4u + tap];
                kp += kraw[sr * 64ul + tid] * convw[(64u + tid) * 4u + tap];
                vp += vraw[sr * 64ul + tid] * convw[(128u + tid) * 4u + tap];
            }
        }
        const float qx = qcv[row * 64ul + tid], kx = kcv[row * 64ul + tid];
        const float vx = vcv[row * 64ul + tid];
        qred[tid] = qx * qx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float qn = 0.0f;
            for (uint i = 0u; i < 64u; ++i) qn += qred[i];
            qred[0] = qn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float iq = rsqrt(qred[0] + 1.0e-6f) * rsqrt(64.0f);
        kred[tid] = kx * kx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float kn = 0.0f;
            for (uint i = 0u; i < 64u; ++i) kn += kred[i];
            kred[0] = kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ik = rsqrt(kred[0] + 1.0e-6f);
        const float bb = beta_o[row];
        const float aa = ab[row * 64ul];
        const float sp = (aa + dt_bias > 20.0f)
            ? (aa + dt_bias)
            : log(1.0f + exp(aa + dt_bias));
        const float gg = exp(-ea * sp);
        const ulong cur = ((ulong)bi * (T + 1u) + ti + 1u) * stride;
        const ulong prev = cur - stride;

        // o = Sᵀq̂; one state row per lane.
        float dQf = 0.0f;
        const float qf = qx * iq;
        for (uint j = 0u; j < 64u; ++j) {
            dQf += states[cur + (ulong)tid * 64ul + j] * doo_sh[j];
            dS[tid * 64u + j] += qf * doo_sh[j];
        }

        // kv = (g·S_prev)ᵀk̂; one column reduction per lane.
        float kvv = 0.0f;
        for (uint i = 0u; i < 64u; ++i) {
            kvv += gg * states[prev + (ulong)i * 64ul + tid] * (kcv[row * 64ul + i] * ik);
        }
        kvred[tid] = kvv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_col = kvred[tid];

        // Per-column du/dV and per-row dK.
        float du = 0.0f, dVf = 0.0f, dKf = 0.0f;
        const float kf = kx * ik;
        for (uint i = 0u; i < 64u; ++i) {
            du += dS[i * 64u + tid] * (kcv[row * 64ul + i] * ik);
            dVf += dS[i * 64u + tid] * (kcv[row * 64ul + i] * ik) * bb;
        }
        for (uint j = 0u; j < 64u; ++j) {
            dKf += dS[tid * 64u + j] * (vcv[row * 64ul + j] - kvred[j]) * bb;
        }
        betared[tid] = du * (vcv[row * 64ul + tid] - kv_col);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float dbeta = 0.0f;
            for (uint j = 0u; j < 64u; ++j) dbeta += betared[j];
            betared[0] = dbeta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float dbeta = betared[0];

        // Reverse the state transition and accumulate the decay derivative.
        // Publish -beta·du for all columns before the row loop.
        const float dkv = -bb * du;
        float dg_row = 0.0f;
        kvred[tid] = dkv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < 64u; ++j) {
            const float sprev = states[prev + (ulong)tid * 64ul + j];
            const float dspre = dS[tid * 64u + j] + kf * kvred[j];
            dKf += (gg * sprev) * kvred[j];
            dg_row += dspre * sprev;
            dS[tid * 64u + j] = gg * dspre;
        }
        dgred[tid] = dg_row;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float dg = 0.0f;
            for (uint i = 0u; i < 64u; ++i) dg += dgred[i];
            dgred[0] = dg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float dg = dgred[0];

        // Normalization transpose and causal convolution transpose.  The
        // q/k dot products are reduced in channel order for oracle parity.
        dqred[tid] = dQf * qx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float dotq = 0.0f;
            for (uint i = 0u; i < 64u; ++i) dotq += dqred[i];
            dqred[0] = dotq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float dqdot = dqred[0];
        dkred[tid] = dKf * kx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float dotk = 0.0f;
            for (uint i = 0u; i < 64u; ++i) dotk += dkred[i];
            dkred[0] = dotk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float dkdot = dkred[0];
        const float dqc = iq * dQf - qx * iq * iq * iq * 64.0f * dqdot;
        const float dkc = ik * dKf - kx * ik * ik * ik * dkdot;
        const float dqp = dqc * silu_bwd_f32(qp);
        const float dkp = dkc * silu_bwd_f32(kp);
        const float dvp = dVf * silu_bwd_f32(vp);
        for (uint tap = 0u; tap < 4u; ++tap) {
            int si = (int)ti - (3 - (int)tap);
            if (si >= 0) {
                ulong sr = seq + (uint)si;
                dqraw[sr * 64ul + tid] += dqp * convw[tid * 4u + tap];
                dkraw[sr * 64ul + tid] += dkp * convw[(64u + tid) * 4u + tap];
                dvraw[sr * 64ul + tid] += dvp * convw[(128u + tid) * 4u + tap];
                atomic_fetch_add_explicit(gconv + tid * 4u + tap, dqp * qraw[sr * 64ul + tid], memory_order_relaxed);
                atomic_fetch_add_explicit(gconv + (64u + tid) * 4u + tap, dkp * kraw[sr * 64ul + tid], memory_order_relaxed);
                atomic_fetch_add_explicit(gconv + (128u + tid) * 4u + tap, dvp * vraw[sr * 64ul + tid], memory_order_relaxed);
            }
        }
        if (tid == 0u) {
            const float d_a = dg * (-ea * (1.0f / (1.0f + exp(-(aa + dt_bias)))) * gg);
            const float d_alog = dg * (-ea * sp * gg);
            dab[row * 64ul] = d_a;
            dab[row * 64ul + 1ul] = dbeta * bb * (1.0f - bb);
            atomic_fetch_add_explicit(galog, d_alog, memory_order_relaxed);
            atomic_fetch_add_explicit(gdt, d_a, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        (void)dz;
    }
}

// ---------------------------------------------------------------------
// GDN mixer token scan (plan S7): the SCAN of `linear_core::gdn_step` for
// nv heads of dk×dv state, one 128-lane threadgroup per (sequence, head),
// lane = state row / output column, the running state in device memory
// (`live`, [B·nv][dk][dv]) and ONE checkpoint per 64 tokens (`states`,
// [B·nv][T/64+1][dk][dv], slot 0 = S_0). No per-token history: the
// backward replays each chunk from its checkpoint into `chunk`
// ([B·nv][65][dk][dv]) and walks it in reverse. Every cross-lane reduction
// is summed by lane 0 in lane order — bit-exact run to run, and the same
// arithmetic as the WGSL port (vulkan.rs `gdn_fwd` / `gdn_bwd`).
//   q̂ = q/(‖q‖√dk), k̂ = k/‖k‖ (ε = 1e-6 inside the roots)
//   g = exp(−e^{A_log}·softplus(a + dt_bias)), β = σ(b)
//   kv = Sᵀk̂ ; u = β(v − g·kv) ; S ← g·S + k̂ ⊗ u ; o = Sᵀq̂
// ---------------------------------------------------------------------

struct GdnScanArgs {
    uint b; uint t; uint nv; uint dk;
    uint dv; uint c_dim; uint ab_ld; uint flags; // &1: fwd S_0 from ckpt 0 / bwd dS_T from dlive; &2: β ≡ 1
};

// Two deterministic threadgroup sums (lane 0, lane order over `n` lanes).
inline void gdn_sum2(threadgroup float* ra, threadgroup float* rb, float va, float vb,
                     uint tid, uint n, thread float& oa, thread float& ob) {
    ra[tid] = va;
    rb[tid] = vb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sa = 0.0f, sb = 0.0f;
        for (uint i = 0u; i < n; ++i) { sa += ra[i]; sb += rb[i]; }
        ra[0] = sa;
        rb[0] = sb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    oa = ra[0];
    ob = rb[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void gdn_scan_fwd_f32(
    device const float* qkv_cv [[buffer(0)]],
    device const float* a_pre  [[buffer(1)]],
    device const float* b_pre  [[buffer(2)]],
    device const float* alog   [[buffer(3)]],
    device const float* dtb    [[buffer(4)]],
    device float*       raw_o  [[buffer(5)]],
    device float*       states [[buffer(6)]],
    device float*       live   [[buffer(7)]],
    constant GdnScanArgs& a    [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint B = a.b, T = a.t, nv = a.nv, dk = a.dk, dv = a.dv, cd = a.c_dim, ld = a.ab_ld;
    if (tg >= B * nv) return;
    const uint bi = tg / nv, h = tg % nv;
    const uint nch = (T + 63u) / 64u;
    const ulong ss = (ulong)dk * dv;
    device float* S = live + (ulong)tg * ss;
    device float* ck = states + (ulong)tg * (nch + 1u) * ss;
    threadgroup float qs[128], ks[128], vs[128], ra[128], rb[128];
    const float ea = exp(alog[h]), dt_h = dtb[h];
    const float sdk = rsqrt((float)dk);
    if (tid < dv) {
        if ((a.flags & 1u) != 0u) {
            for (uint i = 0u; i < dk; ++i) S[i * dv + tid] = ck[i * dv + tid];
        } else {
            for (uint i = 0u; i < dk; ++i) { S[i * dv + tid] = 0.0f; ck[i * dv + tid] = 0.0f; }
        }
    }
    for (uint ti = 0u; ti < T; ++ti) {
        const ulong row = (ulong)bi * T + ti;
        const ulong base = row * cd;
        if ((ti % 64u) == 0u && ti > 0u && tid < dv) {
            device float* c = ck + (ulong)(ti / 64u) * ss;
            for (uint i = 0u; i < dk; ++i) c[i * dv + tid] = S[i * dv + tid];
        }
        float qq = 0.0f, kk = 0.0f;
        if (tid < dk) {
            qs[tid] = qkv_cv[base + h * dk + tid];
            ks[tid] = qkv_cv[base + nv * dk + h * dk + tid];
            qq = qs[tid] * qs[tid];
            kk = ks[tid] * ks[tid];
        }
        if (tid < dv) vs[tid] = qkv_cv[base + 2u * nv * dk + h * dv + tid];
        float qn, kn;
        gdn_sum2(ra, rb, qq, kk, tid, dk, qn, kn);
        const float iq = rsqrt(qn + 1.0e-6f) * sdk;
        const float ik = rsqrt(kn + 1.0e-6f);
        const float aa = a_pre[row * ld + h] + dt_h;
        const float sp = (aa > 20.0f) ? aa : log(1.0f + exp(aa));
        const float g = exp(-ea * sp);
        const float beta = ((a.flags & 2u) != 0u) ? 1.0f : 1.0f / (1.0f + exp(-b_pre[row * ld + h]));
        if (tid < dv) {
            float kv = 0.0f;
            for (uint i = 0u; i < dk; ++i) kv += S[i * dv + tid] * ks[i];
            kv *= ik;
            const float u = beta * (vs[tid] - g * kv);
            float o = 0.0f;
            for (uint i = 0u; i < dk; ++i) {
                const float c = g * S[i * dv + tid] + (ks[i] * ik) * u;
                S[i * dv + tid] = c;
                o += (qs[i] * iq) * c;
            }
            raw_o[row * (ulong)(nv * dv) + h * dv + tid] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < dv) {
        device float* c = ck + (ulong)nch * ss;
        for (uint i = 0u; i < dk; ++i) c[i * dv + tid] = S[i * dv + tid];
    }
}

kernel void gdn_scan_bwd_f32(
    device const float* qkv_cv [[buffer(0)]],
    device const float* a_pre  [[buffer(1)]],
    device const float* b_pre  [[buffer(2)]],
    device const float* alog   [[buffer(3)]],
    device const float* dtb    [[buffer(4)]],
    device const float* states [[buffer(5)]],
    device const float* doo    [[buffer(6)]],
    device float*       chunk  [[buffer(7)]],
    device float*       dlive  [[buffer(8)]],
    device float*       dcv    [[buffer(9)]],
    device float*       da     [[buffer(10)]],
    device float*       db     [[buffer(11)]],
    device float*       part   [[buffer(12)]],
    constant GdnScanArgs& a    [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint B = a.b, T = a.t, nv = a.nv, dk = a.dk, dv = a.dv, cd = a.c_dim, ld = a.ab_ld;
    if (tg >= B * nv) return;
    const uint bi = tg / nv, h = tg % nv;
    const uint nch = (T + 63u) / 64u;
    const ulong ss = (ulong)dk * dv;
    device float* ch = chunk + (ulong)tg * 65u * ss;
    device float* dS = dlive + (ulong)tg * ss;
    device const float* ck = states + (ulong)tg * (nch + 1u) * ss;
    threadgroup float qs[128], ks[128], vs[128], dos[128], us[128], dkvs[128], ra[128], rb[128];
    const float ea = exp(alog[h]), dt_h = dtb[h];
    const float sdk = rsqrt((float)dk);
    if ((a.flags & 1u) == 0u && tid < dv) {
        for (uint i = 0u; i < dk; ++i) dS[i * dv + tid] = 0.0f;
    }
    float p_alog = 0.0f, p_dt = 0.0f;
    for (uint c = nch; c-- > 0u;) {
        const uint start = c * 64u;
        const uint clen = min(64u, T - start);
        // replay S_{start..start+clen} from the checkpoint into the chunk arena
        if (tid < dv) {
            for (uint i = 0u; i < dk; ++i) ch[i * dv + tid] = ck[(ulong)c * ss + i * dv + tid];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (uint j = 0u; j < clen; ++j) {
            const ulong row = (ulong)bi * T + start + j;
            const ulong base = row * cd;
            float qq = 0.0f, kk = 0.0f;
            if (tid < dk) {
                qs[tid] = qkv_cv[base + h * dk + tid];
                ks[tid] = qkv_cv[base + nv * dk + h * dk + tid];
                qq = qs[tid] * qs[tid];
                kk = ks[tid] * ks[tid];
            }
            if (tid < dv) vs[tid] = qkv_cv[base + 2u * nv * dk + h * dv + tid];
            float qn, kn;
            gdn_sum2(ra, rb, qq, kk, tid, dk, qn, kn);
            const float ik = rsqrt(kn + 1.0e-6f);
            const float aa = a_pre[row * ld + h] + dt_h;
            const float sp = (aa > 20.0f) ? aa : log(1.0f + exp(aa));
            const float g = exp(-ea * sp);
            const float beta = ((a.flags & 2u) != 0u) ? 1.0f : 1.0f / (1.0f + exp(-b_pre[row * ld + h]));
            device const float* Sp = ch + (ulong)j * ss;
            device float* Sc = ch + (ulong)(j + 1u) * ss;
            if (tid < dv) {
                float kv = 0.0f;
                for (uint i = 0u; i < dk; ++i) kv += Sp[i * dv + tid] * ks[i];
                kv *= ik;
                const float u = beta * (vs[tid] - g * kv);
                for (uint i = 0u; i < dk; ++i) Sc[i * dv + tid] = g * Sp[i * dv + tid] + (ks[i] * ik) * u;
            }
            (void)qn;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // reverse walk of the chunk
        for (uint j = clen; j-- > 0u;) {
            const ulong row = (ulong)bi * T + start + j;
            const ulong base = row * cd;
            float qq = 0.0f, kk = 0.0f;
            if (tid < dk) {
                qs[tid] = qkv_cv[base + h * dk + tid];
                ks[tid] = qkv_cv[base + nv * dk + h * dk + tid];
                qq = qs[tid] * qs[tid];
                kk = ks[tid] * ks[tid];
            }
            if (tid < dv) {
                vs[tid] = qkv_cv[base + 2u * nv * dk + h * dv + tid];
                dos[tid] = doo[row * (ulong)(nv * dv) + h * dv + tid];
            }
            float qn, kn;
            gdn_sum2(ra, rb, qq, kk, tid, dk, qn, kn);
            const float iq = rsqrt(qn + 1.0e-6f) * sdk;
            const float ik = rsqrt(kn + 1.0e-6f);
            const float aa = a_pre[row * ld + h] + dt_h;
            const float sp = (aa > 20.0f) ? aa : log(1.0f + exp(aa));
            const float sig = 1.0f / (1.0f + exp(-aa));
            const float g = exp(-ea * sp);
            const float beta = ((a.flags & 2u) != 0u) ? 1.0f : 1.0f / (1.0f + exp(-b_pre[row * ld + h]));
            device const float* Sp = ch + (ulong)j * ss;
            device const float* Sc = ch + (ulong)(j + 1u) * ss;
            // A (column): kv, u; dS += q̂ ⊗ do
            float kvc = 0.0f;
            if (tid < dv) {
                for (uint i = 0u; i < dk; ++i) kvc += Sp[i * dv + tid] * ks[i];
                kvc *= ik * g;
                us[tid] = beta * (vs[tid] - kvc);
                const float dd = dos[tid];
                for (uint i = 0u; i < dk; ++i) dS[i * dv + tid] += (qs[i] * iq) * dd;
            }
            // B (row): dq̂ = S_t · do
            float dqh = 0.0f;
            if (tid < dk) {
                for (uint jj = 0u; jj < dv; ++jj) dqh += Sc[tid * dv + jj] * dos[jj];
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            // C (column): du, dv, dkv, dβ partial
            float bpart = 0.0f;
            if (tid < dv) {
                float du = 0.0f;
                for (uint i = 0u; i < dk; ++i) du += dS[i * dv + tid] * (ks[i] * ik);
                dcv[base + 2u * nv * dk + h * dv + tid] = beta * du;
                dkvs[tid] = -beta * du;
                bpart = du * (vs[tid] - kvc);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            // D (row): dk̂, dg partial, dS ← g·(dS + k̂ ⊗ dkv)
            float dkh = 0.0f, gpart = 0.0f;
            if (tid < dk) {
                const float kf = ks[tid] * ik;
                for (uint jj = 0u; jj < dv; ++jj) {
                    const float sprev = Sp[tid * dv + jj];
                    const float dspre = dS[tid * dv + jj] + kf * dkvs[jj];
                    dkh += dS[tid * dv + jj] * us[jj] + (g * sprev) * dkvs[jj];
                    gpart += dspre * sprev;
                    dS[tid * dv + jj] = g * dspre;
                }
            }
            float dbeta, dg;
            gdn_sum2(ra, rb, bpart, gpart, tid, 128u, dbeta, dg);
            float dqdot, dkdot;
            gdn_sum2(ra, rb, (tid < dk) ? dqh * qs[tid] : 0.0f, (tid < dk) ? dkh * ks[tid] : 0.0f, tid, dk, dqdot, dkdot);
            if (tid < dk) {
                dcv[base + h * dk + tid] = iq * dqh - qs[tid] * iq * iq * iq * (float)dk * dqdot;
                dcv[base + nv * dk + h * dk + tid] = ik * dkh - ks[tid] * ik * ik * ik * dkdot;
            }
            if (tid == 0u) {
                const float d_a = dg * (-ea * sig * g);
                da[row * ld + h] = d_a;
                db[row * ld + h] = dbeta * beta * (1.0f - beta);
                p_alog += dg * (-ea * sp * g);
                p_dt += d_a;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
    if (tid == 0u) {
        part[(ulong)tg * 2u] = p_alog;
        part[(ulong)tg * 2u + 1u] = p_dt;
    }
}

// g_alog[h] += Σ_b part[(b·nv+h)·2], g_dt[h] += Σ_b part[..+1] — one lane per head.
kernel void gdn_scan_fold_f32(
    device const float* part  [[buffer(0)]],
    device float*       galog [[buffer(1)]],
    device float*       gdt   [[buffer(2)]],
    constant uint2&     bn    [[buffer(3)]], // b, nv
    uint h [[thread_position_in_grid]])
{
    if (h >= bn.y) return;
    float sa = 0.0f, sd = 0.0f;
    for (uint b = 0u; b < bn.x; ++b) {
        sa += part[(ulong)(b * bn.y + h) * 2u];
        sd += part[(ulong)(b * bn.y + h) * 2u + 1u];
    }
    galog[h] += sa;
    gdt[h] += sd;
}

kernel void silu_fwd_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    y[gid] = silu_f32(x[gid]);
}

// dx = dy · silu'(x) with x the pre-activation
kernel void silu_bwd_f32(
    device const float* x  [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float*       dx [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dx[gid] = dy[gid] * silu_bwd_f32(x[gid]);
}

// ---------------------------------------------------------------------
// GEMM:  C[M,N] = alpha · op(A)[M,K] · op(B)[K,N] + beta · C[M,N]
//
//   TA = false: A stored row-major [M,K], lda ≥ K   (A[m,k] = A[m·lda + k])
//   TA = true : A stored row-major [K,M], lda ≥ M   (A[m,k] = A[k·lda + m])
//   TB = false: B stored row-major [K,N], ldb ≥ N   (B[k,n] = B[k·ldb + n])
//   TB = true : B stored row-major [N,K], ldb ≥ K   (B[k,n] = B[n·ldb + k])
//
// A linear layer y = x·Wᵀ (x[M,K], W[N,K]) is (TA=0,TB=1);
// its input grad dx = dy·W (dy[M,N], W[N,K]→[K',N'] with K'=N) is (0,0);
// its weight grad dW = dyᵀ·x (dyᵀ: [N,M] from dy[M,N]; x[M,K]) is (1,0).
//
// Tile 64×64×32, 128 threads = 4 simdgroups, each simdgroup owns a 32×32
// quadrant as 4×4 simdgroup_float8x8 accumulators. Host guarantees
// M%64 == 0, N%64 == 0, K%32 == 0 and 16-byte aligned rows (all Embryo
// shapes are multiples of 64 by construction; the host asserts).
// ---------------------------------------------------------------------

constant bool TA [[function_constant(0)]];
constant bool TB [[function_constant(1)]];

struct GemmArgs {
    // batch strides (elements) for the (b, h, c) decomposition of tgid.z
    ulong sa_b, sa_h, sa_c;
    ulong sb_b, sb_h, sb_c;
    ulong sc_b, sc_h, sc_c;
    uint M, N, K;
    uint lda, ldb, ldc;
    float alpha, beta;
    uint nb_h, nb_c;   // z = (b·nb_h + h)·nb_c + c
    uint mask;         // 1: causal — C[i,j] = 0 for j > i (global indices)
    uint kdyn;         // 1: K = min(round64(kcount[z]), K) — dynamic reduction length
};

#define BM 64u
#define BN 64u
#define BK 32u
#define NTHREADS 128u

kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant GemmArgs&  g [[buffer(3)]],
    device const uint*  kcount [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    uint Kdim = g.K;
    {
        uint z = tgid.z;
        if (g.kdyn == 1u) { Kdim = min(((kcount[z] + 63u) / 64u) * 64u, g.K); }
        uint cb = z % g.nb_c;
        uint rem = z / g.nb_c;
        uint hb = rem % g.nb_h;
        uint bb = rem / g.nb_h;
        A += bb * g.sa_b + hb * g.sa_h + cb * g.sa_c;
        B += bb * g.sb_b + hb * g.sb_h + cb * g.sb_c;
        C += bb * g.sc_b + hb * g.sc_h + cb * g.sc_c;
    }
    // One 16 KB arena: sA = [BM][BK] (m-major, k contiguous),
    // sB = [BK][BN] (k-major, n contiguous). After the K loop the same
    // 4096 floats are the [BM][BN] C staging tile.
    threadgroup float smem[BM * BN];
    threadgroup float* sA = smem;
    threadgroup float* sB = smem + BM * BK;

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint sm = (sgid >> 1) * 32u;   // this simdgroup's rows in the tile
    const uint sn = (sgid & 1u) * 32u;   // this simdgroup's cols in the tile

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < Kdim; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- A tile: BM×BK = 2048 floats, 512 float4, 4 per thread ----
        if (!TA) {
            // contiguous along k: 64 rows × 8 float4
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint r = i >> 3, c4 = i & 7u;
                float4 v = *(device const float4*)(A + (ulong)(m0 + r) * g.lda + k0 + c4 * 4u);
                *(threadgroup float4*)(sA + r * BK + c4 * 4u) = v;
            }
        } else {
            // A stored [K,M]: contiguous along m: 32 k-rows × 16 float4,
            // scattered into sA transposed (scalar stores; see the
            // runtime's note on threadgroup pointer casts).
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint kk = i >> 4, c4 = i & 15u;
                float4 v = *(device const float4*)(A + (ulong)(k0 + kk) * g.lda + m0 + c4 * 4u);
                uint r = c4 * 4u;
                sA[(r + 0u) * BK + kk] = v.x;
                sA[(r + 1u) * BK + kk] = v.y;
                sA[(r + 2u) * BK + kk] = v.z;
                sA[(r + 3u) * BK + kk] = v.w;
            }
        }
        // ---- B tile: BK×BN = 2048 floats ----
        if (!TB) {
            // B[K,N]: contiguous along n: 32 k-rows × 16 float4
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint kk = i >> 4, c4 = i & 15u;
                float4 v = *(device const float4*)(B + (ulong)(k0 + kk) * g.ldb + n0 + c4 * 4u);
                *(threadgroup float4*)(sB + kk * BN + c4 * 4u) = v;
            }
        } else {
            // B stored [N,K]: contiguous along k: 64 n-rows × 8 float4,
            // scattered into sB transposed.
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint r = i >> 3, c4 = i & 7u;
                float4 v = *(device const float4*)(B + (ulong)(n0 + r) * g.ldb + k0 + c4 * 4u);
                uint kk = c4 * 4u;
                sB[(kk + 0u) * BN + r] = v.x;
                sB[(kk + 1u) * BN + r] = v.y;
                sB[(kk + 2u) * BN + r] = v.z;
                sB[(kk + 3u) * BN + r] = v.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(a[i], sA + (sm + i * 8u) * BK + kk, BK);
            }
            #pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                simdgroup_load(b[j], sB + kk * BN + sn + j * 8u, BN);
            }
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
    }

    // ---- epilogue: stage the 64×64 tile, then alpha/beta with float4 stores ----
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (short i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], smem + (sm + i * 8u) * BN + sn + j * 8u, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool accumulate = g.beta != 0.0f;
    #pragma clang loop unroll(full)
    for (uint it = 0; it < 8u; ++it) {
        uint i = tid + it * NTHREADS;      // 0..1023 float4 slots
        uint r = i >> 4, c4 = i & 15u;
        float4 v = *(threadgroup float4*)(smem + r * BN + c4 * 4u) * g.alpha;
        device float4* dst = (device float4*)(C + (ulong)(m0 + r) * g.ldc + n0 + c4 * 4u);
        if (accumulate) { v += *dst * g.beta; }
        if (g.mask == 1u) {
            uint i = m0 + r, j0 = n0 + c4 * 4u;
            if (j0 + 0u > i) v.x = 0.0f;
            if (j0 + 1u > i) v.y = 0.0f;
            if (j0 + 2u > i) v.z = 0.0f;
            if (j0 + 3u > i) v.w = 0.0f;
        }
        *dst = v;
    }
}

// ---------------------------------------------------------------------
// Elementwise / reduction companions (one thread per element or row).
// ---------------------------------------------------------------------

// y = a*x + b*y  (axpby over n floats)
kernel void axpby_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant float&     a [[buffer(2)]],
    constant float&     b [[buffer(3)]],
    constant uint&      n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { y[gid] = a * x[gid] + b * y[gid]; }
}

// AdamW (decoupled weight decay), one thread per parameter.
// m = b1·m + (1−b1)·g ; v = b2·v + (1−b2)·g² ;
// p -= lr·( m̂/(√v̂+eps) + wd·p ),  m̂ = m/(1−b1ᵗ), v̂ = v/(1−b2ᵗ)
struct AdamArgs {
    uint  n;
    float lr, beta1, beta2, eps, wd;
    float bc1, bc2;      // 1/(1−b1ᵗ), 1/(1−b2ᵗ)
    float gscale;        // global grad scale (1/accum · clip factor)
};
kernel void adamw_f32(
    device float*       p [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float*       m [[buffer(2)]],
    device float*       v [[buffer(3)]],
    constant AdamArgs&  a [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    float gr = g[gid] * a.gscale;
    float mm = a.beta1 * m[gid] + (1.0f - a.beta1) * gr;
    float vv = a.beta2 * v[gid] + (1.0f - a.beta2) * gr * gr;
    m[gid] = mm;
    v[gid] = vv;
    float upd = (mm * a.bc1) / (sqrt(vv * a.bc2) + a.eps);
    p[gid] -= a.lr * (upd + a.wd * p[gid]);
}

// Sum of squares of n floats into partial[tg] (one threadgroup = 256
// threads, 4 floats each per step) — the grad-norm clip reads the
// partials back and finishes on the host.
kernel void sumsq_f32(
    device const float* x       [[buffer(0)]],
    device float*       partial [[buffer(1)]],
    constant uint&      n       [[buffer(2)]],
    uint gid  [[thread_position_in_grid]],
    uint tpg  [[threads_per_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    float s = 0.0f;
    for (uint i = gid; i < n; i += tpg) { s += x[i] * x[i]; }
    s = simd_sum(s);
    if (lane == 0) { red[sgid] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float t = (lane < (tgs + 31u) / 32u) ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) { partial[tgid] = t; }
    }
}

// RMSNorm forward, one threadgroup (128 threads) per row of width d
// (d ≤ 128·4 handled by the loop). y = x · inv · w,  inv = 1/√(mean x²+eps).
// Stores inv per row for the backward.
kernel void rmsnorm_fwd_f32(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    device float*       inv [[buffer(3)]],
    constant uint&      d   [[buffer(4)]],
    constant float&     eps [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[4];
    device const float* xr = x + (ulong)row * d;
    device float* yr = y + (ulong)row * d;
    float s = 0.0f;
    for (uint i = tid; i < d; i += 128u) { float v = xr[i]; s += v * v; }
    s = simd_sum(s);
    if (lane == 0) { red[sgid] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3];
    float r = rsqrt(tot / (float)d + eps);
    if (tid == 0) { inv[row] = r; }
    for (uint i = tid; i < d; i += 128u) { yr[i] = xr[i] * r * w[i]; }
}

// RMSNorm backward (per row):  g = dy·w ;  dx = inv·(g − x·inv²·(g·x)/d)
// dw is accumulated on the host side via a GEMM-free column reduction
// kernel (rmsnorm_dw_f32) to keep this one race-free.
kernel void rmsnorm_bwd_dx_f32(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device const float* dy  [[buffer(2)]],
    device const float* inv [[buffer(3)]],
    device float*       dx  [[buffer(4)]],
    constant uint&      d   [[buffer(5)]],
    constant float&     beta [[buffer(6)]],   // dx = dx·beta + result
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[4];
    device const float* xr = x + (ulong)row * d;
    device const float* dyr = dy + (ulong)row * d;
    device float* dxr = dx + (ulong)row * d;
    float r = inv[row];
    float dot = 0.0f;
    for (uint i = tid; i < d; i += 128u) { dot += dyr[i] * w[i] * xr[i]; }
    dot = simd_sum(dot);
    if (lane == 0) { red[sgid] = dot; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3];
    float c = r * r * r * tot / (float)d;
    for (uint i = tid; i < d; i += 128u) {
        float v = r * dyr[i] * w[i] - c * xr[i];
        dxr[i] = (beta == 0.0f) ? v : (dxr[i] * beta + v);
    }
}

// dw[j] += Σ_rows dy[row,j]·x[row,j]·inv[row]   — one thread per column j,
// looping over rows (rows ~ B·T = 16k: fine for d ≤ 1k columns).
kernel void rmsnorm_dw_f32(
    device const float* x    [[buffer(0)]],
    device const float* dy   [[buffer(1)]],
    device const float* inv  [[buffer(2)]],
    device float*       dw   [[buffer(3)]],
    constant uint&      d    [[buffer(4)]],
    constant uint&      rows [[buffer(5)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= d) return;
    float s = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        s += dy[(ulong)r * d + j] * x[(ulong)r * d + j] * inv[r];
    }
    dw[j] += s;
}

// SwiGLU forward: h = silu(gate) · up   (elementwise over n)
kernel void swiglu_fwd_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       h    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float gv = gate[gid];
    float sg = gv / (1.0f + exp(-gv));
    h[gid] = sg * up[gid];
}

// SwiGLU backward: dgate = dh·up·silu'(gate) ; dup = dh·silu(gate)
kernel void swiglu_bwd_f32(
    device const float* gate  [[buffer(0)]],
    device const float* up    [[buffer(1)]],
    device const float* dh    [[buffer(2)]],
    device float*       dgate [[buffer(3)]],
    device float*       dup   [[buffer(4)]],
    constant uint&      n     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float gv = gate[gid];
    float sig = 1.0f / (1.0f + exp(-gv));
    float sg = gv * sig;
    float dsg = sig * (1.0f + gv * (1.0f - sig));
    float d = dh[gid];
    dgate[gid] = d * up[gid] * dsg;
    dup[gid] = d * sg;
}

// Embedding gather: out[row,:] = E[tok[row],:]  (d floats per row)
kernel void embed_gather_f32(
    device const float* E   [[buffer(0)]],
    device const uint*  tok [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])   // x: column, y: row
{
    if (gid.x >= d) return;
    out[(ulong)gid.y * d + gid.x] = E[(ulong)tok[gid.y] * d + gid.x];
}

// Fused softmax cross-entropy over a row of `n` logits, one threadgroup
// (256 threads) per row. Writes loss[row] = −log p[target] and, in
// place of the logits, dlogits = (p − onehot)·scale. Row-wise two-pass
// (max, then sum) in f32 with the max subtracted — n ≤ 64k.
kernel void softmax_ce_f32(
    device float*       logits [[buffer(0)]],
    device const uint*  target [[buffer(1)]],
    device float*       loss   [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant float&     scale  [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device float* lr = logits + (ulong)row * n;
    uint t0 = target[row];
    if (t0 == 0xFFFFFFFFu) {   // ignore: no loss, no gradient
        for (uint i = tid; i < n; i += 256u) lr[i] = 0.0f;
        if (tid == 0) loss[row] = 0.0f;
        return;
    }
    float mx = -INFINITY;
    for (uint i = tid; i < n; i += 256u) { mx = max(mx, lr[i]); }
    mx = simd_max(mx);
    if (lane == 0) { red[sgid] = mx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint s = 1; s < 8u; ++s) { mx = max(mx, red[s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256u) { sum += exp(lr[i] - mx); }
    sum = simd_sum(sum);
    if (lane == 0) { red[sgid] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) { tot += red[s]; }
    float lse = mx + log(tot);
    uint t = target[row];
    if (tid == 0) { loss[row] = lse - lr[t]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += 256u) {
        float p = exp(lr[i] - lse);
        lr[i] = (p - ((i == t) ? 1.0f : 0.0f)) * scale;
    }
}

// ---------------------------------------------------------------------
// hybrid_k mixer — chunked scan (chunk C = 64), the runtime's vmf_phase
// core + κ write gate (linear_core.rs::phase_step) made trainable.
//
//   S_t = γ ⊙ S_{t−1} + κ_t·φk_t ⊗ v_t ,  o_t = φq_tᵀ·S_t ,  φ = [cos θ; sin θ]
//
// Layouts (row-major, one row per (b,t)):
//   phq, phk : [B·T, nh·P2]    kv = κ⊙v, out, dout, dkv : [B·T, nh·dv]
//   pow      : [nh, C+1, P2]   γ_{h,f}^δ, δ = 0..C
//   states   : [B, nh, nchunks+1, P2, dv]  S entering chunk c (S_0 = 0)
//   dstates  : [B, nh, nchunks+1, P2, dv]  ∂L/∂S_c from chunks ≥ c only
// P2 ≤ 64 (nph ≤ 32), dv ≤ 128 (one thread per value channel).
// ---------------------------------------------------------------------

#define HK_C   64u
#define HK_P2  64u

struct HkArgs {
    uint B, T, nh, nph, dv;   // P2 = 2·nph
    uint carry;               // 1: the forward starts from checkpoint slot 0 (state carried across windows)
};

// φ tables from θ:  phq[row][h·P2 + i] = cos θ, [.. + nph + i] = sin θ
kernel void hk_phi_f32(
    device const float* th  [[buffer(0)]],
    device float*       ph  [[buffer(1)]],
    constant HkArgs&    a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·nph
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.nph;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / a.nph, i = r % a.nph;
    uint p2 = 2u * a.nph;
    float t = th[gid];
    ph[(ulong)row * a.nh * p2 + h * p2 + i]         = cos(t);
    ph[(ulong)row * a.nh * p2 + h * p2 + a.nph + i] = sin(t);
}

// Phase-Delta features use the exact unit-pair normalization selected by the
// P1 contract: ||[cos(theta),sin(theta)]/sqrt(nphase)|| = 1.
kernel void phase_delta_phi_f32(
    device const float* th  [[buffer(0)]],
    device float*       ph  [[buffer(1)]],
    constant HkArgs&    a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.nph;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / a.nph, i = r % a.nph;
    uint p2 = 2u * a.nph;
    float c = rsqrt((float)a.nph);
    float t = th[gid];
    ph[(ulong)row * a.nh * p2 + h * p2 + i]         = c * cos(t);
    ph[(ulong)row * a.nh * p2 + h * p2 + a.nph + i] = c * sin(t);
}

// In-place Phase-Delta forward. One value-channel thread owns one complete
// feature column, so the rank-1 correction is causal and race-free.
kernel void phase_delta_fwd_f32(
    device const float* phq   [[buffer(0)]],
    device const float* phk   [[buffer(1)]],
    device const float* v     [[buffer(2)]],
    device const float* kap   [[buffer(3)]],
    device const float* pow_t [[buffer(4)]],
    device float*       states[[buffer(5)]],
    device float*       out   [[buffer(6)]],
    constant HkArgs&    a     [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint nblocks = (a.dv + 31u) / 32u;
    if (tg >= a.B * a.nh * nblocks) return;
    uint block = tg % nblocks;
    uint d = block * 32u + lane, bh = tg / nblocks;
    const bool active = d < a.dv;
    uint b = bh / a.nh, h = bh % a.nh;
    uint p2 = 2u * a.nph;
    uint nch = (a.T + HK_C - 1u) / HK_C;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float S[HK_P2];
    ulong st_base = ((ulong)bh) * (nch + 1u) * p2 * a.dv;
    for (uint f = 0; f < p2; ++f)
        S[f] = active ? states[st_base + f * a.dv + d] : 0.0f;
    for (uint t = 0; t < a.T; ++t) {
        if ((t % HK_C) == 0u) {
            uint c = t / HK_C;
            if (active) {
                for (uint f = 0; f < p2; ++f)
                    states[st_base + ((ulong)c * p2 + f) * a.dv + d] = S[f];
            }
        }
        ulong row = ((ulong)b * a.T + t) * a.nh + h;
        device const float* qrow = phq + row * p2;
        device const float* krow = phk + row * p2;
        float kap_t = kap[row];
        float r = 0.0f;
        uint f = 0u;
        if (active) {
        for (; f + 4u <= p2; f += 4u) {
            float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
            float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
            float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
            float4 rr = kk * gg * ss;
            r += rr.x + rr.y + rr.z + rr.w;
        }
        for (; f < p2; ++f) r += krow[f] * (gam[f] * S[f]);
        }
        float4 vv = active ? float4(v[row * a.dv + d]) : float4(0.0f);
        f = 0u;
        if (active) for (; f + 4u <= p2; f += 4u) {
            float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
            float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
            float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
            ss = gg * ss + kap_t * kk * (vv - float4(r));
            S[f + 0u] = ss.x; S[f + 1u] = ss.y; S[f + 2u] = ss.z; S[f + 3u] = ss.w;
        }
        if (active) for (; f < p2; ++f)
            S[f] = gam[f] * S[f] + kap_t * krow[f] * (v[row * a.dv + d] - r);
        float o = 0.0f;
        f = 0u;
        if (active) for (; f + 4u <= p2; f += 4u) {
            float4 qq = float4(qrow[f + 0u], qrow[f + 1u], qrow[f + 2u], qrow[f + 3u]);
            float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
            float4 oo = qq * ss;
            o += oo.x + oo.y + oo.z + oo.w;
        }
        if (active) for (; f < p2; ++f) o += qrow[f] * S[f];
        if (active) out[row * a.dv + d] = o;
    }
    if (active) for (uint f = 0; f < p2; ++f)
        states[st_base + ((ulong)nch * p2 + f) * a.dv + d] = S[f];
}

// Replay one reverse chunk from its stored entry boundary.  The host encodes
// these kernels in descending chunk order, so the bounded chunk buffer is
// reused after every q reduction.  No output is produced: slots 0..clen hold
// S_{start}..S_{start+clen} for the following token-parallel reductions.
kernel void phase_delta_legacy_replay_f32(
    device const float* phk   [[buffer(0)]],
    device const float* v     [[buffer(1)]],
    device const float* kap   [[buffer(2)]],
    device const float* pow_t [[buffer(3)]],
    device const float* states[[buffer(4)]],
    device float*       chunk [[buffer(5)]],
    constant HkArgs&    a     [[buffer(6)]],
    constant uint&      chunk_id [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (tg >= a.B * a.nh) return;
    uint bh = tg;
    uint b = bh / a.nh, h = bh % a.nh;
    uint p2 = 2u * a.nph;
    uint nch = (a.T + HK_C - 1u) / HK_C;
    uint start = chunk_id * HK_C;
    uint clen = min(HK_C, a.T - start);
    uint d = tid;
    bool active = d < a.dv;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    ulong st_base = ((ulong)bh) * (nch + 1u) * p2 * a.dv;
    ulong ch_base = ((ulong)bh) * (HK_C + 1u) * p2 * a.dv;
    float S[HK_P2];
    for (uint f = 0; f < p2; ++f) {
        S[f] = active ? states[st_base + ((ulong)chunk_id * p2 + f) * a.dv + d] : 0.0f;
        if (active) chunk[ch_base + f * a.dv + d] = S[f];
    }
    for (uint j = 0; j < HK_C; ++j) {
        if (j >= clen) break;
        uint t = start + j;
        ulong row = ((ulong)b * a.T + t) * a.nh + h;
        device const float* krow = phk + row * p2;
        float r = 0.0f;
        if (active) {
            uint f = 0u;
            for (; f + 4u <= p2; f += 4u) {
                float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
                float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
                float4 rr = kk * gg * ss;
                r += rr.x + rr.y + rr.z + rr.w;
            }
            for (; f < p2; ++f) r += krow[f] * (gam[f] * S[f]);
            float kap_t = kap[row];
            f = 0u;
            for (; f + 4u <= p2; f += 4u) {
                float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
                float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
                ss = gg * ss + kap_t * kk * (float4(v[row * a.dv + d]) - float4(r));
                S[f + 0u] = ss.x; S[f + 1u] = ss.y; S[f + 2u] = ss.z; S[f + 3u] = ss.w;
            }
            for (; f < p2; ++f)
                S[f] = gam[f] * S[f] + kap_t * krow[f] * (v[row * a.dv + d] - r);
            for (f = 0u; f < p2; ++f)
                chunk[ch_base + ((ulong)(j + 1u) * p2 + f) * a.dv + d] = S[f];
        }
    }
}

// Token-parallel query-angle reduction.  One 32-lane SIMD group owns a small
// token tile; each lane covers four value channels and the SIMD sum is folded
// in fixed channel order.
// The chunk state has been replayed and is read before reverse overwrites it.
kernel void phase_delta_legacy_qreduce_f32(
    device const float* phq   [[buffer(0)]],
    device const float* dout  [[buffer(1)]],
    device const float* chunk [[buffer(2)]],
    device float*       dthq  [[buffer(3)]],
    constant HkArgs&    a     [[buffer(4)]],
    constant uint&      chunk_id [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint start = chunk_id * HK_C;
    uint clen = min(HK_C, a.T - start);
    const uint TOK = 4u;
    uint first = tg * TOK;
    uint total = a.B * a.nh * clen;
    if (first >= total) return;
    for (uint local = 0u; local < TOK; ++local) {
        uint item = first + local;
        if (item >= total) break;
        uint bh = item / clen, j = item % clen;
    uint b = bh / a.nh, h = bh % a.nh;
    uint p2 = 2u * a.nph;
    ulong row = ((ulong)b * a.T + start + j) * a.nh + h;
    ulong ch_base = ((ulong)bh) * (HK_C + 1u) * p2 * a.dv;
    device const float* qrow = phq + row * p2;
    device const float* drow = dout + row * a.dv;
    for (uint i = 0; i < a.nph; ++i) {
        float4 part = float4(0.0f);
        uint d0 = lane * 4u;
        for (uint n = 0; n < 4u; ++n) {
            uint d = d0 + n;
            if (d < a.dv) {
                float cur_c = chunk[ch_base + ((ulong)(j + 1u) * p2 + i) * a.dv + d];
                float cur_s = chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i) * a.dv + d];
                float z = drow[d];
                part[n] = (-qrow[a.nph + i] * cur_c + qrow[i] * cur_s) * z;
            }
        }
        float4 sum = simd_sum(part);
        if (lane == 0u) {
            ulong dst = row * a.nph;
            dthq[dst + i] = sum.x + sum.y + sum.z + sum.w;
        }
    }
    }
}

// Reverse one chunk for one 32-value block.  Incoming dS is read from the
// next chunk boundary and the resulting dS_c is written to dstates.  Once the
// q reduction has consumed S_t, that slot is reused for the per-value k
// feature gradients, avoiding a second full token×DV partial table.
kernel void phase_delta_legacy_bwd_chunk_f32(
    device const float* phq    [[buffer(0)]],
    device const float* phk    [[buffer(1)]],
    device const float* v      [[buffer(2)]],
    device const float* kap    [[buffer(3)]],
    device const float* pow_t  [[buffer(4)]],
    device const float* dout   [[buffer(5)]],
    device const float* states [[buffer(6)]],
    device float*       dstates[[buffer(7)]],
    device float*       chunk  [[buffer(8)]],
    device float*       dv_o   [[buffer(9)]],
    device float*       partial[[buffer(10)]],
    constant HkArgs&    a      [[buffer(11)]],
    constant uint&      chunk_id [[buffer(12)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint nblocks = (a.dv + 31u) / 32u;
    if (tg >= a.B * a.nh * nblocks) return;
    uint block = tg % nblocks, bh = tg / nblocks;
    uint b = bh / a.nh, h = bh % a.nh;
    uint p2 = 2u * a.nph;
    uint nch = (a.T + HK_C - 1u) / HK_C;
    uint start = chunk_id * HK_C;
    uint clen = min(HK_C, a.T - start);
    uint d = block * 32u + lane;
    bool active = d < a.dv;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    ulong st_base = ((ulong)bh) * (nch + 1u) * p2 * a.dv;
    ulong ch_base = ((ulong)bh) * (HK_C + 1u) * p2 * a.dv;
    float G[HK_P2];
    for (uint f = 0; f < p2; ++f)
        G[f] = active ? dstates[st_base + ((ulong)(chunk_id + 1u) * p2 + f) * a.dv + d] : 0.0f;
    for (uint rev = 0; rev < HK_C; ++rev) {
        if (rev >= clen) break;
        uint j = clen - 1u - rev;
        uint t = start + j;
        ulong row = ((ulong)b * a.T + t) * a.nh + h;
        device const float* qrow = phq + row * p2;
        device const float* krow = phk + row * p2;
        float r = 0.0f;
        if (active) {
            uint f = 0u;
            for (; f + 4u <= p2; f += 4u) {
                float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                float4 ss = float4(
                    chunk[ch_base + ((ulong)j * p2 + f + 0u) * a.dv + d],
                    chunk[ch_base + ((ulong)j * p2 + f + 1u) * a.dv + d],
                    chunk[ch_base + ((ulong)j * p2 + f + 2u) * a.dv + d],
                    chunk[ch_base + ((ulong)j * p2 + f + 3u) * a.dv + d]);
                float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
                float4 rr = kk * gg * ss;
                r += rr.x + rr.y + rr.z + rr.w;
            }
            for (; f < p2; ++f)
                r += krow[f] * (gam[f] * chunk[ch_base + ((ulong)j * p2 + f) * a.dv + d]);
        }
        float e = active ? v[row * a.dv + d] - r : 0.0f;
        float od = active ? dout[row * a.dv + d] : 0.0f;
        float u = 0.0f;
        if (active) {
            uint f = 0u;
            for (; f + 4u <= p2; f += 4u) {
                float4 gt = float4(
                    G[f + 0u] + qrow[f + 0u] * od,
                    G[f + 1u] + qrow[f + 1u] * od,
                    G[f + 2u] + qrow[f + 2u] * od,
                    G[f + 3u] + qrow[f + 3u] * od);
                float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                float4 uu = gt * kk;
                u += uu.x + uu.y + uu.z + uu.w;
            }
            for (; f < p2; ++f) u += (G[f] + qrow[f] * od) * krow[f];
            float kap_t = kap[row];
            dv_o[row * a.dv + d] = kap_t * u;
        }
        float beta_part = simd_sum(u * e);
        if (lane == 0u)
            partial[((ulong)bh * nblocks + block) * a.T + t] = beta_part;
        if (active) {
            uint f = 0u;
            float kap_t = kap[row];
            for (; f + 4u <= p2; f += 4u) {
                float4 gc = float4(
                    G[f + 0u] + qrow[f + 0u] * od,
                    G[f + 1u] + qrow[f + 1u] * od,
                    G[f + 2u] + qrow[f + 2u] * od,
                    G[f + 3u] + qrow[f + 3u] * od);
                float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                float4 pp = float4(
                    gam[f + 0u] * chunk[ch_base + ((ulong)j * p2 + f + 0u) * a.dv + d],
                    gam[f + 1u] * chunk[ch_base + ((ulong)j * p2 + f + 1u) * a.dv + d],
                    gam[f + 2u] * chunk[ch_base + ((ulong)j * p2 + f + 2u) * a.dv + d],
                    gam[f + 3u] * chunk[ch_base + ((ulong)j * p2 + f + 3u) * a.dv + d]);
                float4 gk = kap_t * (gc * e - pp * u);
                chunk[ch_base + ((ulong)(j + 1u) * p2 + f + 0u) * a.dv + d] = gk.x;
                chunk[ch_base + ((ulong)(j + 1u) * p2 + f + 1u) * a.dv + d] = gk.y;
                chunk[ch_base + ((ulong)(j + 1u) * p2 + f + 2u) * a.dv + d] = gk.z;
                chunk[ch_base + ((ulong)(j + 1u) * p2 + f + 3u) * a.dv + d] = gk.w;
                float4 gn = kk * (kap_t * u);
                G[f + 0u] = gam[f + 0u] * (gc.x - gn.x);
                G[f + 1u] = gam[f + 1u] * (gc.y - gn.y);
                G[f + 2u] = gam[f + 2u] * (gc.z - gn.z);
                G[f + 3u] = gam[f + 3u] * (gc.w - gn.w);
            }
            for (; f < p2; ++f) {
                float gt = G[f] + qrow[f] * od;
                float gk = kap_t * (gt * e - gam[f] * chunk[ch_base + ((ulong)j * p2 + f) * a.dv + d] * u);
                chunk[ch_base + ((ulong)(j + 1u) * p2 + f) * a.dv + d] = gk;
                G[f] = gam[f] * (gt - krow[f] * kap_t * u);
            }
        }
    }
    if (active) for (uint f = 0; f < p2; ++f)
        dstates[st_base + ((ulong)chunk_id * p2 + f) * a.dv + d] = G[f];
}

// Token-parallel key-angle reduction over the per-value gradients written by
// Legacy chunk reverse path retained only as source archaeology; the host no
// longer creates a pipeline for it. The fixed SIMD-group order mirrors q reduction.
kernel void phase_delta_legacy_kreduce_f32(
    device const float* phk   [[buffer(0)]],
    device const float* chunk [[buffer(1)]],
    device float*       dthk  [[buffer(2)]],
    constant HkArgs&    a     [[buffer(3)]],
    constant uint&      chunk_id [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint start = chunk_id * HK_C;
    uint clen = min(HK_C, a.T - start);
    const uint TOK = 4u;
    uint first = tg * TOK;
    uint total = a.B * a.nh * clen;
    if (first >= total) return;
    for (uint local = 0u; local < TOK; ++local) {
        uint item = first + local;
        if (item >= total) break;
        uint bh = item / clen, j = item % clen;
    uint b = bh / a.nh, h = bh % a.nh;
    uint p2 = 2u * a.nph;
    ulong row = ((ulong)b * a.T + start + j) * a.nh + h;
    ulong ch_base = ((ulong)bh) * (HK_C + 1u) * p2 * a.dv;
    device const float* krow = phk + row * p2;
    for (uint i = 0; i < a.nph; ++i) {
        float4 part = float4(0.0f);
        uint d0 = lane * 4u;
        for (uint n = 0; n < 4u; ++n) {
            uint d = d0 + n;
            if (d < a.dv) {
                float gc = chunk[ch_base + ((ulong)(j + 1u) * p2 + i) * a.dv + d];
                float gs = chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i) * a.dv + d];
                part[n] = -krow[a.nph + i] * gc + krow[i] * gs;
            }
        }
        float4 sum = simd_sum(part);
        if (lane == 0u) {
            ulong dst = row * a.nph;
            dthk[dst + i] = sum.x + sum.y + sum.z + sum.w;
        }
    }
    }
}

// Fold beta partials in ascending value-block order.  This remains a
// separate kernel so its deterministic reduction is visible in profiling.
kernel void phase_delta_legacy_betafold_f32(
    device const float* partial [[buffer(0)]],
    device float*       dkap    [[buffer(1)]],
    constant HkArgs&    a       [[buffer(2)]],
    constant uint&      chunk_id [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint start = chunk_id * HK_C;
    uint clen = min(HK_C, a.T - start);
    uint total = a.B * a.nh * clen;
    if (gid >= total) return;
    uint bh = gid / clen, j = gid % clen;
    uint b = bh / a.nh, h = bh % a.nh;
    uint nblocks = (a.dv + 31u) / 32u;
    float s = 0.0f;
    for (uint block = 0; block < nblocks; ++block)
        s += partial[((ulong)bh * nblocks + block) * a.T + start + j];
    dkap[((ulong)b * a.T + start + j) * a.nh + h] = s;
}

// Exact reverse recurrence by 64-token chunks. One 32-lane SIMD group owns a
// 32-value block of one (batch,head) pair. Each block reconstructs only the
// current chunk from its stored entry boundary into the shared bounded
// [B,nh,65,P2,dv] scratch, scans it backwards, and emits one deterministic
// beta/q/k partial per token into [B,nh,nblocks,T,1+2*nph]. Blocks never
// synchronize with each other; phase_delta_fold_f32 performs the fixed-order
// reduction after this kernel. Arithmetic remains O(T*P2*dv), with no
// full-token state history and no per-token threadgroup barriers.
kernel void phase_delta_bwd_f32(
    device const float* thq    [[buffer(0)]],
    device const float* thk    [[buffer(1)]],
    device const float* phq    [[buffer(2)]],
    device const float* phk    [[buffer(3)]],
    device const float* v      [[buffer(4)]],
    device const float* kap    [[buffer(5)]],
    device const float* pow_t  [[buffer(6)]],
    device const float* dout   [[buffer(7)]],
    device float*       dthq  [[buffer(8)]],
    device float*       dthk  [[buffer(9)]],
    device float*       dv_o   [[buffer(10)]],
    device float*       dkap  [[buffer(11)]],
    device const float* states [[buffer(12)]],
    device float*       dstates[[buffer(13)]],
    device float*       chunk  [[buffer(14)]],
    device float*       partial[[buffer(15)]],
    constant HkArgs&    a      [[buffer(16)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint nblocks = (a.dv + 31u) / 32u;
    if (tg >= a.B * a.nh * nblocks) return;
    uint block = tg % nblocks;
    uint bh = tg / nblocks;
    uint b = bh / a.nh, h = bh % a.nh;
    uint p2 = 2u * a.nph;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    uint nch = (a.T + HK_C - 1u) / HK_C;
    ulong st_base = ((ulong)bh) * (nch + 1u) * p2 * a.dv;
    ulong ch_base = ((ulong)bh) * (HK_C + 1u) * p2 * a.dv;
    const uint d = block * 32u + lane;
    const bool active = d < a.dv;
    float G[HK_P2];
    for (uint f = 0; f < p2; ++f) G[f] = 0.0f;
    for (uint rev_c = 0; rev_c < nch; ++rev_c) {
        uint c = nch - 1u - rev_c;
        uint start = c * HK_C;
        uint clen = min(HK_C, a.T - start);
        // Recover S_{start..start+clen} from this chunk's entry boundary.
        // Only this bounded scratch is materialized; all earlier chunks stay
        // represented by their stored entry boundaries.
        float S[HK_P2];
        for (uint f = 0; f < p2; ++f) {
            S[f] = active ? states[st_base + ((ulong)c * p2 + f) * a.dv + d] : 0.0f;
            if (active) chunk[ch_base + f * a.dv + d] = S[f];
        }
        for (uint j = 0; j < HK_C; ++j) {
            if (j >= clen) break;
            uint t = start + j;
            ulong row = ((ulong)b * a.T + t) * a.nh + h;
            device const float* qrow = phq + row * p2;
            device const float* krow = phk + row * p2;
            float r = 0.0f;
            if (active) {
                uint f = 0u;
                for (; f + 4u <= p2; f += 4u) {
                    float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                    float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
                    float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
                    float4 rr = kk * gg * ss;
                    r += rr.x + rr.y + rr.z + rr.w;
                }
                for (; f < p2; ++f) r += krow[f] * (gam[f] * S[f]);
                float4 vv = float4(v[row * a.dv + d]);
                f = 0u;
                for (; f + 4u <= p2; f += 4u) {
                    float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                    float4 ss = float4(S[f + 0u], S[f + 1u], S[f + 2u], S[f + 3u]);
                    float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
                    ss = gg * ss + kap[row] * kk * (vv - float4(r));
                    S[f + 0u] = ss.x; S[f + 1u] = ss.y; S[f + 2u] = ss.z; S[f + 3u] = ss.w;
                }
                for (; f < p2; ++f)
                    S[f] = gam[f] * S[f] + kap[row] * krow[f] * (v[row * a.dv + d] - r);
                for (f = 0u; f < p2; ++f)
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + f) * a.dv + d] = S[f];
            }
        }
        // Reverse the recovered chunk, carrying dS across its entry boundary.
        for (uint rev_j = 0; rev_j < HK_C; ++rev_j) {
            if (rev_j >= clen) break;
            uint j = clen - 1u - rev_j;
            uint t = start + j;
            ulong row = ((ulong)b * a.T + t) * a.nh + h;
            device const float* qrow = phq + row * p2;
            device const float* krow = phk + row * p2;
            float r = 0.0f;
            if (active) {
                uint f = 0u;
                for (; f + 4u <= p2; f += 4u) {
                    float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                    float4 ss = float4(
                        chunk[ch_base + ((ulong)j * p2 + f + 0u) * a.dv + d],
                        chunk[ch_base + ((ulong)j * p2 + f + 1u) * a.dv + d],
                        chunk[ch_base + ((ulong)j * p2 + f + 2u) * a.dv + d],
                        chunk[ch_base + ((ulong)j * p2 + f + 3u) * a.dv + d]);
                    float4 gg = float4(gam[f + 0u], gam[f + 1u], gam[f + 2u], gam[f + 3u]);
                    float4 rr = kk * gg * ss;
                    r += rr.x + rr.y + rr.z + rr.w;
                }
                for (; f < p2; ++f)
                    r += krow[f] * (gam[f] * chunk[ch_base + ((ulong)j * p2 + f) * a.dv + d]);
            }
            float e = active ? v[row * a.dv + d] - r : 0.0f;
            float u = 0.0f;
            float dv_t = active ? dout[row * a.dv + d] : 0.0f;
            if (active) {
                uint f = 0u;
                for (; f + 4u <= p2; f += 4u) {
                    float4 gg = float4(
                        G[f + 0u] + qrow[f + 0u] * dv_t,
                        G[f + 1u] + qrow[f + 1u] * dv_t,
                        G[f + 2u] + qrow[f + 2u] * dv_t,
                        G[f + 3u] + qrow[f + 3u] * dv_t);
                    float4 kk = float4(krow[f + 0u], krow[f + 1u], krow[f + 2u], krow[f + 3u]);
                    float4 uu = gg * kk;
                    u += uu.x + uu.y + uu.z + uu.w;
                }
                for (; f < p2; ++f)
                    u += (G[f] + qrow[f] * dv_t) * krow[f];
                dv_o[row * a.dv + d] = kap[row] * u;
            }
            // Every lane participates in the SIMD sum; inactive tail lanes
            // contribute zero when dv is not a multiple of 32.
            float beta_part = simd_sum(u * e);
            ulong pbase = (((ulong)bh * nblocks + block) * a.T + t) * (1u + 2u * a.nph);
            if (lane == 0u) partial[pbase] = beta_part;
            uint i = 0u;
            for (; i + 4u <= a.nph; i += 4u) {
                float4 cur_c = active ? float4(
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + i + 0u) * a.dv + d],
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + i + 1u) * a.dv + d],
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + i + 2u) * a.dv + d],
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + i + 3u) * a.dv + d]) : float4(0.0f);
                float4 cur_s = active ? float4(
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i + 0u) * a.dv + d],
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i + 1u) * a.dv + d],
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i + 2u) * a.dv + d],
                    chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i + 3u) * a.dv + d]) : float4(0.0f);
                float4 gc = active ? float4(
                    G[i + 0u] + qrow[i + 0u] * dv_t,
                    G[i + 1u] + qrow[i + 1u] * dv_t,
                    G[i + 2u] + qrow[i + 2u] * dv_t,
                    G[i + 3u] + qrow[i + 3u] * dv_t) : float4(0.0f);
                float4 gs = active ? float4(
                    G[a.nph + i + 0u] + qrow[a.nph + i + 0u] * dv_t,
                    G[a.nph + i + 1u] + qrow[a.nph + i + 1u] * dv_t,
                    G[a.nph + i + 2u] + qrow[a.nph + i + 2u] * dv_t,
                    G[a.nph + i + 3u] + qrow[a.nph + i + 3u] * dv_t) : float4(0.0f);
                float4 gkc = active ? kap[row] * (gc * e - float4(
                    gam[i + 0u] * chunk[ch_base + ((ulong)j * p2 + i + 0u) * a.dv + d] * u,
                    gam[i + 1u] * chunk[ch_base + ((ulong)j * p2 + i + 1u) * a.dv + d] * u,
                    gam[i + 2u] * chunk[ch_base + ((ulong)j * p2 + i + 2u) * a.dv + d] * u,
                    gam[i + 3u] * chunk[ch_base + ((ulong)j * p2 + i + 3u) * a.dv + d] * u)) : float4(0.0f);
                float4 gks = active ? kap[row] * (gs * e - float4(
                    gam[a.nph + i + 0u] * chunk[ch_base + ((ulong)j * p2 + a.nph + i + 0u) * a.dv + d] * u,
                    gam[a.nph + i + 1u] * chunk[ch_base + ((ulong)j * p2 + a.nph + i + 1u) * a.dv + d] * u,
                    gam[a.nph + i + 2u] * chunk[ch_base + ((ulong)j * p2 + a.nph + i + 2u) * a.dv + d] * u,
                    gam[a.nph + i + 3u] * chunk[ch_base + ((ulong)j * p2 + a.nph + i + 3u) * a.dv + d] * u)) : float4(0.0f);
                float4 qpart = active ? -float4(
                    qrow[a.nph + i + 0u], qrow[a.nph + i + 1u],
                    qrow[a.nph + i + 2u], qrow[a.nph + i + 3u]) * (cur_c * dv_t)
                    + float4(qrow[i + 0u], qrow[i + 1u],
                        qrow[i + 2u], qrow[i + 3u]) * (cur_s * dv_t) : float4(0.0f);
                float4 kpart = active ? -float4(
                    krow[a.nph + i + 0u], krow[a.nph + i + 1u],
                    krow[a.nph + i + 2u], krow[a.nph + i + 3u]) * gkc
                    + float4(krow[i + 0u], krow[i + 1u],
                        krow[i + 2u], krow[i + 3u]) * gks : float4(0.0f);
                float4 qsum = simd_sum(qpart);
                float4 ksum = simd_sum(kpart);
                if (lane == 0u) {
                    partial[pbase + 1u + i + 0u] = qsum.x;
                    partial[pbase + 1u + i + 1u] = qsum.y;
                    partial[pbase + 1u + i + 2u] = qsum.z;
                    partial[pbase + 1u + i + 3u] = qsum.w;
                    partial[pbase + 1u + a.nph + i + 0u] = ksum.x;
                    partial[pbase + 1u + a.nph + i + 1u] = ksum.y;
                    partial[pbase + 1u + a.nph + i + 2u] = ksum.z;
                    partial[pbase + 1u + a.nph + i + 3u] = ksum.w;
                }
                if (active) {
                    float4 gn_c = float4(krow[i + 0u], krow[i + 1u], krow[i + 2u], krow[i + 3u]) * (kap[row] * u);
                    float4 gn_s = float4(krow[a.nph + i + 0u], krow[a.nph + i + 1u], krow[a.nph + i + 2u], krow[a.nph + i + 3u]) * (kap[row] * u);
                    float4 gm_c = float4(gam[i + 0u], gam[i + 1u], gam[i + 2u], gam[i + 3u]);
                    float4 gm_s = float4(gam[a.nph + i + 0u], gam[a.nph + i + 1u], gam[a.nph + i + 2u], gam[a.nph + i + 3u]);
                    float4 new_c = gm_c * (gc - gn_c);
                    float4 new_s = gm_s * (gs - gn_s);
                    G[i + 0u] = new_c.x; G[i + 1u] = new_c.y; G[i + 2u] = new_c.z; G[i + 3u] = new_c.w;
                    G[a.nph + i + 0u] = new_s.x; G[a.nph + i + 1u] = new_s.y; G[a.nph + i + 2u] = new_s.z; G[a.nph + i + 3u] = new_s.w;
                }
            }
            for (; i < a.nph; ++i) {
                float cur_c = active ? chunk[ch_base + ((ulong)(j + 1u) * p2 + i) * a.dv + d] : 0.0f;
                float cur_s = active ? chunk[ch_base + ((ulong)(j + 1u) * p2 + a.nph + i) * a.dv + d] : 0.0f;
                float gkc = active ? kap[row] * ((G[i] + qrow[i] * dv_t) * e
                    - gam[i] * chunk[ch_base + ((ulong)j * p2 + i) * a.dv + d] * u) : 0.0f;
                float gks = active ? kap[row] * ((G[a.nph + i] + qrow[a.nph + i] * dv_t) * e
                    - gam[a.nph + i] * chunk[ch_base + ((ulong)j * p2 + a.nph + i) * a.dv + d] * u) : 0.0f;
                // phq/phk already contain cos/sin divided by sqrt(nphase),
                // so this is exactly dtheta = -sin(theta)dphi_c +
                // cos(theta)dphi_s without reverse-loop trigonometry.
                float q_part = active ? -qrow[a.nph + i] * (cur_c * dv_t)
                    + qrow[i] * (cur_s * dv_t) : 0.0f;
                float k_part = active ? -krow[a.nph + i] * gkc
                    + krow[i] * gks : 0.0f;
                float q_sum = simd_sum(q_part);
                float k_sum = simd_sum(k_part);
                if (lane == 0u) {
                    partial[pbase + 1u + i] = q_sum;
                    partial[pbase + 1u + a.nph + i] = k_sum;
                }
                if (active) {
                    G[i] = gam[i] * (G[i] + qrow[i] * dv_t - krow[i] * kap[row] * u);
                    G[a.nph + i] = gam[a.nph + i] * (G[a.nph + i] + qrow[a.nph + i] * dv_t - krow[a.nph + i] * kap[row] * u);
                }
            }
        }
        // Boundary c now contains ∂L/∂S_c, useful for diagnostics and for
        // exact continuation semantics even when the initial state is nonzero.
        if (active) {
            for (uint f = 0; f < p2; ++f)
                dstates[st_base + ((ulong)c * p2 + f) * a.dv + d] = G[f];
        }
    }
    // There is no future loss after the final output in the normal trainer;
    // retain the terminal boundary gradient as zero for an explicit witness.
    if (active) {
        for (uint f = 0; f < p2; ++f)
            dstates[st_base + ((ulong)nch * p2 + f) * a.dv + d] = 0.0f;
    }
}

// Fold the at-most-four value blocks in ascending order. Each output token
// and head has one owner, so the final gradients are race-free and repeatable.
kernel void phase_delta_fold_f32(
    device const float* partial [[buffer(0)]],
    device float*       dthq    [[buffer(1)]],
    device float*       dthk    [[buffer(2)]],
    device float*       dkap    [[buffer(3)]],
    constant HkArgs&    a       [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = a.B * a.T * a.nh;
    if (gid >= total) return;
    uint row = gid / a.nh, h = gid % a.nh;
    uint b = row / a.T, t = row % a.T;
    uint nblocks = (a.dv + 31u) / 32u;
    uint width = 1u + 2u * a.nph;
    float sb = 0.0f;
    for (uint block = 0; block < nblocks; ++block) {
        ulong base = (((ulong)(b * a.nh + h) * nblocks + block) * a.T + t) * width;
        sb += partial[base];
    }
    dkap[(ulong)row * a.nh + h] = sb;
    ulong angle = ((ulong)row * a.nh + h) * a.nph;
    uint i = 0u;
    for (; i + 4u <= a.nph; i += 4u) {
        float4 sq = float4(0.0f), sk = float4(0.0f);
        for (uint block = 0; block < nblocks; ++block) {
            ulong base = (((ulong)(b * a.nh + h) * nblocks + block) * a.T + t) * width;
            sq += float4(partial[base + 1u + i + 0u], partial[base + 1u + i + 1u], partial[base + 1u + i + 2u], partial[base + 1u + i + 3u]);
            sk += float4(partial[base + 1u + a.nph + i + 0u], partial[base + 1u + a.nph + i + 1u], partial[base + 1u + a.nph + i + 2u], partial[base + 1u + a.nph + i + 3u]);
        }
        dthq[angle + i + 0u] = sq.x; dthq[angle + i + 1u] = sq.y;
        dthq[angle + i + 2u] = sq.z; dthq[angle + i + 3u] = sq.w;
        dthk[angle + i + 0u] = sk.x; dthk[angle + i + 1u] = sk.y;
        dthk[angle + i + 2u] = sk.z; dthk[angle + i + 3u] = sk.w;
    }
    for (; i < a.nph; ++i) {
        float sq = 0.0f, sk = 0.0f;
        for (uint block = 0; block < nblocks; ++block) {
            ulong base = (((ulong)(b * a.nh + h) * nblocks + block) * a.T + t) * width;
            sq += partial[base + 1u + i];
            sk += partial[base + 1u + a.nph + i];
        }
        dthq[angle + i] = sq;
        dthk[angle + i] = sk;
    }
}

// ---------------------------------------------------------------------
// dθ from dφ:  dθ_i = −sin θ_i·dφ[i] + cos θ_i·dφ[nph+i]
kernel void hk_dtheta_f32(
    device const float* th  [[buffer(0)]],
    device const float* dph [[buffer(1)]],
    device float*       dth [[buffer(2)]],
    constant HkArgs&    a   [[buffer(3)]],
    constant float&     beta [[buffer(4)]],   // dth = beta·dth + result
    uint gid [[thread_position_in_grid]])
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.nph;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / a.nph, i = r % a.nph;
    uint p2 = 2u * a.nph;
    float t = th[gid];
    ulong base = (ulong)row * a.nh * p2 + h * p2;
    float g = -sin(t) * dph[base + i] + cos(t) * dph[base + a.nph + i];
    dth[gid] = (beta == 0.0f) ? g : (beta * dth[gid] + g);
}

// kv = κ ⊙ v  (κ per (row, head))
kernel void hk_kv_f32(
    device const float* v   [[buffer(0)]],
    device const float* kap [[buffer(1)]],
    device float*       kv  [[buffer(2)]],
    constant HkArgs&    a   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·dv
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.dv;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, h = (gid % per_row) / a.dv;
    kv[gid] = v[gid] * kap[row * a.nh + h];
}

// dv = κ⊙dkv ;  dκ = Σ_d dkv·v   (one thread per (row, head))
kernel void hk_dkv_split_f32(
    device const float* v    [[buffer(0)]],
    device const float* kap  [[buffer(1)]],
    device const float* dkv  [[buffer(2)]],
    device float*       dv_o [[buffer(3)]],
    device float*       dkap [[buffer(4)]],
    constant HkArgs&    a    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh
{
    if (gid >= a.B * a.T * a.nh) return;
    uint row = gid / a.nh, h = gid % a.nh;
    float k = kap[gid];
    ulong base = (ulong)row * a.nh * a.dv + h * a.dv;
    float s = 0.0f;
    for (uint d = 0; d < a.dv; ++d) {
        float g = dkv[base + d];
        dv_o[base + d] = k * g;
        s += g * v[base + d];
    }
    dkap[gid] = s;
}

// Forward states: one threadgroup per (b,h), threads = dv × FS where each
// thread owns value channel d and a 16-feature slice fg (FS = ceil(P2/16))
// — 16 accumulators per thread keeps the register file small enough for
// several threadgroups per core; φk of the chunk is staged once per
// threadgroup. The literal recurrence; S at every chunk boundary written.
#define HK_FT 16u
kernel void hk_states_fwd_f32(
    device const float* phk    [[buffer(0)]],
    device const float* kv     [[buffer(1)]],
    device const float* pow_t  [[buffer(2)]],
    device float*       states [[buffer(3)]],
    constant HkArgs&    a      [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float sk[HK_C * HK_P2];   // φk of the current chunk
    uint b = tg / a.nh, h = tg % a.nh;
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint d = tid % a.dv, fg = tid / a.dv;
    uint f0 = fg * HK_FT;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float S[HK_FT], gm[HK_FT];
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) {
        // carry: S_0 is the state left in checkpoint slot 0 by the caller
        S[i] = (a.carry != 0u && f0 + i < p2) ? states[st_base + ((ulong)(f0 + i)) * a.dv + d] : 0.0f;
        gm[i] = (f0 + i < p2) ? gam[f0 + i] : 0.0f;
    }
    for (uint c = 0; c < nchunks; ++c) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < HK_FT; ++i) {
            if (f0 + i < p2) states[st_base + ((ulong)c * p2 + f0 + i) * a.dv + d] = S[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < HK_C * HK_P2; i += nth) {
            uint s = i / HK_P2, f = i % HK_P2;
            sk[i] = (f < p2) ? phk[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * p2 + f] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < HK_C; ++s) {
            float kvv = kv[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * a.dv + d];
            threadgroup const float* skr = sk + s * HK_P2 + f0;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < HK_FT; ++i) S[i] = gm[i] * S[i] + skr[i] * kvv;
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) {
        if (f0 + i < p2) states[st_base + ((ulong)nchunks * p2 + f0 + i) * a.dv + d] = S[i];
    }
}

// Forward per chunk: one threadgroup per (b,h,c), nth threads (≥ dv).
//   A[t,s] = Σ_f φq_t[f]·φk_s[f]·γ_f^{t−s}  (s ≤ t)
//   o_t[d] = Σ_{s≤t} A[t,s]·kv_s[d] + Σ_f φq_t[f]·γ_f^{t−t0+1}·S_c[f][d]
kernel void hk_chunk_fwd_f32(
    device const float* phq    [[buffer(0)]],
    device const float* phk    [[buffer(1)]],
    device const float* kv     [[buffer(2)]],
    device const float* pow_t  [[buffer(3)]],
    device const float* states [[buffer(4)]],
    device float*       out    [[buffer(5)]],
    constant HkArgs&    a      [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float A[HK_C * HK_C];
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint c = tg % nchunks, bh = tg / nchunks;
    uint b = bh / a.nh, h = bh % a.nh;
    uint t0 = c * HK_C;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;   // pw[δ·p2 + f]
    // stage 1: A
    for (uint idx = tid; idx < HK_C * HK_C; idx += nth) {
        uint t = idx / HK_C, s = idx % HK_C;
        float acc = 0.0f;
        if (s <= t) {
            device const float* q = phq + ((ulong)(b * a.T + t0 + t) * a.nh + h) * p2;
            device const float* k = phk + ((ulong)(b * a.T + t0 + s) * a.nh + h) * p2;
            device const float* g = pw + (t - s) * p2;
            for (uint f = 0; f < p2; ++f) acc += q[f] * k[f] * g[f];
        }
        A[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 2: outputs, thread per d
    if (tid < a.dv) {
        uint d = tid;
        float Sc[HK_P2];
        device const float* st = states + (((ulong)bh * (nchunks + 1u) + c) * p2) * a.dv + d;
        for (uint f = 0; f < p2; ++f) Sc[f] = st[(ulong)f * a.dv];
        for (uint t = 0; t < HK_C; ++t) {
            float o = 0.0f;
            for (uint s = 0; s <= t; ++s) {
                o += A[t * HK_C + s] * kv[((ulong)(b * a.T + t0 + s) * a.nh + h) * a.dv + d];
            }
            device const float* q = phq + ((ulong)(b * a.T + t0 + t) * a.nh + h) * p2;
            device const float* g = pw + (t + 1u) * p2;
            for (uint f = 0; f < p2; ++f) o += q[f] * g[f] * Sc[f];
            out[((ulong)(b * a.T + t0 + t) * a.nh + h) * a.dv + d] = o;
        }
    }
}

// Backward state gradients (reverse over positions), same thread layout
// as hk_states_fwd_f32:
//   G ← γ ⊙ (G + φq_t ⊗ do_t)   for t = T−1 … 0,
// G at each chunk boundary written to dstates[c] = ∂L/∂S_c restricted
// to reads by chunks ≥ c (dstates[nchunks] = 0).
kernel void hk_dstates_bwd_f32(
    device const float* phq     [[buffer(0)]],
    device const float* dout    [[buffer(1)]],
    device const float* pow_t   [[buffer(2)]],
    device float*       dstates [[buffer(3)]],
    constant HkArgs&    a       [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float sq[HK_C * HK_P2];
    uint b = tg / a.nh, h = tg % a.nh;
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint d = tid % a.dv, fg = tid / a.dv;
    uint f0 = fg * HK_FT;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float G[HK_FT], gm[HK_FT];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) { G[i] = 0.0f; gm[i] = (f0 + i < p2) ? gam[f0 + i] : 0.0f; }
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) {
        if (f0 + i < p2) dstates[st_base + ((ulong)nchunks * p2 + f0 + i) * a.dv + d] = 0.0f;
    }
    for (uint cc = 0; cc < nchunks; ++cc) {
        uint c = nchunks - 1u - cc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < HK_C * HK_P2; i += nth) {
            uint s = i / HK_P2, f = i % HK_P2;
            sq[i] = (f < p2) ? phq[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * p2 + f] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ss = 0; ss < HK_C; ++ss) {
            uint s = HK_C - 1u - ss;
            float dov = dout[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * a.dv + d];
            threadgroup const float* sqr = sq + s * HK_P2 + f0;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < HK_FT; ++i) G[i] = gm[i] * (G[i] + sqr[i] * dov);
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < HK_FT; ++i) {
            if (f0 + i < p2) dstates[st_base + ((ulong)c * p2 + f0 + i) * a.dv + d] = G[i];
        }
    }
}

// Backward per chunk. Writes dkv (→ hk_dkv_split), dphq/dphk (→ hk_dtheta).
//   dkv_s[d]  = Σ_{t≥s} A[t,s]·do_t[d] + Σ_f φk_s[f]·γ_f^{C−1−s}·Gn[f][d]
//   dA[t,s]   = Σ_d do_t[d]·kv_s[d]
//   dφq_t[f]  = Σ_{s≤t} dA[t,s]·φk_s[f]·γ_f^{t−s} + γ_f^{t+1}·Σ_d do_t[d]·Sc[f][d]
//   dφk_s[f]  = Σ_{t≥s} dA[t,s]·φq_t[f]·γ_f^{t−s} + γ_f^{C−1−s}·Σ_d kv_s[d]·Gn[f][d]
// (t, s relative to the chunk; Sc = states[c], Gn = dstates[c+1])
kernel void hk_chunk_bwd_f32(
    device const float* phq     [[buffer(0)]],
    device const float* phk     [[buffer(1)]],
    device const float* kv      [[buffer(2)]],
    device const float* pow_t   [[buffer(3)]],
    device const float* states  [[buffer(4)]],
    device const float* dstates [[buffer(5)]],
    device const float* dout    [[buffer(6)]],
    device float*       dkv     [[buffer(7)]],
    device float*       dphq    [[buffer(8)]],
    device float*       dphk    [[buffer(9)]],
    constant HkArgs&    a       [[buffer(10)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float A[HK_C * HK_C];    // A, then reused for dA
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint c = tg % nchunks, bh = tg / nchunks;
    uint b = bh / a.nh, h = bh % a.nh;
    uint t0 = c * HK_C;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    #define ROW(t) ((ulong)(b * a.T + t0 + (t)) * a.nh + h)
    // stage 1: A
    for (uint idx = tid; idx < HK_C * HK_C; idx += nth) {
        uint t = idx / HK_C, s = idx % HK_C;
        float acc = 0.0f;
        if (s <= t) {
            device const float* q = phq + ROW(t) * p2;
            device const float* k = phk + ROW(s) * p2;
            device const float* g = pw + (t - s) * p2;
            for (uint f = 0; f < p2; ++f) acc += q[f] * k[f] * g[f];
        }
        A[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 2: dkv, thread per d
    if (tid < a.dv) {
        uint d = tid;
        float Gn[HK_P2];
        device const float* gn = dstates + (((ulong)bh * (nchunks + 1u) + c + 1u) * p2) * a.dv + d;
        for (uint f = 0; f < p2; ++f) Gn[f] = gn[(ulong)f * a.dv];
        for (uint s = 0; s < HK_C; ++s) {
            float acc = 0.0f;
            for (uint t = s; t < HK_C; ++t) {
                acc += A[t * HK_C + s] * dout[ROW(t) * a.dv + d];
            }
            device const float* k = phk + ROW(s) * p2;
            device const float* g = pw + (HK_C - 1u - s) * p2;
            for (uint f = 0; f < p2; ++f) acc += k[f] * g[f] * Gn[f];
            dkv[ROW(s) * a.dv + d] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 3: dA (overwrites A)
    for (uint idx = tid; idx < HK_C * HK_C; idx += nth) {
        uint t = idx / HK_C, s = idx % HK_C;
        float acc = 0.0f;
        if (s <= t) {
            device const float* dq = dout + ROW(t) * a.dv;
            device const float* kvs = kv + ROW(s) * a.dv;
            for (uint d = 0; d < a.dv; ++d) acc += dq[d] * kvs[d];
        }
        A[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 4: dφq, dφk — thread per (t, f)
    for (uint idx = tid; idx < HK_C * p2; idx += nth) {
        uint t = idx / p2, f = idx % p2;
        // dφq_t[f]
        float acc = 0.0f;
        for (uint s = 0; s <= t; ++s) {
            acc += A[t * HK_C + s] * phk[ROW(s) * p2 + f] * pw[(t - s) * p2 + f];
        }
        {
            device const float* dq = dout + ROW(t) * a.dv;
            device const float* sc = states + (((ulong)bh * (nchunks + 1u) + c) * p2 + f) * a.dv;
            float dot = 0.0f;
            for (uint d = 0; d < a.dv; ++d) dot += dq[d] * sc[d];
            acc += pw[(t + 1u) * p2 + f] * dot;
        }
        dphq[ROW(t) * p2 + f] = acc;
        // dφk_s[f] with s = t (same index space)
        uint s = t;
        float acck = 0.0f;
        for (uint tt = s; tt < HK_C; ++tt) {
            acck += A[tt * HK_C + s] * phq[ROW(tt) * p2 + f] * pw[(tt - s) * p2 + f];
        }
        {
            device const float* kvs = kv + ROW(s) * a.dv;
            device const float* gn = dstates + (((ulong)bh * (nchunks + 1u) + c + 1u) * p2 + f) * a.dv;
            float dot = 0.0f;
            for (uint d = 0; d < a.dv; ++d) dot += kvs[d] * gn[d];
            acck += pw[(HK_C - 1u - s) * p2 + f] * dot;
        }
        dphk[ROW(s) * p2 + f] = acck;
    }
    #undef ROW
}

// ---------------------------------------------------------------------
// Anchor attention companions (the softmax layer: GQA, RoPE, causal).
// The matmuls are the generic GEMM with column-block offsets; only the
// row-wise softmax and RoPE are custom.
// ---------------------------------------------------------------------

// RoPE (neox halves: pair (i, i+hd/2)) in place on x [rows, nheads·hd];
// position = row % T. sign=+1 forward, −1 = inverse rotation (backward).
struct RopeArgs { uint T, nheads, hd; float base; float sign; uint pos0; };
kernel void rope_f32(
    device float*      x [[buffer(0)]],
    constant RopeArgs& a [[buffer(1)]],
    uint gid [[thread_position_in_grid]])   // over rows·nheads·(hd/2)
{
    uint half_hd = a.hd / 2u;
    uint per_row = a.nheads * half_hd;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / half_hd, i = r % half_hd;
    uint pos = row % a.T + a.pos0;
    float inv_freq = pow(a.base, -(float)(2u * i) / (float)a.hd);
    float ang = (float)pos * inv_freq;
    float c = cos(ang), s = sin(ang) * a.sign;
    device float* p = x + (ulong)row * a.nheads * a.hd + h * a.hd;
    float x0 = p[i], x1 = p[i + half_hd];
    p[i] = x0 * c - x1 * s;
    p[i + half_hd] = x0 * s + x1 * c;
}

// Band + sink softmax over the rows of a [T, LD] score block, in place
// (LD = sink_pad + T; the bounded anchor of docs/EMBRYO_BOUNDED_ANCHOR.md).
// Row `row` keeps the sink columns 0..sink, zeroes the pad columns
// sink..sink_pad, and in the causal block (column sink_pad + j) keeps
// j ≤ row with row − j < window (window = 0: the whole causal triangle).
// One threadgroup per row (256 threads); grid over T rows × blocks.
// With (ld, sink, sink_pad, window) = (T, 0, 0, 0) every thread visits
// exactly the columns it visited in the legacy causal kernel, in the same
// order, so the legacy anchor stays bit-identical.
struct SoftmaxArgs {
    uint t;         // rows per block
    uint ld;        // row length (sink_pad + carry_pad + t)
    uint sink;      // live sink columns
    uint sink_pad;  // column of the carried block (= column of the causal block when carry_pad == 0)
    uint window;    // band width in keys incl. the current one (0 = full)
    uint carry_pad; // columns of the carried-key block (keys of the previous window, positions −carry_pad..−1)
    uint bps;       // blocks per sequence (the carry mask is per sequence)
};

// Row `row` (query position carry_pad + row of the joint sequence) keeps
// the sinks, the carried keys j ∈ [carry_pad + row + 1 − window, carry_pad)
// when the sequence's carry mask is set, and the causal block as before.
// With carry_pad == 0 every loop below visits exactly the legacy columns in
// the legacy order (the carried loops are empty), so the legacy anchor and
// the bounded anchor without carry stay bit-identical.
kernel void causal_softmax_rows_f32(
    device float*         S [[buffer(0)]],
    constant SoftmaxArgs& a [[buffer(1)]],
    device const uint*    cmask [[buffer(2)]],
    uint2 tgp [[threadgroup_position_in_grid]],   // x: row, y: [T,LD] block
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    uint row = tgp.x;
    uint n = a.ld;
    device float* r = S + (ulong)tgp.y * a.t * n + (ulong)row * n;
    uint cb = a.sink_pad + a.carry_pad;   // column of the causal block
    // valid columns: [0, sink) ∪ [clo, cb) (carried, if valid) ∪ [lo, jmax)
    uint jmax = cb + row + 1u;
    uint lo = cb;
    if (a.window > 0u && row + 1u > a.window) lo = cb + row + 1u - a.window;
    uint clo = cb;   // empty carried range by default
    if (a.carry_pad > 0u && cmask[tgp.y / a.bps] != 0u) {
        uint back = (a.window > 0u) ? a.window - 1u : a.carry_pad;   // carried keys within the band
        clo = (back >= a.carry_pad + row + 1u) ? a.sink_pad : cb - (back - row);
        if (back < row + 1u) clo = cb;
    }
    float mx = -INFINITY;
    for (uint j = clo + tid; j < cb; j += 256u) mx = max(mx, r[j]);
    for (uint j = tid; j < jmax; j += 256u) { if (j < a.sink || j >= lo) mx = max(mx, r[j]); }
    mx = simd_max(mx);
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint s = 1; s < 8u; ++s) mx = max(mx, red[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = clo + tid; j < cb; j += 256u) { float e = exp(r[j] - mx); r[j] = e; sum += e; }
    for (uint j = tid; j < jmax; j += 256u) {
        if (j < a.sink || j >= lo) { float e = exp(r[j] - mx); r[j] = e; sum += e; }
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) tot += red[s];
    float inv = 1.0f / tot;
    for (uint j = clo + tid; j < cb; j += 256u) r[j] *= inv;
    for (uint j = tid; j < jmax; j += 256u) { if (j < a.sink || j >= lo) r[j] *= inv; }
    for (uint j = jmax + tid; j < n; j += 256u) r[j] = 0.0f;
    for (uint j = a.sink + tid; j < a.sink_pad; j += 256u) r[j] = 0.0f;
    for (uint j = a.sink_pad + tid; j < clo; j += 256u) r[j] = 0.0f;
    for (uint j = cb + tid; j < lo; j += 256u) r[j] = 0.0f;
}

// Softmax backward on rows: dS = P ⊙ (dP − Σ_j P·dP), in place on dP.
// Blocks are [t, ld] (row stride ld); P = 0 off the band ⇒ dS = 0 there.
kernel void softmax_bwd_rows_f32(
    device const float* P  [[buffer(0)]],
    device float*       dP [[buffer(1)]],
    constant uint4&     a  [[buffer(2)]],   // x: t (rows per block), y: ld
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    uint row = tgp.x;
    uint n = a.y;
    device const float* p = P + (ulong)tgp.y * a.x * n + (ulong)row * n;
    device float* d = dP + (ulong)tgp.y * a.x * n + (ulong)row * n;
    float dot = 0.0f;
    for (uint j = tid; j < n; j += 256u) dot += p[j] * d[j];
    dot = simd_sum(dot);
    if (lane == 0) red[sgid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) tot += red[s];
    for (uint j = tid; j < n; j += 256u) d[j] = p[j] * (d[j] - tot);
}

// Bounded-anchor sink gradient fold: per-(sequence, head) partial tiles
// src[nb][qh][pad][hd] → dst[(g·S + s)·hd + d] += alpha · Σ_b Σ_j src[(b·qh + g·group+j)·pad + s][d]
// (one thread per output element of the [kvh, S, hd] arena tensor).
struct SinkAccumArgs {
    uint n;      // kvh·sink·hd outputs
    uint sink;
    uint hd;
    uint group;
    uint pad;    // SINK_PAD rows per partial tile
    float alpha;
    uint nb;     // sequences
    uint qh;     // heads per sequence (kvh·group)
};

kernel void sink_grad_accum_f32(
    device const float*     src [[buffer(0)]],
    device float*           dst [[buffer(1)]],
    constant SinkAccumArgs& a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    uint d = gid % a.hd;
    uint s = (gid / a.hd) % a.sink;
    uint g = gid / (a.hd * a.sink);
    float acc = 0.0f;
    for (uint b = 0; b < a.nb; ++b) {
        for (uint j = 0; j < a.group; ++j) {
            acc += src[((ulong)(b * a.qh + g * a.group + j) * a.pad + s) * a.hd + d];
        }
    }
    dst[gid] += a.alpha * acc;
}

// σ(x + bias) forward (y) and backward (dx = dy·y·(1−y)) — the κ gate.
kernel void sigmoid_fwd_f32(
    device const float* x    [[buffer(0)]],
    device float*       y    [[buffer(1)]],
    constant float&     bias [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    y[gid] = 1.0f / (1.0f + exp(-(x[gid] + bias)));
}
kernel void sigmoid_bwd_f32(
    device const float* y  [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float*       dx [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float s = y[gid];
    dx[gid] = dy[gid] * s * (1.0f - s);
}

// dE[tok[row], :] += dx[row, :]  (atomic float adds; tied head)
kernel void embed_scatter_add_f32(
    device atomic_float* dE  [[buffer(0)]],
    device const uint*   tok [[buffer(1)]],
    device const float*  dx  [[buffer(2)]],
    constant uint&       d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])   // x: column, y: row
{
    if (gid.x >= d) return;
    atomic_fetch_add_explicit(&dE[(ulong)tok[gid.y] * d + gid.x], dx[(ulong)gid.y * d + gid.x], memory_order_relaxed);
}

// Strided block copy with an optional per-row mask (state carry-over):
// dst[dst_off + blk·dst_stride + i] = mask[blk / mask_div] ? src[src_off + blk·src_stride + i] : 0
// for blk < nblk, i < len (mask absent when use_mask == 0).
struct BlockCopyArgs { uint nblk, len, src_off, src_stride, dst_off, dst_stride, mask_div, use_mask; };
kernel void block_copy_f32(
    device const float* src  [[buffer(0)]],
    device float*       dst  [[buffer(1)]],
    device const uint*  mask [[buffer(2)]],
    constant BlockCopyArgs& a [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.nblk * a.len) return;
    uint blk = gid / a.len, i = gid % a.len;
    bool on = (a.use_mask == 0u) || (mask[blk / a.mask_div] != 0u);
    dst[(ulong)a.dst_off + (ulong)blk * a.dst_stride + i] =
        on ? src[(ulong)a.src_off + (ulong)blk * a.src_stride + i] : 0.0f;
}

// Copy `n` floats: dst = src (buffer-to-buffer with offsets on the host side).
kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) dst[gid] = src[gid];
}

// κ gate with a padded pre-activation (the projection GEMM writes 64
// columns; only nh are real): kap[row·nh + h] = σ(pre[row·ld + h] + bias)
struct KapArgs { uint rows, nh, ld; float bias; };
kernel void kappa_fwd_f32(
    device const float* pre [[buffer(0)]],
    device float*       kap [[buffer(1)]],
    constant KapArgs&   a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over rows·nh
{
    if (gid >= a.rows * a.nh) return;
    uint row = gid / a.nh, h = gid % a.nh;
    kap[gid] = 1.0f / (1.0f + exp(-(pre[(ulong)row * a.ld + h] + a.bias)));
}
// dpre[row·ld + j] = j < nh ? dkap[row·nh + j]·k(1−k) : 0
kernel void kappa_bwd_f32(
    device const float* kap  [[buffer(0)]],
    device const float* dkap [[buffer(1)]],
    device float*       dpre [[buffer(2)]],
    constant KapArgs&   a    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over rows·ld
{
    if (gid >= a.rows * a.ld) return;
    uint row = gid / a.ld, j = gid % a.ld;
    float g = 0.0f;
    if (j < a.nh) {
        float k = kap[(ulong)row * a.nh + j];
        g = dkap[(ulong)row * a.nh + j] * k * (1.0f - k);
    }
    dpre[gid] = g;
}

// hybrid_k chunk scan, GEMM formulation (log-space decay trick):
//   Q̃[t][f] = φq·γ^t,  K̃[s][f] = φk·γ^{−s},  Q⁺[t][f] = φq·γ^{t+1},  K̂[s][f] = φk·γ^{C−1−s}
// (t, s relative to the chunk; f32 range is ample for horizons ≥ 1 at C = 64).
// Chunk-major layout [B, nh, nchunks, 64, P2]; source phq/phk are the row-major
// [B·T, nh·P2] tables. One thread per (row, h, f).
kernel void hk_scale_f32(
    device const float* phq   [[buffer(0)]],
    device const float* phk   [[buffer(1)]],
    device const float* pow_t [[buffer(2)]],
    device float*       qt    [[buffer(3)]],
    device float*       kt    [[buffer(4)]],
    device float*       qp    [[buffer(5)]],
    device float*       kh    [[buffer(6)]],
    constant HkArgs&    a     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·P2
{
    uint p2 = 2u * a.nph;
    uint rows = a.B * a.T;
    if (gid >= rows * a.nh * p2) return;
    uint row = gid / (a.nh * p2), r = gid % (a.nh * p2);
    uint h = r / p2, f = r % p2;
    uint b = row / a.T, tt = row % a.T;
    uint c = tt / HK_C, t = tt % HK_C;
    uint nch = a.T / HK_C;
    ulong dst = (((ulong)(b * a.nh + h) * nch + c) * HK_C + t) * p2 + f;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    float q = phq[gid], k = phk[gid];
    float g_t = pw[t * p2 + f];
    qt[dst] = q * g_t;
    kt[dst] = k / g_t;
    qp[dst] = q * pw[(t + 1u) * p2 + f];
    kh[dst] = k * pw[(HK_C - 1u - t) * p2 + f];
}

// Inverse: dφq = dQ̃·γ^t + dqi·γ^{t+1};  dφk = dK̃·γ^{−s} + dki·γ^{C−1−s}
// (chunk-major inputs → row-major dphq/dphk).
kernel void hk_unscale_f32(
    device const float* dqt   [[buffer(0)]],
    device const float* dkt   [[buffer(1)]],
    device const float* dqi   [[buffer(2)]],
    device const float* dki   [[buffer(3)]],
    device const float* pow_t [[buffer(4)]],
    device float*       dphq  [[buffer(5)]],
    device float*       dphk  [[buffer(6)]],
    constant HkArgs&    a     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint p2 = 2u * a.nph;
    uint rows = a.B * a.T;
    if (gid >= rows * a.nh * p2) return;
    uint row = gid / (a.nh * p2), r = gid % (a.nh * p2);
    uint h = r / p2, f = r % p2;
    uint b = row / a.T, tt = row % a.T;
    uint c = tt / HK_C, t = tt % HK_C;
    uint nch = a.T / HK_C;
    ulong src = (((ulong)(b * a.nh + h) * nch + c) * HK_C + t) * p2 + f;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    float g_t = pw[t * p2 + f];
    dphq[gid] = dqt[src] * g_t + dqi[src] * pw[(t + 1u) * p2 + f];
    dphk[gid] = dkt[src] / g_t + dki[src] * pw[(HK_C - 1u - t) * p2 + f];
}

// Chunk-state scans, cell-parallel: the recurrence is independent per
// (f, d) cell, sequential only in t — one thread per cell, threadgroup =
// 8 features × 32 value channels (φk broadcast across d, kv coalesced
// across d). Grid: (B·nh, ceil(P2/8), ceil(dv/32)).
#define HK_FB 8u
#define HK_DB 32u

kernel void hk_states_fwd_par_f32(
    device const float* phk    [[buffer(0)]],
    device const float* kv     [[buffer(1)]],
    device const float* pow_t  [[buffer(2)]],
    device float*       states [[buffer(3)]],
    constant HkArgs&    a      [[buffer(4)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])   // x: d (32), y: f (8)
{
    uint b = tg.x / a.nh, h = tg.x % a.nh;
    uint p2 = 2u * a.nph;
    uint f = tg.y * HK_FB + tid.y;
    uint d = tg.z * HK_DB + tid.x;
    if (f >= p2 || d >= a.dv) return;
    uint nchunks = a.T / HK_C;
    float gam = pow_t[((ulong)h * (HK_C + 1u) + 1u) * p2 + f];
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    device const float* pk = phk + ((ulong)b * a.T * a.nh + h) * p2 + f;
    device const float* kvp = kv + ((ulong)b * a.T * a.nh + h) * a.dv + d;
    ulong step_k = (ulong)a.nh * p2, step_v = (ulong)a.nh * a.dv;
    float S = (a.carry != 0u) ? states[st_base + (ulong)f * a.dv + d] : 0.0f;
    states[st_base + (ulong)f * a.dv + d] = S;
    for (uint c = 0; c < nchunks; ++c) {
        // 16 positions a batch: all loads issued before the dependent FMAs
        for (uint s0 = 0; s0 < HK_C; s0 += 16u) {
            float kk[16], vv[16];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                ulong t = (ulong)c * HK_C + s0 + i;
                kk[i] = pk[t * step_k];
                vv[i] = kvp[t * step_v];
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) S = gam * S + kk[i] * vv[i];
        }
        states[st_base + ((ulong)(c + 1u) * p2 + f) * a.dv + d] = S;
    }
}

kernel void hk_dstates_bwd_par_f32(
    device const float* phq     [[buffer(0)]],
    device const float* dout    [[buffer(1)]],
    device const float* pow_t   [[buffer(2)]],
    device float*       dstates [[buffer(3)]],
    constant HkArgs&    a       [[buffer(4)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    uint b = tg.x / a.nh, h = tg.x % a.nh;
    uint p2 = 2u * a.nph;
    uint f = tg.y * HK_FB + tid.y;
    uint d = tg.z * HK_DB + tid.x;
    if (f >= p2 || d >= a.dv) return;
    uint nchunks = a.T / HK_C;
    float gam = pow_t[((ulong)h * (HK_C + 1u) + 1u) * p2 + f];
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    device const float* pq = phq + ((ulong)b * a.T * a.nh + h) * p2 + f;
    device const float* dop = dout + ((ulong)b * a.T * a.nh + h) * a.dv + d;
    ulong step_q = (ulong)a.nh * p2, step_o = (ulong)a.nh * a.dv;
    float G = 0.0f;
    dstates[st_base + ((ulong)nchunks * p2 + f) * a.dv + d] = 0.0f;
    for (uint cc = 0; cc < nchunks; ++cc) {
        uint c = nchunks - 1u - cc;
        for (uint s0 = 0; s0 < HK_C; s0 += 16u) {
            float qq[16], oo[16];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                ulong t = (ulong)c * HK_C + (HK_C - 1u - (s0 + i));
                qq[i] = pq[t * step_q];
                oo[i] = dop[t * step_o];
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) G = gam * (G + qq[i] * oo[i]);
        }
        dstates[st_base + ((ulong)c * p2 + f) * a.dv + d] = G;
    }
}

// ---------------------------------------------------------------------
// Hierarchical head companions (128 clusters × 256, tied to the embedding):
// rows are grouped by target cluster on the host; these gather/scatter the
// grouped rows and run the within-cluster CE with an index map.
// ---------------------------------------------------------------------

// dst[i,:] = idx[i] ≥ 0 ? src[idx[i],:] : 0
kernel void gather_rows_f32(
    device const float* src [[buffer(0)]],
    device const int*   idx [[buffer(1)]],
    device float*       dst [[buffer(2)]],
    constant uint&      d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= d) return;
    int i = idx[gid.y];
    dst[(ulong)gid.y * d + gid.x] = (i >= 0) ? src[(ulong)i * d + gid.x] : 0.0f;
}

// dst[idx[i],:] += src[i,:]  for idx[i] ≥ 0 (indices unique: no atomics)
kernel void scatter_add_rows_f32(
    device float*       dst [[buffer(0)]],
    device const int*   idx [[buffer(1)]],
    device const float* src [[buffer(2)]],
    constant uint&      d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= d) return;
    int i = idx[gid.y];
    if (i >= 0) dst[(ulong)i * d + gid.x] += src[(ulong)gid.y * d + gid.x];
}

// Within-cluster softmax-CE over rows of n logits with an index map:
// row r stands for token idx[r] (< 0: padding → dlogits row = 0, no loss);
// target = tgt[idx[r]] mod n; loss2[idx[r]] = −log p; logits ← (p−onehot)·scale.
kernel void softmax_ce_idx_f32(
    device float*       logits [[buffer(0)]],
    device const int*   idx    [[buffer(1)]],
    device const uint*  tgt    [[buffer(2)]],
    device float*       loss2  [[buffer(3)]],
    constant uint&      n      [[buffer(4)]],
    constant float&     scale  [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device float* lr = logits + (ulong)row * n;
    int i = idx[row];
    if (i < 0) {
        for (uint j = tid; j < n; j += 256u) lr[j] = 0.0f;
        return;
    }
    if (tgt[i] == 0xFFFFFFFFu) {
        for (uint j = tid; j < n; j += 256u) lr[j] = 0.0f;
        if (tid == 0) loss2[i] = 0.0f;
        return;
    }
    uint t = tgt[i] % n;
    float mx = -INFINITY;
    for (uint j = tid; j < n; j += 256u) mx = max(mx, lr[j]);
    mx = simd_max(mx);
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint s = 1; s < 8u; ++s) mx = max(mx, red[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < n; j += 256u) sum += exp(lr[j] - mx);
    sum = simd_sum(sum);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) tot += red[s];
    float lse = mx + log(tot);
    if (tid == 0) loss2[i] = lse - lr[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < n; j += 256u) {
        float p = exp(lr[j] - lse);
        lr[j] = (p - ((j == t) ? 1.0f : 0.0f)) * scale;
    }
}

// GQA group reduction: src is head-major [B][qh][T][hd] (per-q-head partial
// dK or dV), dst is row-major [B·T, kvh·hd]; dst[b,t][g·hd + d] = Σ_j src[b][g·group + j][t][d].
struct GroupSumArgs { uint B, T, qh, kvh, hd; };
kernel void group_sum_heads_f32(
    device const float*    src [[buffer(0)]],
    device float*          dst [[buffer(1)]],
    constant GroupSumArgs& a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over B·T·kvh·hd
{
    uint kd = a.kvh * a.hd;
    uint rows = a.B * a.T;
    if (gid >= rows * kd) return;
    uint row = gid / kd, r = gid % kd;
    uint g = r / a.hd, d = r % a.hd;
    uint b = row / a.T, t = row % a.T;
    uint group = a.qh / a.kvh;
    float s = 0.0f;
    for (uint j = 0; j < group; ++j) {
        uint i = g * group + j;
        s += src[(((ulong)b * a.qh + i) * a.T + t) * a.hd + d];
    }
    dst[gid] = s;
}

// ---------------------------------------------------------------------
// Routed experts (top-1 by RESONANCE, no gate — P1): expert e has a
// descriptor μ_e (+ later a k-dim principal subspace U_e); resonance =
// reconstruction error ‖(x−μ_e) − U_eᵀU_e(x−μ_e)‖²; route to argmin of
// (resonance − bias_e), bias_e = loss-free load balancing.
// Tokens are placed in per-expert slots deterministically (rank among the
// tokens of that expert); slots ≥ cap are dropped (shared expert only).
// ---------------------------------------------------------------------

struct RouteArgs { uint rows, H, E, k, cap; };

// assign[row] = argmin_e (‖x−μ_e‖² − ‖U_e(x−μ_e)‖² − bias_e); one threadgroup
// (E ≤ 64 threads... use 64 threads: thread e computes expert e's score)
kernel void route_f32(
    device const float* x      [[buffer(0)]],   // [rows, H]
    device const float* mu     [[buffer(1)]],   // [E, H]
    device const float* U      [[buffer(2)]],   // [E, k, H] (rows orthonormal; k may be 0)
    device const float* bias   [[buffer(3)]],   // [E]
    device uint*        assign [[buffer(4)]],   // [rows]
    device float*       res    [[buffer(5)]],   // [rows] resonance of the chosen expert
    constant RouteArgs& a      [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint e   [[thread_index_in_threadgroup]])
{
    threadgroup float sc[64];
    threadgroup float rs[64];
    float score = INFINITY;
    float r = 0.0f;
    if (e < a.E) {
        device const float* xr = x + (ulong)row * a.H;
        device const float* m = mu + (ulong)e * a.H;
        float d2 = 0.0f;
        for (uint j = 0; j < a.H; ++j) { float d = xr[j] - m[j]; d2 += d * d; }
        float proj = 0.0f;
        for (uint i = 0; i < a.k; ++i) {
            device const float* u = U + ((ulong)e * a.k + i) * a.H;
            float p = 0.0f;
            for (uint j = 0; j < a.H; ++j) p += (xr[j] - m[j]) * u[j];
            proj += p * p;
        }
        r = d2 - proj;
        score = r - bias[e];
    }
    sc[e] = score;
    rs[e] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        uint best = 0;
        float bs = sc[0];
        for (uint i = 1; i < a.E; ++i) { if (sc[i] < bs) { bs = sc[i]; best = i; } }
        assign[row] = best;
        res[row] = rs[best];
    }
}

// Causal-k4 variant of the reconstruction router.  Each expert thread
// recomputes the raw reconstruction cost for the current row and at most the
// preceding three positions in the same sequence.  The smoothed cost is used
// only for argmin selection; `res` receives the current-row raw cost of the
// winning expert so diagnostics and descriptor balancing retain legacy
// semantics.  No history buffer or persistent state is required.
struct RouteSmoothK4Args { uint rows, H, E, k, cap, seq; };

inline float route_reconstruction_f32(
    device const float* x,
    device const float* mu,
    device const float* U,
    uint row,
    uint e,
    uint H,
    uint k)
{
    device const float* xr = x + (ulong)row * H;
    device const float* m = mu + (ulong)e * H;
    float d2 = 0.0f;
    for (uint j = 0; j < H; ++j) {
        float d = xr[j] - m[j];
        d2 += d * d;
    }
    float proj = 0.0f;
    for (uint i = 0; i < k; ++i) {
        device const float* u = U + ((ulong)e * k + i) * H;
        float p = 0.0f;
        for (uint j = 0; j < H; ++j) p += (xr[j] - m[j]) * u[j];
        proj += p * p;
    }
    return d2 - proj;
}

kernel void route_smooth_k4_f32(
    device const float* x      [[buffer(0)]],   // [rows, H]
    device const float* mu     [[buffer(1)]],   // [E, H]
    device const float* U      [[buffer(2)]],   // [E, k, H]
    device const float* bias   [[buffer(3)]],   // [E]
    device uint*        assign [[buffer(4)]],   // [rows]
    device float*       res    [[buffer(5)]],   // raw winning resonance
    constant RouteSmoothK4Args& a [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint e   [[thread_index_in_threadgroup]])
{
    threadgroup float sc[64];
    threadgroup float rs[64];
    float raw = 0.0f;
    float score = INFINITY;
    if (e < a.E) {
        uint pos = row % a.seq;
        uint first = pos > 3u ? pos - 3u : 0u;
        uint n = pos - first + 1u;
        float sum = 0.0f;
        for (uint p = first; p <= pos; ++p) {
            uint prow = row - (pos - p);
            float rp = route_reconstruction_f32(x, mu, U, prow, e, a.H, a.k);
            sum += rp;
            if (p == pos) raw = rp;
        }
        score = sum / (float)n - bias[e];
    }
    sc[e] = score;
    rs[e] = raw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u) {
        uint best = 0u;
        float bs = sc[0];
        for (uint i = 1u; i < a.E; ++i) {
            if (sc[i] < bs) { bs = sc[i]; best = i; }
        }
        assign[row] = best;
        res[row] = rs[best];
    }
}

// Conditional ambiguity fallback: compute the same custom reconstruction
// scores as the top-1 router, retain a deterministic runner-up, and mark it
// only when the adjusted score margin is below a fixed host-supplied
// threshold.  `seq == 0` uses raw position-local scores; otherwise the score
// is the causal-k4 mean (current plus up to three preceding rows).  The
// winning raw current-row resonance remains available for diagnostics.  This
// kernel is a parity/telemetry primitive; the model's routed-expert backward
// still needs a second bounded activation stream before it can dispatch it.
struct RouteTop2Args { uint rows, H, E, k, cap, seq; float threshold; };

kernel void route_top2_f32(
    device const float* x       [[buffer(0)]],   // [rows, H]
    device const float* mu      [[buffer(1)]],   // [E, H]
    device const float* U       [[buffer(2)]],   // [E, k, H]
    device const float* bias    [[buffer(3)]],   // [E]
    device uint*        assign  [[buffer(4)]],   // top-1 expert
    device uint*        runner  [[buffer(5)]],   // runner-up or UINT_MAX
    device float*       margin  [[buffer(6)]],   // score2 - score1
    device float*       weight  [[buffer(7)]],   // fixed 0.5 when active
    device float*       res     [[buffer(8)]],   // raw winning resonance
    device atomic_uint* fallback_count [[buffer(9)]],
    constant RouteTop2Args& a    [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint e   [[thread_index_in_threadgroup]])
{
    threadgroup float sc[64];
    threadgroup float rs[64];
    float raw = 0.0f;
    float score = INFINITY;
    if (e < a.E) {
        uint pos = (a.seq > 0u) ? (row % a.seq) : 0u;
        uint first = (a.seq > 0u) ? ((pos > 3u) ? (pos - 3u) : 0u) : 0u;
        uint n = (a.seq > 0u) ? (pos - first + 1u) : 1u;
        float sum = 0.0f;
        for (uint p = first; p <= pos; ++p) {
            uint prow = (a.seq > 0u) ? (row - (pos - p)) : row;
            float rp = route_reconstruction_f32(x, mu, U, prow, e, a.H, a.k);
            sum += rp;
            if (p == pos) raw = rp;
        }
        score = sum / (float)n - bias[e];
    }
    sc[e] = score;
    rs[e] = raw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u) {
        uint best = 0u;
        uint second = UINT_MAX;
        float bs = sc[0];
        float ss = INFINITY;
        for (uint i = 1u; i < a.E; ++i) {
            // Strict `<` preserves the lowest-index expert on ties.
            if (sc[i] < bs) {
                second = best;
                ss = bs;
                best = i;
                bs = sc[i];
            } else if (sc[i] < ss) {
                second = i;
                ss = sc[i];
            }
        }
        assign[row] = best;
        res[row] = rs[best];
        runner[row] = UINT_MAX;
        weight[row] = 0.0f;
        float m = (second == UINT_MAX) ? INFINITY : (ss - bs);
        margin[row] = m;
        if (second != UINT_MAX && a.threshold > 0.0f && m < a.threshold) {
            runner[row] = second;
            weight[row] = 0.5f;
            atomic_fetch_add_explicit(fallback_count, 1u, memory_order_relaxed);
        }
    }
}

// Deterministic slot assignment: slot[row] = rank of row among rows with
// the same expert (serial pass, one thread); count[e] = tokens per expert.
kernel void route_group_f32(
    device const uint* assign [[buffer(0)]],
    device uint*       slot   [[buffer(1)]],
    device uint*       count  [[buffer(2)]],   // [E]
    constant RouteArgs& a     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    uint c[64];
    for (uint e = 0; e < a.E; ++e) c[e] = 0;
    for (uint r = 0; r < a.rows; ++r) {
        uint e = assign[r];
        if (e >= a.E) { slot[r] = UINT_MAX; continue; }
        slot[r] = c[e];
        c[e] += 1u;
    }
    for (uint e = 0; e < a.E; ++e) count[e] = c[e];
}

// hg[e][slot][:] = x[row][:] for slot < cap (buffer pre-zeroed)
kernel void moe_gather_f32(
    device const float* x      [[buffer(0)]],
    device const uint*  assign [[buffer(1)]],
    device const uint*  slot   [[buffer(2)]],
    device float*       hg     [[buffer(3)]],   // [E, cap, H]
    constant RouteArgs& a      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])   // x: column, y: row
{
    if (gid.x >= a.H) return;
    uint s = slot[gid.y];
    if (s >= a.cap) return;
    uint e = assign[gid.y];
    hg[((ulong)e * a.cap + s) * a.H + gid.x] = x[(ulong)gid.y * a.H + gid.x];
}

// Weighted gather for conditional top-2 activation/gradient streams.  When
// complement != 0 the row factor is (1 - weight[row]) for the primary stream;
// otherwise it is weight[row] for the runner stream. Invalid runner rows use
// UINT_MAX and are ignored by the slot/capacity checks.
struct RouteWeightArgs { uint rows, H, E, k, cap, complement; };
kernel void moe_gather_weighted_f32(
    device const float* x      [[buffer(0)]],
    device const uint* assign  [[buffer(1)]],
    device const uint* slot    [[buffer(2)]],
    device const float* weight  [[buffer(3)]],
    device float*       hg      [[buffer(4)]],
    constant RouteWeightArgs& a [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.H || gid.y >= a.rows) return;
    uint s = slot[gid.y];
    if (s >= a.cap) return;
    uint e = assign[gid.y];
    if (e >= a.E) return;
    float w = weight[gid.y];
    if (a.complement != 0u) w = 1.0f - w;
    hg[((ulong)e * a.cap + s) * a.H + gid.x] =
        w * x[(ulong)gid.y * a.H + gid.x];
}

// out[row][:] += yh[e][slot][:] for slot < cap
kernel void moe_scatter_add_f32(
    device float*       out    [[buffer(0)]],
    device const uint*  assign [[buffer(1)]],
    device const uint*  slot   [[buffer(2)]],
    device const float* yh     [[buffer(3)]],
    constant RouteArgs& a      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.H) return;
    uint s = slot[gid.y];
    if (s >= a.cap) return;
    uint e = assign[gid.y];
    out[(ulong)gid.y * a.H + gid.x] += yh[((ulong)e * a.cap + s) * a.H + gid.x];
}

// Weighted routed-expert scatter for conditional top-2 residual blending.
kernel void moe_scatter_add_weighted_f32(
    device float*       out    [[buffer(0)]],
    device const uint*  assign [[buffer(1)]],
    device const uint*  slot   [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* yh     [[buffer(4)]],
    constant RouteWeightArgs& a [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.H || gid.y >= a.rows) return;
    uint s = slot[gid.y];
    if (s >= a.cap) return;
    uint e = assign[gid.y];
    if (e >= a.E) return;
    float w = weight[gid.y];
    if (a.complement != 0u) w = 1.0f - w;
    out[(ulong)gid.y * a.H + gid.x] +=
        w * yh[((ulong)e * a.cap + s) * a.H + gid.x];
}

// Descriptor statistics: sums[e][j] = Σ_{slots of e} hg[e][slot][j] (over
// the filled slots — zero rows contribute nothing); one thread per (e, j).
kernel void moe_stats_f32(
    device const float* hg    [[buffer(0)]],
    device const uint*  count [[buffer(1)]],
    device float*       sums  [[buffer(2)]],   // [E, H]
    constant RouteArgs& a     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over E·H
{
    if (gid >= a.E * a.H) return;
    uint e = gid / a.H, j = gid % a.H;
    uint n = min(count[e], a.cap);
    float s = 0.0f;
    for (uint r = 0; r < n; ++r) s += hg[((ulong)e * a.cap + r) * a.H + j];
    sums[gid] = s;
}

// Descriptor update after a step: μ_e ← (1−α)·μ_e + α·sums_e/n_e with
// n_e = min(count_e, cap) — exactly the slots moe_stats_f32 summed (a
// token past the capacity has no slot; dividing the capped sum by the full
// count pulled the μ of an over-capacity expert toward the origin by
// cap/count every step); bias_e += η·(1/E − count_e/rows) with the FULL
// count (loss-free balancing toward equal load). One thread per (e, j);
// thread j == 0 also moves the bias — only for e < bias_frozen_from
// (growth: a grown expert never buys tokens with a balancing bias; its
// record writes bias 0).
struct MoeUpdArgs { uint rows, H, E; float alpha, eta; uint frozen_below; uint bias_frozen_from; uint cap; };
kernel void moe_update_f32(
    device float*       mu    [[buffer(0)]],
    device float*       bias  [[buffer(1)]],
    device const float* sums  [[buffer(2)]],
    device const uint*  count [[buffer(3)]],
    device const float* res   [[buffer(4)]],   // [rows] winning resonances (sets the bias scale)
    constant MoeUpdArgs& a    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.E * a.H) return;
    uint e = gid / a.H, j = gid % a.H;
    if (e < a.frozen_below) return;   // old records: descriptors never move
    uint n = count[e];
    uint n_mu = min(n, a.cap);
    if (n_mu > 0) {
        float mean = sums[gid] / (float)n_mu;
        mu[gid] = (1.0f - a.alpha) * mu[gid] + a.alpha * mean;
    }
    if (j == 0 && e < a.bias_frozen_from) {
        // bias step relative to the typical resonance, so balancing acts at
        // the scale the scores live on
        float rs = 0.0f;
        for (uint r = 0; r < a.rows; ++r) rs += res[r];
        float scale = rs / (float)a.rows;
        float frac = (float)n / (float)a.rows;
        bias[e] += a.eta * scale * (1.0f / (float)a.E - frac);
    }
}

// Indirect dispatch arguments for the per-expert GEMMs: args[e] =
// {ntiles_n, ceil(min(count_e, cap)/64), 1} for two column counts (n1, n2).
struct IndArgs { uint E, cap, n1, n2; };
kernel void moe_indirect_args_f32(
    device const uint* count [[buffer(0)]],
    device uint*       args  [[buffer(1)]],   // [2, E, 3]
    constant IndArgs&  a     [[buffer(2)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= a.E) return;
    uint m = min(count[e], a.cap);
    uint mt = (m + 63u) / 64u;
    args[(0u * a.E + e) * 3u + 0u] = a.n1 / 64u;
    args[(0u * a.E + e) * 3u + 1u] = mt;
    args[(0u * a.E + e) * 3u + 2u] = 1u;
    args[(1u * a.E + e) * 3u + 0u] = a.n2 / 64u;
    args[(1u * a.E + e) * 3u + 1u] = mt;
    args[(1u * a.E + e) * 3u + 2u] = 1u;
}

// μ_e := x[row_e] — data init of the descriptors (k-means++-style seeding
// from the batch itself; rows chosen on the host, e ≤ 64).
kernel void moe_init_mu_f32(
    device const float* x    [[buffer(0)]],
    device const uint*  rows [[buffer(1)]],   // [E]
    device float*       mu   [[buffer(2)]],   // [E, H]
    constant RouteArgs& a    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over E·H
{
    if (gid >= a.E * a.H) return;
    uint e = gid / a.H, j = gid % a.H;
    mu[gid] = x[(ulong)rows[e] * a.H + j];
}

// hgc[e][slot] = hg[e][slot] − μ_e for slot < count_e (0 beyond) — centred
// rows for the descriptor covariance.
kernel void moe_center_f32(
    device const float* hg    [[buffer(0)]],
    device const float* mu    [[buffer(1)]],   // [E, H]
    device const uint*  count [[buffer(2)]],   // [E]
    device float*       hgc   [[buffer(3)]],
    constant RouteArgs& a     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])   // over E·cap·H
{
    if (gid >= a.E * a.cap * a.H) return;
    uint j = gid % a.H;
    uint es = gid / a.H;
    uint e = es / a.cap, s = es % a.cap;
    hgc[gid] = (s < min(count[e], a.cap)) ? hg[gid] - mu[e * a.H + j] : 0.0f;
}

// ---------------------------------------------------------------------
// Skill masks (DTG-MA, P2): a logit per FFN neuron; the down_proj input
// is multiplied by σ(m) (soft) or by 1[σ(m) > τ] (hard).
// ---------------------------------------------------------------------
struct MaskArgs { uint rows, n; uint hard; float tau; float l1; };

// hh[r][j] *= mask_j
kernel void mask_fwd_f32(
    device float*       hh   [[buffer(0)]],
    device const float* m    [[buffer(1)]],
    constant MaskArgs&  a    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.n) return;
    uint j = gid % a.n;
    float s = 1.0f / (1.0f + exp(-m[j]));
    float w = a.hard ? ((s > a.tau) ? 1.0f : 0.0f) : s;
    hh[gid] *= w;
}

// dm[j] += Σ_r dhhm[r][j]·hh_pre[r][j]·σ'(m_j) + l1·σ'(m_j)   (soft only)
// one thread per column
kernel void mask_bwd_dm_f32(
    device const float* dhhm   [[buffer(0)]],
    device const float* hh_pre [[buffer(1)]],
    device const float* m      [[buffer(2)]],
    device float*       dm     [[buffer(3)]],
    constant MaskArgs&  a      [[buffer(4)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= a.n) return;
    if (a.hard) return;
    float s = 1.0f / (1.0f + exp(-m[j]));
    float ds = s * (1.0f - s);
    float acc = 0.0f;
    for (uint r = 0; r < a.rows; ++r) acc += dhhm[(ulong)r * a.n + j] * hh_pre[(ulong)r * a.n + j];
    dm[j] += (acc + a.l1) * ds;
}

// dhhm[r][j] *= mask_j  (→ the gradient w.r.t. the pre-mask activation)
kernel void mask_bwd_dh_f32(
    device float*       dhhm [[buffer(0)]],
    device const float* m    [[buffer(1)]],
    constant MaskArgs&  a    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.rows * a.n) return;
    uint j = gid % a.n;
    float s = 1.0f / (1.0f + exp(-m[j]));
    float w = a.hard ? ((s > a.tau) ? 1.0f : 0.0f) : s;
    dhhm[gid] *= w;
}

// ---------------------------------------------------------------------
// Learnable decay: γ_{h,f} = exp(−exp(A_log)). pow table from A_log and the
// gradient of the loss w.r.t. γ from the chunk-GEMM by-products.
// ---------------------------------------------------------------------

// pow[h][δ][f] = γ^δ, δ = 0..C, γ = exp(−exp(alog[h·P2+f]))
kernel void hk_pow_from_alog_f32(
    device const float* alog [[buffer(0)]],
    device float*       pow_t [[buffer(1)]],
    constant HkArgs&    a    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over nh·P2
{
    uint p2 = 2u * a.nph;
    if (gid >= a.nh * p2) return;
    uint h = gid / p2, f = gid % p2;
    float g = exp(-exp(alog[gid]));
    float acc = 1.0f;
    for (uint d = 0; d <= HK_C; ++d) {
        pow_t[((ulong)h * (HK_C + 1u) + d) * p2 + f] = acc;
        acc *= g;
    }
}

// K̃′[s][f] = φk_s[f]·(−s)·γ^{−s−1}  (chunk-major, like hk_scale's outputs)
kernel void hk_scale_ktp_f32(
    device const float* phk   [[buffer(0)]],
    device const float* pow_t [[buffer(1)]],
    device float*       ktp   [[buffer(2)]],
    constant HkArgs&    a     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·P2
{
    uint p2 = 2u * a.nph;
    uint rows = a.B * a.T;
    if (gid >= rows * a.nh * p2) return;
    uint row = gid / (a.nh * p2), r = gid % (a.nh * p2);
    uint h = r / p2, f = r % p2;
    uint b = row / a.T, tt = row % a.T;
    uint c = tt / HK_C, s = tt % HK_C;
    uint nch = a.T / HK_C;
    ulong dst = (((ulong)(b * a.nh + h) * nch + c) * HK_C + s) * p2 + f;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    float g = pw[1u * p2 + f];                    // γ
    float inv_s1 = 1.0f / (pw[s * p2 + f] * g);   // γ^{−s−1}
    ktp[dst] = phk[gid] * (-(float)s) * inv_s1;
}

// dA_log[h·P2+f] += (dγ_f)·γ·ln γ, with
//   dγ_f = Σ_{b,c,t} [ φq_t·t·γ^{t−1}·dqt + Q̃·dqtp + (t+1)/γ·Q⁺·dqi + (C−1−t)/γ·K̂·dki ](t,f)
//        + Σ_{b,c,d} C·γ^{C−1}·dstates[c+1][f][d]·states[c][f][d]
// One threadgroup (256 threads) per (h, f).
kernel void hk_dgamma_f32(
    device const float* phq     [[buffer(0)]],
    device const float* pow_t   [[buffer(1)]],
    device const float* qt      [[buffer(2)]],
    device const float* qp      [[buffer(3)]],
    device const float* kh      [[buffer(4)]],
    device const float* dqt     [[buffer(5)]],
    device const float* dqtp    [[buffer(6)]],
    device const float* dqi     [[buffer(7)]],
    device const float* dki     [[buffer(8)]],
    device const float* states  [[buffer(9)]],
    device const float* dstates [[buffer(10)]],
    device float*       dalog   [[buffer(11)]],
    constant HkArgs&    a       [[buffer(12)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    uint p2 = 2u * a.nph;
    uint h = tg / p2, f = tg % p2;
    uint nch = a.T / HK_C;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    float g = pw[1u * p2 + f];
    float inv_g = 1.0f / g;
    float acc = 0.0f;
    // chunk-major (b, c, t) items
    uint n_items = a.B * nch * HK_C;
    for (uint it = tid; it < n_items; it += 256u) {
        uint t = it % HK_C;
        uint bc = it / HK_C;               // b·nch + c
        uint b = bc / nch, c = bc % nch;
        ulong cm = (((ulong)(b * a.nh + h) * nch + c) * HK_C + t) * p2 + f;
        ulong rm = ((ulong)(b * a.T + c * HK_C + t) * a.nh + h) * p2 + f;   // row-major phq
        float qprime = (t > 0) ? phq[rm] * (float)t * pw[(t - 1u) * p2 + f] : 0.0f;
        acc += qprime * dqt[cm];
        acc += qt[cm] * dqtp[cm];
        acc += ((float)(t + 1u)) * inv_g * qp[cm] * dqi[cm];
        acc += ((float)(HK_C - 1u - t)) * inv_g * kh[cm] * dki[cm];
    }
    // state path: C·γ^{C−1}·Σ_{b,c,d} G_{c+1}·S_c
    float cg = (float)HK_C * pw[(HK_C - 1u) * p2 + f];
    uint n_state = a.B * nch * a.dv;
    float sacc = 0.0f;
    for (uint it = tid; it < n_state; it += 256u) {
        uint d = it % a.dv;
        uint bc = it / a.dv;
        uint b = bc / nch, c = bc % nch;
        ulong base = (((ulong)b * a.nh + h) * (nch + 1u)) * p2 * a.dv;
        float sc = states[base + ((ulong)c * p2 + f) * a.dv + d];
        float gn = dstates[base + ((ulong)(c + 1u) * p2 + f) * a.dv + d];
        sacc += gn * sc;
    }
    acc += cg * sacc;
    acc = simd_sum(acc);
    if (lane == 0) red[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (uint s = 0; s < 8u; ++s) tot += red[s];
        // dA = dγ · dγ/dA,  γ = exp(−e^A) → dγ/dA = γ·ln γ
        dalog[h * p2 + f] += tot * g * log(g);
    }
}

// Causal depthwise conv1d over [b, t, h], k taps, zero left pad, per-channel
// weights w[c*k + j] (tap j reaches x[t - (k-1-j)]). Identity init = last
// tap 1: the layer starts as a pass-through and LEARNS to mix neighbours.
// `hist` [b, k−1, h] (used when has_hist): the last k−1 rows of the previous
// window per sequence (state carried across windows); row k−1+src for src < 0.
kernel void conv1d_fwd_f32(
    device const float* x  [[buffer(0)]],
    device const float* w  [[buffer(1)]],
    device float*       y  [[buffer(2)]],
    constant uint4&     p  [[buffer(3)]], // b, t, h, k
    device const float* hist [[buffer(4)]],
    constant uint&      has_hist [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint bth = p.x * p.y * p.z;
    if (gid >= bth) return;
    uint c  = gid % p.z;
    uint ti = (gid / p.z) % p.y;
    uint bi = gid / (p.z * p.y);
    float acc = 0.0f;
    for (uint j = 0; j < p.w; ++j) {
        int src = int(ti) - int(p.w - 1 - j);
        if (src >= 0) acc += w[c * p.w + j] * x[((ulong)bi * p.y + (ulong)src) * p.z + c];
        else if (has_hist != 0u) acc += w[c * p.w + j] * hist[((ulong)bi * (p.w - 1u) + (ulong)(int(p.w - 1u) + src)) * p.z + c];
    }
    y[gid] = acc;
}

// dx[t] = sum_j w[c,j] * dy[t + (k-1-j)] — the future taps that read x[t],
// clipped at the sequence end (each sequence pads independently).
kernel void conv1d_bwd_dx_f32(
    device const float* dy [[buffer(0)]],
    device const float* w  [[buffer(1)]],
    device float*       dx [[buffer(2)]],
    constant uint4&     p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint bth = p.x * p.y * p.z;
    if (gid >= bth) return;
    uint c  = gid % p.z;
    uint ti = (gid / p.z) % p.y;
    uint bi = gid / (p.z * p.y);
    float acc = 0.0f;
    for (uint j = 0; j < p.w; ++j) {
        uint dst = ti + (p.w - 1 - j);
        if (dst < p.y) acc += w[c * p.w + j] * dy[((ulong)bi * p.y + dst) * p.z + c];
    }
    dx[gid] = acc;
}

// dW[c,j] += sum_{b,t} dy[b,t,c] * x[b, t-(k-1-j), c] — one thread per
// (c, j), the rmsnorm_dw idiom: serial over rows, += into the grad slot.
kernel void conv1d_dw_f32(
    device const float* x  [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float*       dw [[buffer(2)]],
    constant uint4&     p  [[buffer(3)]],
    device const float* hist [[buffer(4)]],
    constant uint&      has_hist [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint hk = p.z * p.w;
    if (gid >= hk) return;
    uint c = gid / p.w;
    uint j = gid % p.w;
    uint back = p.w - 1 - j;
    float s = 0.0f;
    for (uint bi = 0; bi < p.x; ++bi) {
        for (uint ti = back; ti < p.y; ++ti) {
            s += dy[((ulong)bi * p.y + ti) * p.z + c]
               * x[((ulong)bi * p.y + (ti - back)) * p.z + c];
        }
        // carried history rows (the first `back` outputs read the previous window)
        if (has_hist != 0u) {
            for (uint ti = 0; ti < back && ti < p.y; ++ti) {
                s += dy[((ulong)bi * p.y + ti) * p.z + c]
                   * hist[((ulong)bi * (p.w - 1u) + (ulong)(p.w - 1u + ti - back)) * p.z + c];
            }
        }
    }
    dw[gid] += s;
}
