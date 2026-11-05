#include <metal_stdlib>
using namespace metal;

// nibble k of the low 16 bits times 16^k
inline float4 nib4s(uint u) {
    return float4((float)(u & 0xFu), (float)(u & 0xF0u), (float)(u & 0xF00u), (float)(u & 0xF000u));
}

// NB activation vectors against one weight stream. R rows a simdgroup, one rung a lane; a group of 32
// columns goes in steps of C: the step's activations sit in registers across the R rows, each row's C
// nibbles are unpacked once and multiplied into every vector.
template <int NB, int R, int C>
inline void mb(device const uchar* q, device const float* x, device float* y, uint gpr, uint rows,
               uint sg, uint lane, uint tgpos, uint sgs, threadgroup float* lad)
{
    uint r0 = (tgpos * sgs + sg) * (uint)R;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, (uint)R) : 0u;
    uint cols = gpr * 32u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * (uint)R + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float acc[NB][R];
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) acc[e][ri] = 0.0f;
    }
    for (uint g = lane; g < gpr; g += 32u) {
        uint bit = g * 5u; uint cb = bit >> 3u; uint shf = bit & 7u;
        float scale[R];
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            uint r = min(r0 + (uint)ri, rows - 1u);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            scale[ri] = lad[(sg * (uint)R + min((uint)ri, nr - 1u)) * 32u + code];
        }
        #pragma clang loop unroll(full)
        for (int s = 0; s < 32 / C; ++s) {
            float4 xf[NB][C / 4];
            float m8[NB];
            #pragma clang loop unroll(full)
            for (int e = 0; e < NB; ++e) {
                device const float4* xv = (device const float4*)(x + (ulong)e * cols + g * 32u + (uint)(s * C));
                float4 s4 = float4(0.0f);
                #pragma clang loop unroll(full)
                for (int k = 0; k < C / 4; ++k) {
                    float4 v = xv[k];
                    s4 += v;
                    xf[e][k] = v;
                }
                m8[e] = -8.0f * (s4.x + s4.y + s4.z + s4.w);
            }
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) {
                uint r = min(r0 + (uint)ri, rows - 1u);
                device const uint* p = (device const uint*)(q + ((ulong)r * gpr + (ulong)g) * 16ul) + s * (C / 8);
                float4 n[C / 4];
                #pragma clang loop unroll(full)
                for (int j = 0; j < C / 8; ++j) {
                    uint u = p[j];
                    n[2 * j] = nib4s(u);
                    n[2 * j + 1] = nib4s(u >> 16);
                }
                #pragma clang loop unroll(full)
                for (int e = 0; e < NB; ++e) {
                    float4 v = n[0] * xf[e][0];
                    #pragma clang loop unroll(full)
                    for (int k = 1; k < C / 4; ++k) v += n[k] * xf[e][k];
                    acc[e][ri] += scale[ri] * (dot(v, sc) + m8[e]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            float v = simd_sum(acc[e][ri]);
            if (lane == 0u && (uint)ri < nr) y[(ulong)e * rows + r0 + (uint)ri] = v;
        }
    }
}



template <int NB, int R, int C>
inline void ms(device const uchar* q, device const float* x, device float* y, uint gpr, uint rows,
               uint sg, uint lane, uint tgpos, uint sgs, threadgroup float* lad)
{
    uint r0 = (tgpos * sgs + sg) * (uint)R;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, (uint)R) : 0u;
    uint cols = gpr * 32u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * (uint)R + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float acc[NB][R];
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) acc[e][ri] = 0.0f;
    }
    for (uint g = lane; g < gpr; g += 32u) {
        uint bit = g * 5u; uint cb = bit >> 3u; uint shf = bit & 7u;
        float scale[R];
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            uint r = min(r0 + (uint)ri, rows - 1u);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            scale[ri] = lad[(sg * (uint)R + min((uint)ri, nr - 1u)) * 32u + code];
        }
        #pragma clang loop unroll(disable)
        for (uint s = 0u; s < 32u / (uint)C; ++s) {
            float4 xf[NB][C / 4];
            float m8[NB];
            #pragma clang loop unroll(full)
            for (int e = 0; e < NB; ++e) {
                device const float4* xv = (device const float4*)(x + (ulong)e * cols + g * 32u + s * (uint)C);
                float4 s4 = float4(0.0f);
                #pragma clang loop unroll(full)
                for (int k = 0; k < C / 4; ++k) {
                    float4 v = xv[k];
                    s4 += v;
                    xf[e][k] = v;
                }
                m8[e] = -8.0f * (s4.x + s4.y + s4.z + s4.w);
            }
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) {
                uint r = min(r0 + (uint)ri, rows - 1u);
                device const uint* p = (device const uint*)(q + ((ulong)r * gpr + (ulong)g) * 16ul) + s * (uint)(C / 8);
                float4 n[C / 4];
                #pragma clang loop unroll(full)
                for (int j = 0; j < C / 8; ++j) {
                    uint u = p[j];
                    n[2 * j] = nib4s(u);
                    n[2 * j + 1] = nib4s(u >> 16);
                }
                #pragma clang loop unroll(full)
                for (int e = 0; e < NB; ++e) {
                    float4 v = n[0] * xf[e][0];
                    #pragma clang loop unroll(full)
                    for (int k = 1; k < C / 4; ++k) v += n[k] * xf[e][k];
                    acc[e][ri] += scale[ri] * (dot(v, sc) + m8[e]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            float v = simd_sum(acc[e][ri]);
            if (lane == 0u && (uint)ri < nr) y[(ulong)e * rows + r0 + (uint)ri] = v;
        }
    }
}




// `ms` with the 16^-k weights folded into the staged activations and the -8 offset taken once a group:
// a row's step is two dot products and one multiply-add a vector.
template <int NB, int R, int C>
inline void mt(device const uchar* q, device const float* x, device float* y, uint gpr, uint rows,
               uint sg, uint lane, uint tgpos, uint sgs, threadgroup float* lad)
{
    uint r0 = (tgpos * sgs + sg) * (uint)R;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, (uint)R) : 0u;
    uint cols = gpr * 32u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * (uint)R + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float acc[NB][R];
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) acc[e][ri] = 0.0f;
    }
    for (uint g = lane; g < gpr; g += 32u) {
        uint bit = g * 5u; uint cb = bit >> 3u; uint shf = bit & 7u;
        float scale[R];
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            uint r = min(r0 + (uint)ri, rows - 1u);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            scale[ri] = lad[(sg * (uint)R + min((uint)ri, nr - 1u)) * 32u + code];
        }
        float sx[NB];
        #pragma clang loop unroll(full)
        for (int e = 0; e < NB; ++e) sx[e] = 0.0f;
        #pragma clang loop unroll(disable)
        for (uint s = 0u; s < 32u / (uint)C; ++s) {
            float4 xf[NB][C / 4];
            #pragma clang loop unroll(full)
            for (int e = 0; e < NB; ++e) {
                device const float4* xv = (device const float4*)(x + (ulong)e * cols + g * 32u + s * (uint)C);
                #pragma clang loop unroll(full)
                for (int k = 0; k < C / 4; ++k) {
                    float4 v = xv[k];
                    sx[e] += (v.x + v.y) + (v.z + v.w);
                    xf[e][k] = v * sc;
                }
            }
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) {
                uint r = min(r0 + (uint)ri, rows - 1u);
                device const uint* p = (device const uint*)(q + ((ulong)r * gpr + (ulong)g) * 16ul) + s * (uint)(C / 8);
                float4 n[C / 4];
                #pragma clang loop unroll(full)
                for (int j = 0; j < C / 8; ++j) {
                    uint u = p[j];
                    n[2 * j] = nib4s(u);
                    n[2 * j + 1] = nib4s(u >> 16);
                }
                #pragma clang loop unroll(full)
                for (int e = 0; e < NB; ++e) {
                    float gs = dot(n[0], xf[e][0]);
                    #pragma clang loop unroll(full)
                    for (int k = 1; k < C / 4; ++k) gs += dot(n[k], xf[e][k]);
                    acc[e][ri] += scale[ri] * gs;
                }
            }
        }
        #pragma clang loop unroll(full)
        for (int e = 0; e < NB; ++e) {
            float m8 = -8.0f * sx[e];
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) acc[e][ri] += scale[ri] * m8;
        }
    }
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            float v = simd_sum(acc[e][ri]);
            if (lane == 0u && (uint)ri < nr) y[(ulong)e * rows + r0 + (uint)ri] = v;
        }
    }
}


// steps of four columns: one float4 of activations a vector in registers, a row's step is one 16-bit load
template <int NB, int R>
inline void mq(device const uchar* q, device const float* x, device float* y, uint gpr, uint rows,
               uint sg, uint lane, uint tgpos, uint sgs, threadgroup float* lad)
{
    uint r0 = (tgpos * sgs + sg) * (uint)R;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, (uint)R) : 0u;
    uint cols = gpr * 32u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * (uint)R + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float acc[NB][R];
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) acc[e][ri] = 0.0f;
    }
    for (uint g = lane; g < gpr; g += 32u) {
        uint bit = g * 5u; uint cb = bit >> 3u; uint shf = bit & 7u;
        float scale[R];
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            uint r = min(r0 + (uint)ri, rows - 1u);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            scale[ri] = lad[(sg * (uint)R + min((uint)ri, nr - 1u)) * 32u + code];
        }
        float sx[NB];
        #pragma clang loop unroll(full)
        for (int e = 0; e < NB; ++e) sx[e] = 0.0f;
        #pragma clang loop unroll(disable)
        for (uint s = 0u; s < 8u; ++s) {
            float4 xf[NB];
            #pragma clang loop unroll(full)
            for (int e = 0; e < NB; ++e) {
                float4 v = *(device const float4*)(x + (ulong)e * cols + g * 32u + s * 4u);
                sx[e] += (v.x + v.y) + (v.z + v.w);
                xf[e] = v * sc;
            }
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) {
                uint r = min(r0 + (uint)ri, rows - 1u);
                uint u = (uint)(*((device const ushort*)(q + ((ulong)r * gpr + (ulong)g) * 16ul) + s));
                float4 n = nib4s(u);
                #pragma clang loop unroll(full)
                for (int e = 0; e < NB; ++e) acc[e][ri] += scale[ri] * dot(n, xf[e]);
            }
        }
        #pragma clang loop unroll(full)
        for (int e = 0; e < NB; ++e) {
            float m8 = -8.0f * sx[e];
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) acc[e][ri] += scale[ri] * m8;
        }
    }
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            float v = simd_sum(acc[e][ri]);
            if (lane == 0u && (uint)ri < nr) y[(ulong)e * rows + r0 + (uint)ri] = v;
        }
    }
}

// Activations as 16-bit integers in registers (half the registers of f32), nibbles as integers:
// a group's sum is exact in int32 (32 · 15 · 32767 < 2^31).
template <int NB, int R>
inline void mi(device const uchar* q, device const float* x, device float* y, uint gpr, uint rows,
               uint sg, uint lane, uint tgpos, uint sgs, threadgroup float* lad)
{
    const float QS = 262144.0f;
    uint r0 = (tgpos * sgs + sg) * (uint)R;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, (uint)R) : 0u;
    uint cols = gpr * 32u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * (uint)R + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    float acc[NB][R];
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) acc[e][ri] = 0.0f;
    }
    for (uint g = lane; g < gpr; g += 32u) {
        uint bit = g * 5u; uint cb = bit >> 3u; uint shf = bit & 7u;
        short4 xs[NB][8];
        float m8[NB];
        #pragma clang loop unroll(full)
        for (int e = 0; e < NB; ++e) {
            device const float4* xv = (device const float4*)(x + (ulong)e * cols + g * 32u);
            int4 s4 = int4(0);
            #pragma clang loop unroll(full)
            for (int k = 0; k < 8; ++k) {
                short4 v = short4(rint(xv[k] * QS));
                s4 += int4(v);
                xs[e][k] = v;
            }
            m8[e] = -8.0f * (float)(s4.x + s4.y + s4.z + s4.w);
        }
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            uint r = min(r0 + (uint)ri, rows - 1u);
            uint4 b = *(device const uint4*)(q + ((ulong)r * gpr + (ulong)g) * 16ul);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = lad[(sg * (uint)R + min((uint)ri, nr - 1u)) * 32u + code] * (1.0f / QS);
            short4 n[8];
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                uint u = b[j];
                n[2 * j] = short4(ushort4((ushort)(u & 0xFu), (ushort)((u >> 4u) & 0xFu), (ushort)((u >> 8u) & 0xFu), (ushort)((u >> 12u) & 0xFu)));
                n[2 * j + 1] = short4(ushort4((ushort)((u >> 16u) & 0xFu), (ushort)((u >> 20u) & 0xFu), (ushort)((u >> 24u) & 0xFu), (ushort)(u >> 28u)));
            }
            #pragma clang loop unroll(full)
            for (int e = 0; e < NB; ++e) {
                int4 v = int4(n[0]) * int4(xs[e][0]);
                #pragma clang loop unroll(full)
                for (int k = 1; k < 8; ++k) v += int4(n[k]) * int4(xs[e][k]);
                acc[e][ri] += scale * ((float)(v.x + v.y + v.z + v.w) + m8[e]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            float v = simd_sum(acc[e][ri]);
            if (lane == 0u && (uint)ri < nr) y[(ulong)e * rows + r0 + (uint)ri] = v;
        }
    }
}

// The batched matvec with masked nibbles: packed words of R rows resident, one vector's
// activations in registers at a time, AND + convert + multiply-add a weight a vector.
template <int NB, int R>
inline void mk(device const uchar* q, device const float* x, device float* y, uint gpr, uint rows,
               uint sg, uint lane, uint tgpos, uint sgs, threadgroup float* lad)
{
    uint r0 = (tgpos * sgs + sg) * (uint)R;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, (uint)R) : 0u;
    uint cols = gpr * 32u;
    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * (uint)R + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    const float4 sc = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    float acc[NB][R];
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) acc[e][ri] = 0.0f;
    }
    for (uint g = lane; g < gpr; g += 32u) {
        uint bit = g * 5u; uint cb = bit >> 3u; uint shf = bit & 7u;
        uint4 w[R];
        float scale[R];
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            uint r = min(r0 + (uint)ri, rows - 1u);
            w[ri] = *(device const uint4*)(q + ((ulong)r * gpr + (ulong)g) * 16ul);
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            scale[ri] = lad[(sg * (uint)R + min((uint)ri, nr - 1u)) * 32u + code];
        }
        #pragma clang loop unroll(full)
        for (int e = 0; e < NB; ++e) {
            device const float4* xv = (device const float4*)(x + (ulong)e * cols + g * 32u);
            float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
            float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
            float4 s4 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            float m8 = -8.0f * (s4.x + s4.y + s4.z + s4.w);
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < R; ++ri) {
                uint4 b = w[ri];
                float4 v = nib4s(b.x) * x0 + nib4s(b.x >> 16) * x1 + nib4s(b.y) * x2 + nib4s(b.y >> 16) * x3
                         + nib4s(b.z) * x4 + nib4s(b.z >> 16) * x5 + nib4s(b.w) * x6 + nib4s(b.w >> 16) * x7;
                acc[e][ri] += scale[ri] * (dot(v, sc) + m8);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int e = 0; e < NB; ++e) {
        #pragma clang loop unroll(full)
        for (int ri = 0; ri < R; ++ri) {
            float v = simd_sum(acc[e][ri]);
            if (lane == 0u && (uint)ri < nr) y[(ulong)e * rows + r0 + (uint)ri] = v;
        }
    }
}

#define K(NAME, F, NB, R, C) \
kernel void NAME(device const uchar* q [[buffer(0)]], device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]], constant uint& gpr [[buffer(3)]], constant uint& rows [[buffer(4)]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint tgpos [[threadgroup_position_in_grid]], uint sgs [[simdgroups_per_threadgroup]]) \
{ threadgroup float lad[8u * R * 32u]; F<NB, R, C>(q, x, y, gpr, rows, sg, lane, tgpos, sgs, lad); }
#define K2(NAME, F, NB, R) \
kernel void NAME(device const uchar* q [[buffer(0)]], device const float* x [[buffer(1)]], \
    device float* y [[buffer(2)]], constant uint& gpr [[buffer(3)]], constant uint& rows [[buffer(4)]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint tgpos [[threadgroup_position_in_grid]], uint sgs [[simdgroups_per_threadgroup]]) \
{ threadgroup float lad[8u * R * 32u]; F<NB, R>(q, x, y, gpr, rows, sg, lane, tgpos, sgs, lad); }

K(m2c32r4, mb, 2, 4, 32)
K2(i2r4, mi, 2, 4)
K2(i3r4, mi, 3, 4)
K2(i4r4, mi, 4, 4)
K2(k2r4, mk, 2, 4)
K2(k3r4, mk, 3, 4)
K2(k4r4, mk, 4, 4)
K2(k2r8, mk, 2, 8)
K2(k3r8, mk, 3, 8)
K2(k3r2, mk, 3, 2)

K(s2c16r4, ms, 2, 4, 16)
K(s3c16r4, ms, 3, 4, 16)
K(s3c8r4, ms, 3, 4, 8)
K(s4c16r4, ms, 4, 4, 16)
K(s4c8r4, ms, 4, 4, 8)
K(s3c16r8, ms, 3, 8, 16)
K(s4c8r8, ms, 4, 8, 8)

K(t1c16r4, mt, 1, 4, 16)
K(t2c16r4, mt, 2, 4, 16)
K(t2c8r4, mt, 2, 4, 8)
K(t3c16r4, mt, 3, 4, 16)
K(t3c8r4, mt, 3, 4, 8)
K(t3c8r2, mt, 3, 2, 8)
K(t4c16r4, mt, 4, 4, 16)
K(t4c8r4, mt, 4, 4, 8)
K(t4c8r2, mt, 4, 2, 8)
K(t5c8r4, mt, 5, 4, 8)
K(t6c8r4, mt, 6, 4, 8)

K2(q2r4, mq, 2, 4)
K2(q3r4, mq, 3, 4)
K2(q4r4, mq, 4, 4)
K2(q3r8, mq, 3, 8)
K2(q4r8, mq, 4, 8)
K2(q5r4, mq, 5, 4)
K(s3c8r3, ms, 3, 3, 8)
K(s4c8r3, ms, 4, 3, 8)
K(s3c8r6, ms, 3, 6, 8)
K(s2c8r4, ms, 2, 4, 8)
