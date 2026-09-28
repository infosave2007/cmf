#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// FP32 throughout. No half/TF32 weights, and fast math is disabled by the host.
struct Dims { uint n, h, heads, dh; };
struct LinearDims { uint n, k, m, gelu; };

kernel void embedding(device const uint *ids [[buffer(0)]],
    device const float *word [[buffer(1)]], device const float *pos [[buffer(2)]],
    device const float *type [[buffer(3)]], device float *out [[buffer(4)]],
    constant Dims &d [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < d.n*d.h) out[i] = (word[ids[i/d.h]*d.h+i%d.h] + type[i%d.h]) + pos[i];
}

float gelu_erf(float x) {
    float z = x / sqrt(2.0f);
    float t = 1.0f / (1.0f + 0.5f * abs(z));
    float e = t * exp(-z*z - 1.26551223f + t*(1.00002368f + t*(0.37409196f
        + t*(0.09678418f + t*(-0.18628806f + t*(0.27886807f + t*(-1.13520398f
        + t*(1.48851587f + t*(-0.82215223f + t*0.17087277f)))))))));
    float erf_z = z >= 0.0f ? 1.0f-e : e-1.0f;
    return (x * (erf_z + 1.0f)) * 0.5f;
}

// Four SIMD groups, one 16x16 output tile; FP32 8-wide K tiles.
kernel void linear(device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]], device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]], constant LinearDims &d [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float a[128], b[128], c[256];
    uint r0=group.y*16, m0=group.x*16;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0=0; k0<d.k; k0+=8) {
        uint row=tid/8, col=tid%8;
        a[tid] = r0+row<d.n && k0+col<d.k ? x[(r0+row)*d.k+k0+col] : 0.0f;
        b[tid] = m0+row<d.m && k0+col<d.k ? w[(m0+row)*d.k+k0+col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 av,bv;
        simdgroup_load(av,a+(sg/2)*64,8,ulong2(0),false);
        simdgroup_load(bv,b+(sg%2)*64,8,ulong2(0),true);
        simdgroup_multiply_accumulate(acc,av,bv,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,c+(sg/2)*128+(sg%2)*8,16,ulong2(0),false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid; i<256; i+=128) {
        uint r=r0+i/16, m=m0+i%16;
        if (r<d.n && m<d.m) {
            float v=c[i]+bias[m];
            y[r*d.m+m] = d.gelu ? gelu_erf(v) : v;
        }
    }
}

// Two-pass variance, no low-precision normalization. One group per token.
kernel void norm(device const float *x [[buffer(0)]],
    device const float *residual [[buffer(1)]], device const float *w [[buffer(2)]],
    device const float *bias [[buffer(3)]], device float *out [[buffer(4)]],
    constant uint &h [[buffer(5)]], constant float &eps [[buffer(6)]],
    constant uint &add_residual [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[4];
    float sum=0;
    for(uint j=tid;j<h;j+=128) sum += x[row*h+j] + (add_residual ? residual[row*h+j] : 0.0f);
    sum=simd_sum(sum);
    if(lane==0) sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean=(sums[0]+sums[1]+sums[2]+sums[3])/float(h);
    // No group may overwrite sums while another still reads the mean.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=0;
    for(uint j=tid;j<h;j+=128) {
        float v=x[row*h+j]+(add_residual ? residual[row*h+j] : 0.0f)-mean;
        var+=v*v;
    }
    var=simd_sum(var);
    if(lane==0) sums[sg]=var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=1.0f/sqrt((sums[0]+sums[1]+sums[2]+sums[3])/float(h)+eps);
    for(uint j=tid;j<h;j+=128) {
        float v=x[row*h+j]+(add_residual ? residual[row*h+j] : 0.0f);
        out[row*h+j]=((v-mean)*inv)*w[j]+bias[j];
    }
}

kernel void attention_scores(device const float *qkv [[buffer(0)]],
    device float *scores [[buffer(1)]], constant Dims &d [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.heads*d.n*d.n) return;
    uint key=i%d.n, query=(i/d.n)%d.n, head=i/(d.n*d.n);
    float sum=0;
    for(uint j=0;j<d.dh;j++) sum+=qkv[query*3*d.h+head*d.dh+j]*qkv[key*3*d.h+d.h+head*d.dh+j];
    scores[i]=sum/sqrt(float(d.dh));
}

kernel void attention_softmax(device float *s [[buffer(0)]],
    constant Dims &d [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    if(row>=d.n*d.heads) return;
    uint start=row*d.n;
    float mx=-INFINITY;
    for(uint j=0;j<d.n;j++) mx=max(mx,s[start+j]);
    float sum=0;
    for(uint j=0;j<d.n;j++) { float v=exp(s[start+j]-mx); s[start+j]=v; sum+=v; }
    float inv=1.0f/sum;
    for(uint j=0;j<d.n;j++) s[start+j]*=inv;
}

kernel void attention_context(device const float *qkv [[buffer(0)]],
    device const float *s [[buffer(1)]], device float *ctx [[buffer(2)]],
    constant Dims &d [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i>=d.n*d.h) return;
    uint query=i/d.h, feature=i%d.h, head=feature/d.dh;
    float sum=0;
    for(uint key=0;key<d.n;key++) sum+=s[(head*d.n+query)*d.n+key]*qkv[key*3*d.h+2*d.h+feature];
    ctx[i]=sum;
}

// One SIMD group per (query, head), for head dimensions <= 32. Online
// softmax avoids the N*N score buffer and three separate dispatches.
kernel void attention_fused(device const float *qkv [[buffer(0)]],
    device float *ctx [[buffer(1)]], constant Dims &d [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head=row/d.n, query=row%d.n;
    uint col=head*d.dh+lane;
    float q=lane<d.dh ? qkv[query*3*d.h+col] : 0.0f;
    float maximum=-INFINITY, denominator=0.0f, value=0.0f;
    for(uint key=0;key<d.n;key++) {
        float k=lane<d.dh ? qkv[key*3*d.h+d.h+col] : 0.0f;
        float score=simd_sum(q*k)/sqrt(float(d.dh));
        float next=max(maximum,score);
        float old_scale=exp(maximum-next), weight=exp(score-next);
        float v=lane<d.dh ? qkv[key*3*d.h+2*d.h+col] : 0.0f;
        value=value*old_scale+v*weight;
        denominator=denominator*old_scale+weight;
        maximum=next;
    }
    if(lane<d.dh) ctx[query*d.h+col]=value/denominator;
}

// Aligned matrices: each SIMD group loads directly from resident device memory.
// Unlike the generic tail kernel, this avoids two whole-threadgroup barriers
// and staging/copying both operands for every 8-wide K tile.
kernel void linear_direct(device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]], device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]], constant LinearDims &d [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint r0=group.y*8, m0=group.x*32+sg*8;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    if(m0<d.m) {
        for(uint k0=0;k0<d.k;k0+=8) {
            simdgroup_float8x8 av,bv;
            simdgroup_load(av,x+r0*d.k+k0,d.k,ulong2(0),false);
            simdgroup_load(bv,w+m0*d.k+k0,d.k,ulong2(0),true);
            simdgroup_multiply_accumulate(acc,av,bv,acc);
        }
    }
    threadgroup float c[256];
    simdgroup_store(acc,c+sg*8,32,ulong2(0),false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<256;i+=128) {
        uint r=r0+i/32, m=group.x*32+i%32;
        if(r<d.n && m<d.m) {
            float v=c[i]+bias[m];
            y[r*d.m+m] = d.gelu ? gelu_erf(v) : v;
        }
    }
}

// Preserve row order for pooling and both FP32 L2 passes. One command buffer
// can now continue into resonance without a host readback.
kernel void pool(device const float *hidden [[buffer(0)]],
    device float *signal [[buffer(1)]], constant Dims &d [[buffer(2)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=d.h) return;
    float v=0.0f;
    for(uint row=0;row<d.n;row++) v+=hidden[row*d.h+col];
    signal[col]=v/float(d.n);
}
// One SIMD group reduces the two FP32 norms; coordinate sums differ only in
// rounding order from the CPU. The unchanged golden tolerance guards this path.
kernel void normalize(device float *signal [[buffer(0)]], constant uint &h [[buffer(1)]],
    uint lane [[thread_index_in_simdgroup]]) {
    float sum=0.0f;
    for(uint j=lane;j<h;j+=32) sum+=signal[j]*signal[j];
    float den=sqrt(simd_sum(sum))+1e-12f;
    sum=0.0f;
    for(uint j=lane;j<h;j+=32) {
        float value=signal[j]/den;
        signal[j]=value;
        sum+=value*value;
    }
    float norm=sqrt(simd_sum(sum));
    if(norm>1e-12f) {
        float inv=1.0f/norm;
        for(uint j=lane;j<h;j+=32) signal[j]*=inv;
    }
}

// A token fits in one SIMD group: retain values in registers across mean and
// variance, eliminating repeated device loads and whole-group barriers.
kernel void norm_simd(device const float *x [[buffer(0)]],
    device const float *residual [[buffer(1)]], device const float *w [[buffer(2)]],
    device const float *bias [[buffer(3)]], device float *out [[buffer(4)]],
    constant uint &h [[buffer(5)]], constant float &eps [[buffer(6)]],
    constant uint &add_residual [[buffer(7)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float values[(NORM_HIDDEN+31)/32];
    float sum=0.0f;
    #pragma clang loop unroll(full)
    for(uint i=0;i<(NORM_HIDDEN+31)/32;i++) {
        uint j=i*32+lane;
        float v=j<h ? x[row*h+j]+(add_residual ? residual[row*h+j] : 0.0f) : 0.0f;
        values[i]=v;
        sum+=v;
    }
    float mean=simd_sum(sum)/float(h);
    float var=0.0f;
    #pragma clang loop unroll(full)
    for(uint i=0;i<(NORM_HIDDEN+31)/32;i++) {
        float v=i*32+lane<h ? values[i]-mean : 0.0f;
        var+=v*v;
    }
    float inv=1.0f/sqrt(simd_sum(var)/float(h)+eps);
    #pragma clang loop unroll(full)
    for(uint i=0;i<(NORM_HIDDEN+31)/32;i++) {
        uint j=i*32+lane;
        if(j<h) out[row*h+j]=((values[i]-mean)*inv)*w[j]+bias[j];
    }
}
