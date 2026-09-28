#include <metal_stdlib>
using namespace metal;

// Sequential projection updates, just as reference_error: never assume an
// orthogonal basis or replace the residual by norm(x)^2-sum(projections^2).
// Only the coordinate reductions are parallel, in FP32 without fast math.
kernel void reconstruction_errors(device const float *x [[buffer(0)]],
    device const float *means [[buffer(1)]], device const float *basis [[buffer(2)]],
    device const uint *offsets [[buffer(3)]], device const uint *ranks [[buffer(4)]],
    device float *errors [[buffer(5)]], constant uint &dim [[buffer(6)]],
    uint task [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[4];
    // Per-lane residual in registers: no 18 KB threadgroup allocation that
    // limits occupancy to one topology per GPU core on the release model.
    float residual[(SIGNAL_DIM+127)/128];
    #pragma clang loop unroll(full)
    for(uint j=0;j<(SIGNAL_DIM+127)/128;j++) {
        uint coord=j*128+tid;
        residual[j]=coord<SIGNAL_DIM ? x[coord]-means[task*SIGNAL_DIM+coord] : 0.0f;
    }
    for(uint k=0;k<ranks[task];k++) {
        uint start=offsets[task]+k*dim;
        float sum=0;
        float coefficients[(SIGNAL_DIM+127)/128];
        #pragma clang loop unroll(full)
        for(uint j=0;j<(SIGNAL_DIM+127)/128;j++) {
            uint coord=j*128+tid;
            coefficients[j]=coord<SIGNAL_DIM ? basis[start+coord] : 0.0f;
            sum+=residual[j]*coefficients[j];
        }
        sum=simd_sum(sum);
        if(lane==0) sums[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float c=sums[0]+sums[1]+sums[2]+sums[3];
        #pragma clang loop unroll(full)
        for(uint j=0;j<(SIGNAL_DIM+127)/128;j++) {
            uint coord=j*128+tid;
            residual[j]-=c*coefficients[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum=0;
    #pragma clang loop unroll(full)
    for(uint j=0;j<(SIGNAL_DIM+127)/128;j++) sum+=residual[j]*residual[j];
    sum=simd_sum(sum);
    if(lane==0) sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) errors[task]=sums[0]+sums[1]+sums[2]+sums[3];
}
