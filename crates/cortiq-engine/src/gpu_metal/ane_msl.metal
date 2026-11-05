// The GPU's side of a dense FFN on the Neural Engine (`gpu_metal/ane.rs`), after Splash's
// runtime/metal/kernels/prefill/ane_ffn.metal (Inco AI, Apache-2.0; THIRD_PARTY_NOTICES.md).
#include <metal_stdlib>
using namespace metal;

#define TILE 32u
#define TILE_ROWS 8u

// TILE x TILE tiles of rows x channels: the rows, transposed, as fp16.
kernel void ane_pack16(device const float *input [[buffer(0)]],
                       device half *packed [[buffer(1)]],
                       constant uint &hidden [[buffer(2)]],
                       constant uint &channel0 [[buffer(3)]],
                       constant uint &stride [[buffer(4)]],
                       uint2 tile [[threadgroup_position_in_grid]],
                       uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[TILE][TILE + 1];
  const uint row = tile.x * TILE, channel = tile.y * TILE;
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    staged[j][position.x] = input[(row + j) * hidden + channel0 + channel + position.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    packed[(channel + j) * stride + row + position.x] = half(staged[position.x][j]);
}

// The ANE's output as the residual's delta: each token's intermediate scale (row `hidden` of the
// partial) and input scale multiply in fp32, since the output at full scale could overflow fp16.
kernel void ane_join(device float *output [[buffer(0)]],
                     device const half *partial [[buffer(1)]],
                     device const half *token_scale [[buffer(2)]],
                     device atomic_uint *status [[buffer(3)]],
                     constant uint &hidden [[buffer(4)]],
                     constant uint &stride [[buffer(5)]],
                     uint2 id [[thread_position_in_grid]]) {
  const uint channel = id.x, token = id.y;
  const half value = partial[channel * stride + token];
  const float scale = float(partial[hidden * stride + token]) * float(token_scale[token]);
  if ((as_type<ushort>(value) & 0x7c00) == 0x7c00) atomic_store_explicit(status, 1u, memory_order_relaxed);
  output[token * hidden + channel] = float(value) * scale;
}
