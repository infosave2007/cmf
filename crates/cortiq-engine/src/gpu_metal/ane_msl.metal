// The GPU's side of a dense FFN on the Neural Engine (`gpu_metal/ane.rs`), after Splash's
// runtime/metal/kernels/prefill/ane_ffn.metal (Inco AI, Apache-2.0; THIRD_PARTY_NOTICES.md).
#include <metal_stdlib>
using namespace metal;

#define TILE 32u
#define TILE_ROWS 8u

// TILE x TILE tiles of rows x channels: the rows, transposed, as fp16. Rows past `rows` are zeros.
kernel void ane_pack16(device const float *input [[buffer(0)]],
                       device half *packed [[buffer(1)]],
                       constant uint &hidden [[buffer(2)]],
                       constant uint &channel0 [[buffer(3)]],
                       constant uint &stride [[buffer(4)]],
                       constant uint &rows [[buffer(5)]],
                       uint2 tile [[threadgroup_position_in_grid]],
                       uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[TILE][TILE + 1];
  const uint row = tile.x * TILE, channel = tile.y * TILE;
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    staged[j][position.x] = row + j < rows ? input[(row + j) * hidden + channel0 + channel + position.x] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    packed[(channel + j) * stride + row + position.x] = half(staged[position.x][j]);
}

// The ANE's output as the residual's delta: each token's intermediate scale (row `hidden` of the
// partial) and input scale multiply in fp32, since the output at full scale could overflow fp16.
// The same tiles the other way: a thread per value read 2 of every 512 bytes of the surface.
kernel void ane_join(device float *output [[buffer(0)]],
                     device const half *partial [[buffer(1)]],
                     device const half *token_scale [[buffer(2)]],
                     constant uint &hidden [[buffer(3)]],
                     constant uint &stride [[buffer(4)]],
                     constant uint &rows [[buffer(5)]],
                     uint2 tile [[threadgroup_position_in_grid]],
                     uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[TILE][TILE + 1];
  const uint row = tile.x * TILE, channel = tile.y * TILE;
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    staged[j][position.x] = channel + j < hidden ? float(partial[(channel + j) * stride + row + position.x]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < TILE; j += TILE_ROWS) {
    const uint token = row + j;
    if (token < rows && channel + position.x < hidden) {
      const float scale = float(partial[hidden * stride + token]) * float(token_scale[token]);
      output[token * hidden + channel + position.x] = staged[position.x][j] * scale;
    }
  }
}

// A group of projections' output, rows under rows, as one projection's rows x width.
kernel void ane_unpack(device float *output [[buffer(0)]],
                       device const half *partial [[buffer(1)]],
                       constant uint &width [[buffer(2)]],
                       constant uint &stride [[buffer(3)]],
                       constant uint &first [[buffer(4)]],
                       constant uint &rows [[buffer(5)]],
                       uint2 tile [[threadgroup_position_in_grid]],
                       uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[TILE][TILE + 1];
  const uint row = tile.x * TILE, channel = tile.y * TILE;
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    staged[j][position.x] = channel + j < width ? float(partial[(first + channel + j) * stride + row + position.x]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < TILE; j += TILE_ROWS) {
    const uint token = row + j;
    if (token < rows && channel + position.x < width)
      output[token * width + channel + position.x] = staged[position.x][j];
  }
}
