// The GPU's side of a dense FFN on the Neural Engine (`gpu_metal/ane.rs`), after Splash's
// runtime/metal/kernels/prefill/ane_ffn.metal (Inco AI, Apache-2.0; THIRD_PARTY_NOTICES). The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard rotation
// v -> diag(sign) H v / sqrt(n) of blocks of n = K * 128 values: orthogonal, it keeps the dot
// products of rows it rotates alike, and spreads their outliers before the int8 scales.
#include <metal_stdlib>
using namespace metal;

#define ROT_UNIT 128u
#define INT8_PEAK 127.0f
#define INT8_UNIT 128.0f
#define PEAK_FLOOR 6.103515625e-05f
#define ROTATE_BLOCKS 8u
#define TILE 32u
#define TILE_ROWS 8u
#define WEIGHT_ROWS 8u

// One block of K * 128 values held four values per lane in each 128 (value
// k * 128 + lane * 4 + e): two butterfly stages in registers and five across the simdgroup
// within each 128, then log2(K) in registers across them.
template <uint K>
inline void rotate_block(thread float (&value)[K][4], uint lane, device const float *sign) {
  for (uint k = 0; k < K; ++k) {
    thread float (&v)[4] = value[k];
    for (uint half_span = 1; half_span < 4; half_span <<= 1)
      for (uint e = 0; e < 4; ++e)
        if (!(e & half_span)) {
          const float a = v[e], b = v[e + half_span];
          v[e] = a + b;
          v[e + half_span] = a - b;
        }
    for (uint mask = 1; mask < 32; mask <<= 1)
      for (uint e = 0; e < 4; ++e) {
        const float other = simd_shuffle_xor(v[e], mask);
        v[e] = (lane & mask) ? other - v[e] : v[e] + other;
      }
  }
  for (uint half_span = 1; half_span < K; half_span <<= 1)
    for (uint k = 0; k < K; ++k)
      if (!(k & half_span))
        for (uint e = 0; e < 4; ++e) {
          const float a = value[k][e], b = value[k + half_span][e];
          value[k][e] = a + b;
          value[k + half_span][e] = a - b;
        }
  const float norm = rsqrt(float(K * ROT_UNIT));
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) value[k][e] *= sign[k * ROT_UNIT + lane * 4 + e] * norm;
}

// One threadgroup of 256 per token, each simdgroup rotating its input blocks: the rotated row
// scaled to +-INT8_PEAK, and the scale back times the ANE's INT8_UNIT.
kernel void ane_rotate(device const float *input [[buffer(0)]],
                       device const float *sign [[buffer(1)]],
                       device half *rotated [[buffer(2)]],
                       device half *token_scale [[buffer(3)]],
                       constant uint &hidden [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint simd_group [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd_groups [[simdgroups_per_threadgroup]]) {
  threadgroup float peaks[32];
  const uint blocks = hidden / ROT_UNIT / simd_groups;
  float value[ROTATE_BLOCKS][1][4];
  float peak = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * hidden + (simd_group * blocks + block) * ROT_UNIT + lane * 4;
    for (uint e = 0; e < 4; ++e) value[block][0][e] = input[origin + e];
    rotate_block<1>(value[block], lane, sign);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[block][0][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak, PEAK_FLOOR) / INT8_PEAK, inverse = 1.0f / scale;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * hidden + (simd_group * blocks + block) * ROT_UNIT + lane * 4;
    for (uint e = 0; e < 4; ++e) rotated[origin + e] = half(value[block][0][e] * inverse);
  }
  if (simd_group == 0 && lane == 0) token_scale[row] = half(scale * INT8_UNIT);
}

// TILE x TILE tiles of rows x channels: the rotated rows, transposed, as int8.
kernel void ane_pack(device const half *rotated [[buffer(0)]],
                     device char *packed [[buffer(1)]],
                     constant uint &hidden [[buffer(2)]],
                     constant uint &channel0 [[buffer(3)]],
                     constant uint &stride [[buffer(4)]],
                     uint2 tile [[threadgroup_position_in_grid]],
                     uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup half staged[TILE][TILE + 1];
  const uint row = tile.x * TILE, channel = tile.y * TILE;
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    staged[j][position.x] = rotated[(row + j) * hidden + channel0 + channel + position.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < TILE; j += TILE_ROWS)
    packed[(channel + j) * stride + row + position.x] =
        char(clamp(rint(float(staged[position.x][j])), -INT8_PEAK, INT8_PEAK));
}

// The four weights of `row` at inputs [input, input + 4) of a q4tp tensor.
inline void q4tp_values(thread float (&value)[4], device const uchar *q, uint gpr, uint rows, uint row, uint input) {
  const uint group = input >> 5;
  const ulong params = (ulong)rows * gpr * 16ul, codes = params + (ulong)rows * 4ul;
  const uint stride = (gpr * 5u + 7u) / 8u;
  device const half *ph = (device const half *)(q + params + (ulong)row * 4ul);
  const uint bit = group * 5u, shf = bit & 7u;
  device const uchar *cp = q + codes + (ulong)row * stride + (bit >> 3u);
  const uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
  const float s = exp2((float)ph[0] + (float)code * (float)ph[1]);
  device const uchar *b = q + ((ulong)row * gpr + group) * 16ul + ((input & 31u) >> 1u);
  const uint u = (uint)b[0] | ((uint)b[1] << 8);
  for (uint e = 0; e < 4; ++e) value[e] = ((float)((u >> (4u * e)) & 15u) - 8.0f) * s;
}

struct AneWeightParams {
  uint gpr, rows, input, width, stride, scale_stride;
};

// One simdgroup per weight row, WEIGHT_ROWS rows x one rotation block of K * 128 inputs per
// threadgroup: the int8 rows under their shared scale, which the first block also copies into
// the ANE's scale surface.
template <uint K>
kernel void ane_weights(device const uchar *q [[buffer(0)]],
                        device const half *row_scale [[buffer(1)]],
                        device char *output [[buffer(2)]],
                        device half *scale [[buffer(3)]],
                        device const float *sign [[buffer(4)]],
                        constant AneWeightParams &p [[buffer(5)]],
                        uint2 tile [[threadgroup_position_in_grid]],
                        uint simd_group [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile.x * WEIGHT_ROWS + simd_group, block = tile.y * K * ROT_UNIT;
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    q4tp_values(value[k], q, p.gpr, p.rows, row, p.input + block + k * ROT_UNIT + lane * 4);
  rotate_block<K>(value, lane, sign);
  const float inverse = INT8_UNIT / float(row_scale[row]);
  for (uint k = 0; k < K; ++k)
    *(device char4 *)(output + ulong(row) * p.stride + block + k * ROT_UNIT + lane * 4) =
        char4(clamp(rint(float4(value[k][0], value[k][1], value[k][2], value[k][3]) * inverse), -INT8_PEAK, INT8_PEAK));
  if (tile.y == 0 && lane == 0) scale[row * p.scale_stride] = row_scale[row];
}

// Each row's largest weight over the rotated blocks, as the shared scale, times INT8_UNIT.
template <uint K>
kernel void ane_row_scale(device const uchar *q [[buffer(0)]],
                          device half *row_scale [[buffer(1)]],
                          device const float *sign [[buffer(2)]],
                          constant AneWeightParams &p [[buffer(3)]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile * WEIGHT_ROWS + simd_group;
  float peak = 0.0f;
  for (uint block = 0; block < p.width; block += K * ROT_UNIT) {
    float value[K][4];
    for (uint k = 0; k < K; ++k)
      q4tp_values(value[k], q, p.gpr, p.rows, row, p.input + block + k * ROT_UNIT + lane * 4);
    rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, PEAK_FLOOR) / INT8_PEAK * INT8_UNIT);
}

template [[host_name("ane_weights_inputs")]] kernel void ane_weights<1u>(
    device const uchar *, device const half *, device char *, device half *, device const float *,
    constant AneWeightParams &, uint2, uint, uint);
template [[host_name("ane_weights_intermediate")]] kernel void ane_weights<4u>(
    device const uchar *, device const half *, device char *, device half *, device const float *,
    constant AneWeightParams &, uint2, uint, uint);
template [[host_name("ane_row_scale_inputs")]] kernel void ane_row_scale<1u>(
    device const uchar *, device half *, device const float *, constant AneWeightParams &, uint, uint, uint);
template [[host_name("ane_row_scale_intermediate")]] kernel void ane_row_scale<4u>(
    device const uchar *, device half *, device const float *, constant AneWeightParams &, uint, uint, uint);

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
