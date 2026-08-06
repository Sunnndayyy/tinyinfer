#include <metal_stdlib>
using namespace metal;

constant uint GROUP_SIZE = 32;
constant uint OUTPUTS_PER_THREADGROUP = 4;
constant uint VALUES_PER_LANE = 4;
constant uint VALUES_PER_SIMDGROUP = GROUP_SIZE * VALUES_PER_LANE;

kernel void q8_linear_bf16(
    device const bfloat* inputs,
    device const char* weights,
    device const half* scales,
    device const bfloat* bias,
    device bfloat* output,
    constant uint& input_width,
    constant uint& output_width,
    constant uint& has_bias,
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  // Each SIMD group owns one output value; its 32 lanes walk Q8 groups together.
  uint output_column = group.x * OUTPUTS_PER_THREADGROUP + simdgroup;
  if (output_column >= output_width) return;

  uint input_row = group.y;
  uint input_offset = input_row * input_width;
  uint weight_offset = output_column * input_width;
  uint scale_offset = output_column * (input_width / GROUP_SIZE);
  float total = 0.0f;

  for (uint first = lane * VALUES_PER_LANE; first < input_width;
       first += VALUES_PER_SIMDGROUP) {
    // Widths are divisible by 32, so every aligned four-value load is complete.
    float scale = float(scales[scale_offset + first / GROUP_SIZE]);
    bfloat4 values = *reinterpret_cast<device const bfloat4*>(inputs + input_offset + first);
    char4 packed = *reinterpret_cast<device const char4*>(weights + weight_offset + first);
    total += dot(float4(values), float4(packed)) * scale;
  }

  total = simd_sum(total);
  if (lane == 0) {
    if (has_bias) total += float(bias[output_column]);
    output[input_row * output_width + output_column] = bfloat(total);
  }
}

kernel void q8_embedding_bf16(
    device const long* token_ids,
    device const char* weights,
    device const half* scales,
    device bfloat* output,
    constant uint& width,
    uint vector_index [[thread_position_in_grid]]) {
  uint vectors_per_row = width / 4;
  uint token_position = vector_index / vectors_per_row;
  uint first = (vector_index % vectors_per_row) * 4;
  uint token = uint(token_ids[token_position]);
  uint weight_offset = token * width + first;
  uint scale_offset = token * (width / GROUP_SIZE) + first / GROUP_SIZE;
  char4 packed = *reinterpret_cast<device const char4*>(weights + weight_offset);
  bfloat4 restored = bfloat4(float4(packed) * float(scales[scale_offset]));
  *reinterpret_cast<device bfloat4*>(output + token_position * width + first) = restored;
}
