#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace rocm {

#if defined(__HIPCC__)

__device__ __forceinline__ float gfx906_ds_swizzle_xor(float value,
                                                       int xor_mask) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
  const int int_value = __float_as_int(value);
  int result_int = int_value;
  switch (xor_mask) {
    case 1:
      result_int = __builtin_amdgcn_ds_swizzle(int_value, 0x041F);
      break;
    case 2:
      result_int = __builtin_amdgcn_ds_swizzle(int_value, 0x081F);
      break;
    case 4:
      result_int = __builtin_amdgcn_ds_swizzle(int_value, 0x101F);
      break;
    case 8:
      result_int = __builtin_amdgcn_ds_swizzle(int_value, 0x201F);
      break;
    case 16:
      result_int = __builtin_amdgcn_ds_swizzle(int_value, 0x401F);
      break;
    default:
      break;
  }
  return __int_as_float(result_int);
#else
  return __shfl_xor(value, xor_mask);
#endif
}

template <int warp_size>
__device__ __forceinline__ float gfx906_wave_reduce_sum(float value) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
  value += gfx906_ds_swizzle_xor(value, 1);
  value += gfx906_ds_swizzle_xor(value, 2);
  value += gfx906_ds_swizzle_xor(value, 4);
  value += gfx906_ds_swizzle_xor(value, 8);
  value += gfx906_ds_swizzle_xor(value, 16);
  if constexpr (warp_size == 64) {
    value += __shfl_xor(value, 32, 64);
  }
  return value;
#else
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset, warp_size);
  }
  return value;
#endif
}

template <int warp_size>
__device__ __forceinline__ float gfx906_wave_reduce_max(float value) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
  float shuffled = gfx906_ds_swizzle_xor(value, 1);
  value = value > shuffled ? value : shuffled;
  shuffled = gfx906_ds_swizzle_xor(value, 2);
  value = value > shuffled ? value : shuffled;
  shuffled = gfx906_ds_swizzle_xor(value, 4);
  value = value > shuffled ? value : shuffled;
  shuffled = gfx906_ds_swizzle_xor(value, 8);
  value = value > shuffled ? value : shuffled;
  shuffled = gfx906_ds_swizzle_xor(value, 16);
  value = value > shuffled ? value : shuffled;
  if constexpr (warp_size == 64) {
    float cross = __shfl_xor(value, 32, 64);
    value = value > cross ? value : cross;
  }
  return value;
#else
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    float other = __shfl_xor(value, offset, warp_size);
    value = value > other ? value : other;
  }
  return value;
#endif
}

__device__ __forceinline__ void gfx906_mad(float &acc, const __half2 v,
                                           const __half2 u) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
  const float2 vf = __half22float2(v);
  const float2 uf = __half22float2(u);
  acc += vf.x * uf.x + vf.y * uf.y;
#endif
}

__device__ __forceinline__ int gfx906_dp4a(int a, int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
  return __builtin_amdgcn_sdot4(a, b, c, false);
#else
  const int8_t *a8 = reinterpret_cast<const int8_t *>(&a);
  const int8_t *b8 = reinterpret_cast<const int8_t *>(&b);
  return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

#endif  // defined(__HIPCC__)

}  // namespace rocm
}  // namespace vllm
