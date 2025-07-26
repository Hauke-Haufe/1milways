#pragma once
#include <cuda_fp16.h>

struct __align__(8) half4 {
    __half x, y, z, w;
};

__host__ __device__ inline half4 make_half4(__half x, __half y, __half z, __half w) {
    half4 h = {x, y, z, w};
    return h;
}

__host__ __device__ inline half4 float4_to_half4(const float4 &f) {
    return make_half4(
        __float2half_rn(f.x),
        __float2half_rn(f.y),
        __float2half_rn(f.z),
        __float2half_rn(f.w)
    );
}

__host__ __device__ inline float4 half4_to_float4(const half4 &h) {
    return make_float4(
        __half2float(h.x),
        __half2float(h.y),
        __half2float(h.z),
        __half2float(h.w)
    );
}