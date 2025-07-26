#pragma once
inline  __device__ float4 ODE(const float4 &s) {
    return make_float4(
            s.x - s.y,
            s.x - s.x * s.z - s.y,
            s.x * s.y - s.z,
            0.0f
        );
}