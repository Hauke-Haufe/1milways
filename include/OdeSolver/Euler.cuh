#pragma once    
#include "half4.cuh"

__global__ void euler_pm_p2(
    half4 *buffer,  
    float4 *state_buf, 
    int K, int N,
    int cur,                      
    float dt, int substeps
);

__global__ void euler_pm(
    half4 *buffer,  
    float4 *state_buf, 
    int K, int N,
    int cur,                      
    float dt, int substeps
);

void launch_euler_pm(dim3 grid, dim3 block, half4 *buffer, float4 *state_buf, int K, int N,int cur, float dt, int substep);

void launch_euler_pm_p2(dim3 grid, dim3 block, half4 *buffer, float4 *state_buf, int K, int N,int cur, float dt, int substep);