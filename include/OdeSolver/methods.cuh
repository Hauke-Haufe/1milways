#pragma once
#include "half4.cuh"
#include "FIXCONSTANT.cuh"
#include "ODE.cuh"

enum class SolverMethod{
    NEWTON
};


__global__ void euler_pm_p2(
    half4 *buffer,  // [particle][K] circular buffer
    float4 *state_buf, 
    int K, int N,
    int cur,                      
    float dt, int substeps
);

__global__ void euler_pm(
    half4 *buffer,  // [particle][K] circular buffer
    float4 *state_buf, 
    int K, int N,
    int cur,                      
    float dt, int substeps
);
