#pragma once

template <typename F>
__global__ void gen1DKernelUniform(float4* out, size_t n, F f);

template <typename F>
__global__ void gen2DUniform(float4* out, size_t nx, size_t ny, F f);

template <typename F>
__global__ void gen3DKernel(float4* out, size_t nx, size_t ny, size_t nz, F f);

__global__
void generateRandomSample(float* randBuf, size_t n, unsigned long seed);

template <typename F>
__global__
void gen1DKernelRandom(float4* out, size_t n, F f, const float* randBuf);

template <typename F>
__global__
void gen2DRandomKernel(float4* out, size_t Nx, size_t Ny, F f, const float* uRand, const float* vRand);

template <typename F>
__global__
void gen2DRandomKernel(float4* out, size_t Nx, size_t Ny, F f, const float* uRand, const float* vRand, const float* wRand);