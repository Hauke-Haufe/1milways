#include <ICGenerator/GeneratorKernels.cuh>

#include <curand_kernel.h>

template <typename F>
__global__ void gen1DKernelUniform(float4* out, size_t n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double t = double(idx) / double(n-1);
        out[idx] = f(t);
    }
}

template <typename F>
__global__ void gen2DUniform(float4* out, size_t nx, size_t ny, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx < total) {
        int i = idx % nx;      // x index
        int j = idx / nx;      // y index

        double u = double(i) / double(nx - 1);
        double v = double(j) / double(ny - 1);

        out[idx] = f(u, v);
    }
}

template <typename F>
__global__ void gen3DKernel(float4* out, size_t nx, size_t ny, size_t nz, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx < total) {
        
        int i = idx % nx;
        int j = (idx / nx) % ny;
        int k = idx / (nx * ny);

        double u = double(i) / double(nx - 1);
        double v = double(j) / double(ny - 1);
        double w = double(k) / double(nz - 1);

        out[idx] = f(u, v, w);
    }
}

__global__
void generateRandoms(float* randBuf, size_t n, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        randBuf[idx] = curand_uniform(&state);
    }
}

template <typename F>
__global__
void gen1DKernelRandom(float4* out, size_t n, F f, const float* randBuf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double u = randBuf[idx]; 
        out[idx] = f(u);
    }
}

template <typename F>
__global__
void gen2DRandomKernel(float4* out, size_t Nx, size_t Ny,
                       F f, const float* uRand, const float* vRand)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = Nx * Ny;
    if (idx < N) {
        double u = uRand[idx];
        double v = vRand[idx];
        out[idx] = f(u, v);
    }
}

template <typename F>
__global__
void gen2DRandomKernel(float4* out, size_t Nx, size_t Ny,
                       F f, const float* uRand, const float* vRand, const float* wRand)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t N = Nx * Ny;
    if (idx < N) {
        double u = uRand[idx];
        double v = vRand[idx];
        double w = WRand[idx];
        out[idx] = f(u, v, w);
    }
}




