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
        // map 1D index -> 2D coordinates
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