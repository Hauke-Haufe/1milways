#include "OdeSolver/half4.cuh"
#include "OdeSolver/Euler.cuh"


int main() {
    const int N = 2097152;     // number of particles
    const int K = 128;       // trajectory length
    const int substeps = 1000;
    const float dt = 0.0001f;
    int cur = 0;

    // Allocate buffers
    half4 *d_buffer;
    float4 *d_state_buf;
    cudaMalloc(&d_buffer, N * K * sizeof(half4));
    cudaMalloc(&d_state_buf, N * sizeof(float4));

    // Init host state
    float4 *h_state = new float4[N];
    for (int i = 0; i < N; ++i) {
        h_state[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    cudaMemcpy(d_state_buf, h_state, N * sizeof(float4), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    for(int i = 0; i<600; i ++){
        euler_pm_p2<<<grid, block>>>(
            d_buffer, d_state_buf, K, N, cur, dt, substeps);
        cur =(cur + OPTIMAL_EULER_STEPS) %K;
        cudaDeviceSynchronize();
    }

    // Copy back one trajectory to check
    half4 *h_buffer = new half4[N * K];
    cudaMemcpy(h_buffer, d_buffer, N * K * sizeof(half4), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_state;
    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_state_buf);

    return 0;
}