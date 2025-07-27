#include "OdeSolver/Euler.cuh"
#include "OdeSolver/ODE.cuh"
#include "OdeSolver/FIXCONSTANT.cuh"

__device__ void euler_step(float4 &state, float dt, int sub_steps) {
    
    #pragma unroll
    for(int i = 0; i<sub_steps; ++i){
        float4 f = ODE(state);
        state.x += dt * f.x;
        state.y += dt * f.y;
        state.z += dt * f.z;
        state.w += dt;
    }    
}

__global__ void euler_pm_p2(
    half4 *buffer,  // [particle][K] circular buffer
    float4 *state_buf, 
    int K, 
    int N,
    int cur,                      
    float dt, 
    int substeps
) {
    int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= N) return;

    float4 state = state_buf[particle];
    
    half4 local_buf [EULER_STEPS];
    #pragma unroll
    for(int s = 0; s < EULER_STEPS; ++s) {
        euler_step(state, dt,  substeps);
        local_buf[s] = float4_to_half4(state);
    }

    int head = cur;
    #pragma unroll
    for (int s = 0; s < EULER_STEPS; ++s){
        head = (head + 1) & (K - 1);
        int write_idx = particle * K + (head);
        buffer[write_idx] = local_buf[s];
    }

    state_buf[particle] = state;
    head = (head + 1) & (K - 1);
    cur = head;
}

__global__ void  euler_pm(
    half4 *buffer,  // [particle][K]
    float4 *state_buf, 
    int K, int N,
    int cur,                      // current circular index
    float dt, int substeps
) {
    int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= N) return;


    float4 state = state_buf[particle];

    half4 local_buf [EULER_STEPS];
    #pragma unroll
    for (int s = 0; s < EULER_STEPS; ++s) {
        euler_step(state, dt, substeps);
        local_buf[s] = float4_to_half4(state);
    }

    int head =  cur;
    #pragma unroll
    for (int s = 0; s <EULER_STEPS; ++s){
        head++;
        if (head == K) head = 0;
        int write_idx = particle * K + (head);
        buffer[write_idx] = local_buf[s];
    }

    state_buf[particle] = state;
}


void launch_euler_pm(dim3 grid, dim3 block, half4 *buffer, float4 *state_buf, int K, int N,int cur, float dt, int substep){
    euler_pm<<<grid, block>>>(buffer, state_buf, K, N, cur, dt, substep);
};

void launch_euler_pm_p2(dim3 grid, dim3 block, half4 *buffer, float4 *state_buf, int K, int N,int cur, float dt, int substep){
    euler_pm_p2<<<grid, block>>>(buffer, state_buf, K, N, cur, dt, substep);
};