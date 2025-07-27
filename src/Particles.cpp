#include "Particles.h"
#include "OdeSolver/Methods.hpp"
#include <unordered_map>
#include <cuda_gl_interop.h>
#include <stdexcept>

Particles::Particles(unsigned int numParticles, unsigned int trailLenght, SolverMethod method, IVPConifg iv)
    :numParticles_{numParticles}, trailLenght_{trailLenght}, head_{0}{

    GLuint pob_;
    glGenBuffers(1, &pob_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pob_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * trailLenght_ * sizeof(half4), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pob_);

    cudaGraphicsResource *cupob_;
    cudaGraphicsGLRegisterBuffer(&cupob_, pob_, cudaGraphicsMapFlagsWriteDiscard);

    float4 *statebuf_;
    cudaMalloc((void**)&statebuf_, numParticles_ * sizeof(float4));
    
    initIVP(iv, statebuf_);
    
    SolverFlags config = SolverFlags::FLAG_DEFAULT;

    if ((trailLenght_ & (trailLenght_ - 1)) == 0){
        config = config | SolverFlags::FLAG_P2;
    }

    if (numParticles_ * trailLenght_ <  MEMLAYOUT_SWITCH_VAL){
        config = config | SolverFlags::FLAG_TM_MEM;
    }

    auto it = kernelMap.find({method, config});
    if (it == kernelMap.end()) {
        throw std::runtime_error("No matching kernel for this method + flags!");
    }
    kernel_ = it->second;

    numSteps_ = NumSteps[static_cast<int>(method)];
    
}

Particles::Particles(unsigned int numParticles, unsigned int trailLenght, SolverMethod method, float4* iv)
    :numParticles_{numParticles}, trailLenght_{trailLenght}, head_{0}{

    GLuint pob_;
    glGenBuffers(1, &pob_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pob_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * trailLenght_ * sizeof(half4), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pob_);

    cudaGraphicsResource *cupob_;
    cudaGraphicsGLRegisterBuffer(&cupob_, pob_, cudaGraphicsMapFlagsWriteDiscard);

    float4 *statebuf_;
    cudaMalloc((void**)&statebuf_, numParticles_ * sizeof(float4));
    cudaMemcpy(statebuf_, iv, numParticles_ * sizeof(float4), cudaMemcpyHostToDevice);

    SolverFlags config = SolverFlags::FLAG_DEFAULT;

    if ((trailLenght_ & (trailLenght_ - 1)) == 0){
        config = config | SolverFlags::FLAG_P2;
    }

    if (numParticles_ * trailLenght_ <  MEMLAYOUT_SWITCH_VAL){
        config = config | SolverFlags::FLAG_TM_MEM;
    }

    auto it = kernelMap.find({method, config});
    if (it == kernelMap.end()) {
        throw std::runtime_error("No matching kernel for this method + flags!");
    }
    kernel_ = it->second;

    numSteps_ = NumSteps[static_cast<int>(method)];
}


void Particles::initIVP(IVPConifg iv, float4* statebuf){

}

void Particles::generate(double dt, int subSteps){
    
    cudaGraphicsMapResources(1, &cupob_, 0);

    void *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(&devPtr, &size, cupob_);

    dim3 block(128);
    dim3 grid((numParticles_ + block.x - 1) / block.x);
    kernel_(block, grid, (half4*)cupob_ , statebuf_, trailLenght_, numParticles_, head_,dt, subSteps);
    head_ =(head_ + numSteps_) %  trailLenght_;

    cudaGraphicsUnmapResources(1, &cupob_, 0);
}