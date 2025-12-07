#include "Particles.h"
#include "OdeSolver/Methods.hpp"
#include <unordered_map>
#include <cuda_gl_interop.h>
#include <stdexcept>

Particles::Particles(PointBuffer points, unsigned int trailLenght, SolverMethod method)
    :stateBuf_(std::move(points)), trailLenght_(trailLenght){

    GLuint pob_;
    glGenBuffers(1, &pob_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pob_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, points.size() * trailLenght_ * sizeof(half4), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pob_);

    cudaGraphicsResource *cupob_;
    cudaGraphicsGLRegisterBuffer(&cupob_, pob_, cudaGraphicsMapFlagsWriteDiscard);

    SolverFlags config = SolverFlags::FLAG_DEFAULT;

    if ((trailLenght_ & (trailLenght_ - 1)) == 0){
        config = config | SolverFlags::FLAG_P2;
    }

    if (points.size() * trailLenght_ <  MEMLAYOUT_SWITCH_VAL){
        config = config | SolverFlags::FLAG_TM_MEM;
    }

    auto it = kernelMap.find({method, config});
    if (it == kernelMap.end()) {
        throw std::runtime_error("No matching kernel for this method + flags!");
    }
    kernel_ = it->second;

    numSteps_ = NumSteps[static_cast<int>(method)];
}

void Particles::generate(double dt, int subSteps){
    
    cudaGraphicsMapResources(1, &cupob_, 0);

    void *devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(&devPtr, &size, cupob_);

    dim3 block(128);
    dim3 grid((stateBuf_.size() + block.x - 1) / block.x);
    kernel_(block, grid, (half4*)cupob_ , stateBuf_.data(), trailLenght_, stateBuf_.size(), head_,dt, subSteps);
    head_ =(head_ + numSteps_) %  trailLenght_;

    cudaGraphicsUnmapResources(1, &cupob_, 0);
}