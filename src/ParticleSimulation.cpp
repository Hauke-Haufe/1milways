#include "ParticleSimulation.h"
#include "OdeSolver/Methods.hpp"
#include <unordered_map>
#include <cuda_gl_interop.h>
#include <stdexcept>

ParticleSimulation::ParticleSimulation(PointBuffer<float4> ic, unsigned int trailLenght, SolverMethod method)
    :stateBuf_(std::move(ic)), trailLenght_(trailLenght){

    SolverFlags config = SolverFlags::FLAG_NONE;

    if ((trailLenght_ & (trailLenght_ - 1)) == 0){
        config = config | SolverFlags::FLAG_P2;
    }

    if (ic.size() * trailLenght_ <  MEMLAYOUT_SWITCH_VAL){
        config = config | SolverFlags::FLAG_TIME_MAJOR;
    }

    auto it = kernelMap.find({method, config});
    if (it == kernelMap.end()) {
        throw std::runtime_error("No matching kernel for this method + flags!");
    }
    kernel_ = it->second;

    numSteps_ = NumSteps[static_cast<int>(method)];
}

void ParticleSimulation::generate(half4* out, uint32_t trailLength, double dt, int subSteps){

    SolverFlags config = SolverFlags::FLAG_NONE;

    if ((trailLenght_ & (trailLenght_ - 1)) == 0){
        config = config | SolverFlags::FLAG_P2;
    }

    if (stateBuf_.size() * trailLenght_ <  MEMLAYOUT_SWITCH_VAL){
        config = config | SolverFlags::FLAG_TIME_MAJOR;
    }

    auto it = kernelMap.find({method, config});
    if (it == kernelMap.end()) {
        throw std::runtime_error("No matching kernel for this method + flags!");
    }
    kernel_ = it->second;

    dim3 block(128);
    dim3 grid((stateBuf_.size() + block.x - 1) / block.x);
    kernel_(block, grid, out, stateBuf_.data(), trailLenght_, stateBuf_.size(), head_,dt, subSteps);
    head_ =(head_ + numSteps_) %  trailLenght_;
}
