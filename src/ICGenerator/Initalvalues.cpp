#include "ICGenerator/InitialValues.h"

#include <cstdlib>
#include "string.h"



InitialValues::InitialValues(sizt_t n, MemSpace space){

    space_ = space;
    count_ = n;

    switch (space_)
    {
    case MemSpace::Host:
        buf_ = (float4*)malloc(count_ * sizeof(float4));
        break;

    case MemSpace::Device:
        cudaMalloc(&buf_, count_* sizeof(float4));
        break;

    case MemSpace::Unified:
        cudaMallocManaged(&buf_, count_ * sizeof(float4));
        break;
    default:
        break;
    }

}

InitialValues::InitialValues(InitialValues&& other) noexcept
    : count_(other.count_), buf_(other.buf_), space_(other.space_)
{
    other.count_  = 0;
    other.buf_ = nullptr;
}

InitialValues& InitialValues::operator+=(const InitialValues& rhs) {

    float4* newBuf;
    switch (space_)
    {
    case MemSpace::Host:
        
        newBuf = (float4*)malloc((count_ + rhs.count_) * sizeof(float4));
        memcpy(newBuf, buf_, count_ * sizeof(float4));

        if (rhs.space_ == MemSpace::Host){
            memcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4));
        }
        else{
            cudaMemcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4), cudaMemcpyDeviceToHost);
        }
        free(buf_);
        break;
    case MemSpace::Device:

        cudaMalloc(&newBuf, (count_ + rhs.count_) * sizeof(float4));
        cudaMemcpy(newBuf, buf_, count_ * sizeof(float4), cudaMemcpyDefault);

        if (rhs.space_ == MemSpace::Host){
            cudaMemcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4), cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4), cudaMemcpyDefault);
        }
        cudaFree(buf_);
        break;
    
    case MemSpace::Unified:

        cudaMallocManaged(&newBuf, (count_ +rhs.count_) * sizeof(float4));
        memcpy(newBuf, buf_, count_ * sizeof(float4));
        memcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4));
        cudaFree(buf_);

    default:
        break;
    }

    
    buf_ = newBuf;
    count_ += rhs.count_;
    return *this;
}

InitialValues& InitialValues::operator+=(InitialValues&& rhs) noexcept {


    float4* newBuf;

    if (count_ == 0) {

        buf_ = rhs.buf_;
        count_  = rhs.count_;
        space_  = rhs.space_;
        rhs.buf_ = nullptr;
        rhs.count_  = 0;
    }
    else if (rhs.count_> 0) {
        switch (space_)
        {
        case MemSpace::Host:
            
            newBuf = (float4*)malloc((count_ + rhs.count_) * sizeof(float4));
            memcpy(newBuf, buf_, count_ * sizeof(float4));

            if (rhs.space_ == MemSpace::Host){
                memcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4));
                free(rhs.buf_);
            }
            else{
                cudaMemcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4), cudaMemcpyDeviceToHost);
                cudaFree(rhs.buf_);
            }
            free(buf_);
            break;
        case MemSpace::Device:

            cudaMalloc(&newBuf, (count_ + rhs.count_) * sizeof(float4));
            cudaMemcpy(newBuf, buf_, count_ * sizeof(float4), cudaMemcpyDefault);

            if (rhs.space_ == MemSpace::Host){
                cudaMemcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4), cudaMemcpyHostToDevice);
                free(rhs.buf_);
            }
            else{
                cudaMemcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4), cudaMemcpyDefault);
                cudaFree(rhs.buf_ );
            }
            cudaFree(buf_);
            break;
        
        case MemSpace::Unified:

            cudaMallocManaged(&newBuf, (count_ +rhs.count_) * sizeof(float4));
            memcpy(newBuf, buf_, count_ * sizeof(float4));
            memcpy(newBuf + count_, rhs.buf_, rhs.count_ * sizeof(float4));
            cudaFree(rhs.buf_);
            cudaFree(buf_);

        default:
            break;
        }

    }

    buf_ = newBuf;
    count_ += rhs.count_;
    rhs.buf_ = nullptr;
    rhs.count_  = 0;
    
    return *this;
}

InitialValues OneD::genline(size_t n, float4 start, float4 end, MemSpace space){

    InitialValues iv(n, space);
    auto buffer = iv.getDataPtr(); 

    auto f = [=] __device__ (double u) {
        return make_float4(
            u * start.x + (1-u) * end.x,
            u * start.y + (1-u) * end.y,
            u * start.z + (1-u) * end.z,
            u * start.w + (1-u) * end.w
        );
    };

    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x);
    gen1DKernelUniform<<<grid, block>>>(buffer, n, f);
}

InitialValues OneD::gencircle(size_t n, float4 center, double radius, float4 normal, MemSpace space){

    InitialValues iv(n, space);


    auto f = [] __device__ (double u) {
        

    };

    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x);
    gen1DKernelUniform<<<grid, block>>>(buffer, n, f);
}