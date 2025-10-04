#pragma once
//#include "half4.cuh"
#include <atomic>

using size_t = unsigned long;
using float4 = float*;

enum class Opt{
    uniform = 0, 
    rand = 1
};


template <typename F>
__global__ void gen1DKernel(float4* out, size_t n, F f);

template <typename F>
__global__ void gen2DKernel(float4* out, size_t n, F f);


//parametrirization should be on [0,1]^n
using oneDGenerator = float4(*)(double samplePointX); 
using twoDGenerator = float4(*)(double samplePointX, double samplePointY);


enum class MemSpace { Host, Device, Unified };

class InitialValues {
public:
    InitialValues(size_t n, MemSpace space = MemSpace::Unified);
    ~InitialValues();

    InitialValues(const InitialValues&) = delete;
    InitialValues& operator=(const InitialValues&) = delete;
    InitialValues(InitialValues&& other) noexcept;
    InitialValues& operator=(InitialValues&& other) noexcept;

    InitialValues& operator+=(const InitialValues& other) noexcept;
    InitialValues& operator+=(InitialValues&& other) noexcept;

    float4* data() noexcept { return buf_; }
    const float4* data() const noexcept { return buf_; }
    size_t size() const noexcept { return count_; }

private:
    size_t count_{};
    float4* buf_{};
    MemSpace space_;
};


struct OneD {
    static InitialValues genline(size_t n, float4 start, float4 end, MemSpace space = MemSpace::Unified);
    static InitialValues gencircle(size_t n, float4 center, double radius, float4 planeModel, MemSpace space = MemSpace::Unified);

    template <typename F>
    static InitialValues gencustom(size_t n, F&& f, MemSpace space = MemSpace::Unified);
};

struct TwoD {
    static InitialValues plane(size_t n, float4 model, float4 obb, MemSpace space = MemSpace::Unified);
    static InitialValues sphere(size_t n, float4 center, double radius, MemSpace space = MemSpace::Unified);

    template <typename F>
    static InitialValues custom(size_t n, F&& f, MemSpace space = MemSpace::Unified);
};

struct ThreeD {
    static InitialValues sphere(size_t n, float4 model, float4 radius, MemSpace space = MemSpace::Unified);
};