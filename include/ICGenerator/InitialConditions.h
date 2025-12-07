#pragma once

#include "PointBuffer/PointBuffer.h"

#include "cuda_runtime.h"

enum class SampleOption{ uniform, rand};

struct OneD {

    static PointBuffer genLine(size_t n, 
        float3 start, 
        float3 end,
        SampleOption opt = SampleOption::uniform,
        MemSpace space = MemSpace::Unified);

    static PointBuffer genCircle(
        size_t n, 
        float3 center, 
        double radius, 
        float3 planeModel, 
        SampleOption opt = SampleOption::uniform,
        MemSpace space = MemSpace::Unified);

    //template <typename F>
    //static InitialConditions gencustom(size_t n, F&& f, MemSpace space = MemSpace::Unified);
};

struct TwoD {

    static PointBuffer genPlane(
        size_t n, 
        float3 center, 
        float3 normal, 
        float width, 
        float height,
        SampleOption opt = SampleOption::uniform, 
        MemSpace space = MemSpace::Unified);

    static PointBuffer genDisk(
        size_t n, 
        float3 center, 
        float3 normal,
        double radius,
        SampleOption opt = SampleOption::uniform, 
        MemSpace space = MemSpace::Unified);
    
    //    template <typename F>
    //static InitialConditions custom(size_t n, F&& f, MemSpace space = MemSpace::Unified);
};

struct ThreeD {
    static PointBuffer genSphere(
        size_t n, 
        float4 model, 
        float4 radius,
        SampleOption opt = SampleOption::uniform, 
        MemSpace space = MemSpace::Unified);
};