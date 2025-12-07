#include "ICGenerator/InitialConditions.h"
#include "ICGenerator/InitialConditions.cuh"

#include <cuda_runtime.h>


PointBuffer OneD::genLine(size_t n, float3 start, float3 end, SampleOption opt,  MemSpace space){

    PointBuffer ic(n, space);
    auto buffer = ic.data(); 

    launchGenLine(buffer, n, start, end, opt);

    return ic;
}

PointBuffer OneD::genCircle(size_t n, float3 center, double radius, float3 normal, SampleOption opt, MemSpace space){

    PointBuffer ic(n, space);
    
    auto buffer = ic.data();

    launchGenCircle(buffer, n, center, radius, normal, opt);

    return ic;
}

PointBuffer TwoD::genPlane(size_t n, float3 center,  float3 normal, float width, float height, SampleOption opt, MemSpace space){

    PointBuffer ic(n, space);
    
    auto buffer = ic.data();

    launchGenPlane(buffer, n, center, normal, width, height, opt);
    return ic;
}

PointBuffer TwoD::genDisk(size_t n, float3 center,  float3 normal, double radius,  SampleOption opt, MemSpace space){

    PointBuffer ic(n, space);
    
    auto buffer = ic.data();

    launchGenDisk(buffer, n, center, normal,radius, opt);
    return ic;
}