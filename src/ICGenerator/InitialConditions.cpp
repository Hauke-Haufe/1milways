#include "ICGenerator/InitialConditions.h"
#include "ICGenerator/InitialConditions.cuh"

#include <cuda_runtime.h>
#include <functional>

static PointBuffer prepareAndRunKernel(
    size_t n,
    MemSpace requestedSpace,
    std::function<void(float4*, size_t)> kernelLauncher)
{
    //
    // 1. Always allocate a buffer in device space for kernel execution
    //
    PointBuffer deviceBuf(n, MemSpace::Device);

    //
    // 2. Launch the kernel on the device buffer
    //
    kernelLauncher(deviceBuf.data(), n);

    //
    // 3. Convert the results to the user-requested memory space
    //
    if (requestedSpace == MemSpace::Device)
        return deviceBuf;  // Move return

    return deviceBuf.to(requestedSpace);
}

PointBuffer OneD::genLine(size_t n, float3 start, float3 end,
                          SampleOption opt, MemSpace space)
{
    return prepareAndRunKernel(
        n, space,
        [&](float4* ptr, size_t count)
        {
            launchGenLine(ptr, count, start, end, opt);
        }
    );
}

PointBuffer OneD::genCircle(size_t n, float3 center, double radius,
                            float3 normal, SampleOption opt,
                            MemSpace space)
{
    return prepareAndRunKernel(
        n, space,
        [&](float4* ptr, size_t count)
        {
            launchGenCircle(ptr, count, center, radius, normal, opt);
        }
    );
}

PointBuffer TwoD::genPlane(size_t n, float3 center, float3 normal,
                           float width, float height,
                           SampleOption opt, MemSpace space)
{
    return prepareAndRunKernel(
        n, space,
        [&](float4* ptr, size_t count)
        {
            launchGenPlane(ptr, count, center, normal, width, height, opt);
        }
    );
}

PointBuffer TwoD::genDisk(size_t n, float3 center, float3 normal,
                          double radius, SampleOption opt,
                          MemSpace space)
{
    return prepareAndRunKernel(
        n, space,
        [&](float4* ptr, size_t count)
        {
            launchGenDisk(ptr, count, center, normal, radius, opt);
        }
    );
}
