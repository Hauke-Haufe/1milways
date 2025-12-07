#include <ICGenerator/GeneratorKernels.cuh>
#include <ICGenerator/InitialConditions.cuh>
#include <ICGenerator/InitialConditions.h>

template <typename F>
void launchKernel1D(float4* out, size_t n, F f, SampleOption opt){

    switch (opt)
    {
    case SampleOption::uniform:

        dim3 block(128);
        dim3 grid((n + block.x - 1) / block.x);
        gen1DKernelUniform<<<grid, block>>>(out, n, f);

        break;

    case SampleOption::rand:

        float *uRand;

        cudaMalloc(&uRand, N*sizeof(float));

        dim3 block(128), grid((N + 127) / 128);

        generateRandomSample<<<grid, block>>>(uRand, N, 1234UL);

        gen1DRandomKernel<<<grid, block>>>(out, n, f, uRand);

        cudaFree(uRand);
        break;

    default:

        break;
    }
}

template <typename F>
void launchKernel2D(float4* out, size_t n, F f, SampleOption opt){

    switch (opt)
    {
    case SampleOption::uniform:

        dim3 block(128);
        dim3 grid((n + block.x - 1) / block.x);
        gen2DKernelUniform<<<grid, block>>>(out, n, f);

        break;

    case SampleOption::rand:
    
        float *uRand;
        float *vRand;

        cudaMalloc(&uRand, N*sizeof(float));
        cudaMalloc(&vRand, N*sizeof(float));

        dim3 block(128), grid((N + 127) / 128);

        generateRandomSample<<<grid, block>>>(uRand, N, 1234UL);
        generateRandomSample<<<grid, block>>>(vRand, N, 5678UL);

        gen2DRandomKernel<<<grid, block>>>(out, n, f, uRand, vRand);

        cudaFree(uRand);
        cudaFree(vRand);
        break;

    default:

        break;
    }
}

template <typename F>
void launchKernel3D(float4* out, size_t n, F f, SampleOption opt){

    switch (opt)
    {
    case SampleOption::uniform:

        dim3 block(128);
        dim3 grid((n + block.x - 1) / block.x);
        gen3DKernelUniform<<<grid, block>>>(out, n, f);

        break;

    case SampleOption::rand:
    
        float *uRand;
        float *vRand;
        float *wRand;

        cudaMalloc(&uRand, N*sizeof(float));
        cudaMalloc(&vRand, N*sizeof(float));
        cudaMallow(&wRand, N*sizeof(float));

        dim3 block(128), grid((N + 127) / 128);

        generateRandomSample<<<grid, block>>>(uRand, N, 1234UL);
        generateRandomSample<<<grid, block>>>(vRand, N, 5678UL);
        generateRandomSample<<<grid, block>>>(wRand, N, 9012UL);

        gen3DRandomKernel<<<grid, block>>>(out, n, f, uRand, vRand, wRand);

        cudaFree(uRand);
        cudaFree(vRand);
        cudaFree(wRand);

        break;

    default:

        break;
    }
}

void launchGenLine(float4* out, size_t n, float4 start, float4 end, SampleOption opt){

    auto lineSampler = [start, end] __device__ (double u) {
        return make_float4(
            u * start.x + (1-u) * end.x,
            u * start.y + (1-u) * end.y,
            u * start.z + (1-u) * end.z,
            u * start.w + (1-u) * end.w
        );
    };

    launchKernel1D(out, n, lineSampler, opt);
}

void computeCircleBasis(const float3& normal, float3& udir, float3& vdir)
{
    // Normalize normal
    float3 n = normal;
    float invLen = 1.0f / sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    n.x *= invLen; n.y *= invLen; n.z *= invLen;

    // Pick a nonparallel vector
    float3 temp = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);

    // u_dir = normalize(cross(n, temp))
    udir = make_float3(
        n.y * temp.z - n.z * temp.y,
        n.z * temp.x - n.x * temp.z,
        n.x * temp.y - n.y * temp.x
    );
    float invU = 1.0f / sqrtf(udir.x*udir.x + udir.y*udir.y + udir.z*udir.z);
    udir.x *= invU; udir.y *= invU; udir.z *= invU;

    // v_dir = cross(n, u_dir)
    vdir = make_float3(
        n.y * udir.z - n.z * udir.y,
        n.z * udir.x - n.x * udir.z,
        n.x * udir.y - n.y * udir.x
    );
}

void launchGenCircle(float4* out, size_t n, float3 center, double radius, float3 normal, SampleOption opt){

    float3 udir;
    float3 vdir;

    computeCircleBasis(normal, udir, vdir);

    auto circleSamler = [center, radius, udir, vdir] __device__ (double u){
        float theta = float(u * 6.28318530717958647692);
        float cs = cosf(theta);
        float sn = sinf(theta);

        float4 result;
        result.x = center.x + radius * (cs * udir.x + sn * vdir.x);
        result.y = center.y + radius * (cs * udir.y + sn * vdir.y);
        result.z = center.z + radius * (cs * udir.z + sn * vdir.z);
        result.w = 1.0f;
        return result;
    };

    launchKernel1D(out, n, circleSamler, opt);
}

void computePlaneBasis(const float3& normal, float3& udir, float3& vdir)
{
    // Normalize normal
    float3 n = normal;
    float inv = 1.0f / sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    n.x *= inv; n.y *= inv; n.z *= inv;

    // Pick arbitrary vector not parallel to n
    float3 temp = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);

    // udir = normalized cross(n, temp)
    udir = make_float3(
        n.y * temp.z - n.z * temp.y,
        n.z * temp.x - n.x * temp.z,
        n.x * temp.y - n.y * temp.x
    );
    float invU = 1.0f / sqrtf(udir.x*udir.x + udir.y*udir.y + udir.z*udir.z);
    udir.x *= invU; udir.y *= invU; udir.z *= invU;

    // vdir = cross(n, udir)
    vdir = make_float3(
        n.y * udir.z - n.z * udir.y,
        n.z * udir.x - n.x * udir.z,
        n.x * udir.y - n.y * udir.x
    );
}

void launchGenPlane(float4* out, size_t n, float3 center, float3 normal, float width, float height, SampleOption opt = SampleOption::uniform){

    float3 udir;
    float3 vdir;

    computePlaneBasis(normal, udir, vdir);

    auto planeSampler = [center, width, height, udir, vdir] __device__ (double u, double v) -> float4{
        float du = float(u * 2.0 - 1.0);  
        float dv = float(v * 2.0 - 1.0);

        float4 p;
        p.x = center.x + 0.5f * width  * (du * udir.x) + 0.5f * height * (dv * vdir.x);
        p.y = center.y + 0.5f * width  * (du * udir.y) + 0.5f * height * (dv * vdir.y);
        p.z = center.z + 0.5f * width  * (du * udir.z) + 0.5f * height * (dv * vdir.z);
        p.w = 1.0f;

        return p;
    };

    launchKernel2D(out, n, planeSampler, opt);
}

void computeDiskBasis(const float3& normal, float3& udir, float3& vdir)
{
    // Normalize
    float3 n = normal;
    float inv = 1.0f / sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    n.x *= inv; n.y *= inv; n.z *= inv;

    float3 temp = (fabsf(n.x) > 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);

    udir = make_float3(
        n.y * temp.z - n.z * temp.y,
        n.z * temp.x - n.x * temp.z,
        n.x * temp.y - n.y * temp.x
    );
    float invU = 1.0f / sqrtf(udir.x*udir.x + udir.y*udir.y + udir.z*udir.z);
    udir.x *= invU; udir.y *= invU; udir.z *= invU;

    // vdir = cross(n, udir)
    vdir = make_float3(
        n.y * udir.z - n.z * udir.y,
        n.z * udir.x - n.x * udir.z,
        n.x * udir.y - n.y * udir.x
    );
}

void launchGenDisk(float4* out, size_t n, float3 center, float3 normal, double radius, SampleOption opt){

    float3 udir;
    float3 vdir;

    computeDiskBasis(normal, udir, vdir);

    auto diskSampler = [center, radius, udir, vdir]
    __device__ (double u, double v) -> float4{
        float theta = float(u * 6.28318530717958647692); // 2πu
        float r = radius * sqrtf(float(v));              // sqrt gives uniform area

        float cs = cosf(theta);
        float sn = sinf(theta);

        float4 p;
        p.x = center.x + r * (cs * udir.x + sn * vdir.x);
        p.y = center.y + r * (cs * udir.y + sn * vdir.y);
        p.z = center.z + r * (cs * udir.z + sn * vdir.z);
        p.w = 1.0f;

        return p;
    };

    launchKernel2D(out, n, diskSampler, opt);

}

