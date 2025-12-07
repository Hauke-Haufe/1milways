#pragma once

#include <cuda_runtime.h>

void launchGenLine(
    float4* out, 
    size_t n, 
    float3 start, 
    float3 end, 
    SampleOption opt);

void launchGenCircle(
    float4* out, 
    size_t n, 
    float3 center, 
    double radius, 
    float3 normal, 
    SampleOption opt);

void launchGenPlane(
    float4* out, 
    size_t n, 
    float3 center, 
    float3 normal, 
    float width, 
    float height, 
    SampleOption opt);

void launchGenDisk(
    float4* out, 
    size_t n, 
    float3 center, 
    float3 normal, 
    double radius, 
    SampleOption opt);