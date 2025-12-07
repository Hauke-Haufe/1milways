#pragma once

#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>

enum class MemSpace { Host, Device, Unified };

class PointBuffer {
public:
    PointBuffer(size_t n, MemSpace space = MemSpace::Unified);
    ~PointBuffer();

    PointBuffer(const PointBuffer&) = delete;
    PointBuffer& operator=(const PointBuffer&) = delete;
    PointBuffer(PointBuffer&& other) noexcept;
    PointBuffer& operator=(PointBuffer&& other) noexcept;

    PointBuffer& operator+=(const PointBuffer& other) noexcept;
    PointBuffer& operator+=(PointBuffer&& other) noexcept;

    PointBuffer PointBuffer::to(MemSpace newSpace) const;

    float4* data() noexcept { return buf_; }
    const float4* data() const noexcept { return buf_; }
    size_t size() const noexcept { return count_; }

private:
    size_t count_{};
    float4* buf_{};
    MemSpace space_;
};