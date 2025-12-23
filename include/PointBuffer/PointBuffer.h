#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <cuda_runtime.h>

enum class MemSpace { Host, Device, Unified };

template <typename T>
class PointBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "PointBuffer<T> requires trivially copyable T");

public:
    PointBuffer() = default;

    explicit PointBuffer(size_t n, MemSpace space = MemSpace::Unified)
        : count_(n), space_(space)
    {
        allocate();
    }

    ~PointBuffer() {
        release();
    }

    PointBuffer(const PointBuffer&) = delete;
    PointBuffer& operator=(const PointBuffer&) = delete;

    PointBuffer(PointBuffer&& other) noexcept {
        moveFrom(other);
    }

    PointBuffer& operator=(PointBuffer&& other) noexcept {
        if (this != &other) {
            release();
            moveFrom(other);
        }
        return *this;
    }

    // Append copy
    PointBuffer& operator+=(const PointBuffer& rhs) {
        append(rhs.buf_, rhs.count_, rhs.space_);
        return *this;
    }

    // Append move
    PointBuffer& operator+=(PointBuffer&& rhs) noexcept {
        if (count_ == 0) {
            moveFrom(rhs);
            return *this;
        }

        append(rhs.buf_, rhs.count_, rhs.space_);
        rhs.release();
        rhs.count_ = 0;
        return *this;
    }

    PointBuffer to(MemSpace newSpace) const {
        PointBuffer out(count_, newSpace);
        copyMemory(out.buf_, newSpace, buf_, space_, count_);
        return out;
    }

    T* data() noexcept { return buf_; }
    const T* data() const noexcept { return buf_; }
    size_t size() const noexcept { return count_; }
    MemSpace space() const noexcept { return space_; }

private:
    size_t count_{0};
    T* buf_{nullptr};
    MemSpace space_{MemSpace::Host};

private:
    void allocate() {
        if (count_ == 0) return;

        size_t bytes = count_ * sizeof(T);

        switch (space_) {
        case MemSpace::Host:
            buf_ = static_cast<T*>(std::malloc(bytes));
            break;
        case MemSpace::Device:
            cudaMalloc(&buf_, bytes);
            break;
        case MemSpace::Unified:
            cudaMallocManaged(&buf_, bytes);
            break;
        }
    }

    void release() {
        if (!buf_) return;

        switch (space_) {
        case MemSpace::Host:
            std::free(buf_);
            break;
        case MemSpace::Device:
        case MemSpace::Unified:
            cudaFree(buf_);
            break;
        }

        buf_ = nullptr;
    }

    void moveFrom(PointBuffer& other) {
        buf_   = other.buf_;
        count_ = other.count_;
        space_ = other.space_;

        other.buf_ = nullptr;
        other.count_ = 0;
    }

    static void copyMemory(
        T* dst, MemSpace dstSpace,
        const T* src, MemSpace srcSpace,
        size_t count)
    {
        size_t bytes = count * sizeof(T);

        if (dstSpace == MemSpace::Host && srcSpace == MemSpace::Host) {
            std::memcpy(dst, src, bytes);
            return;
        }

        cudaMemcpyKind kind =
            (srcSpace == MemSpace::Host && dstSpace != MemSpace::Host)
                ? cudaMemcpyHostToDevice :
            (srcSpace != MemSpace::Host && dstSpace == MemSpace::Host)
                ? cudaMemcpyDeviceToHost :
                  cudaMemcpyDeviceToDevice;

        cudaMemcpy(dst, src, bytes, kind);
    }

    void append(const T* rhsBuf, size_t rhsCount, MemSpace rhsSpace) {
        if (rhsCount == 0) return;

        size_t newCount = count_ + rhsCount;
        size_t newBytes = newCount * sizeof(T);

        T* newBuf = nullptr;

        switch (space_) {
        case MemSpace::Host:
            newBuf = static_cast<T*>(std::malloc(newBytes));
            std::memcpy(newBuf, buf_, count_ * sizeof(T));
            copyMemory(newBuf + count_, MemSpace::Host, rhsBuf, rhsSpace, rhsCount);
            break;

        case MemSpace::Device:
            cudaMalloc(&newBuf, newBytes);
            copyMemory(newBuf, MemSpace::Device, buf_, space_, count_);
            copyMemory(newBuf + count_, MemSpace::Device, rhsBuf, rhsSpace, rhsCount);
            break;

        case MemSpace::Unified:
            cudaMallocManaged(&newBuf, newBytes);
            copyMemory(newBuf, MemSpace::Unified, buf_, space_, count_);
            copyMemory(newBuf + count_, MemSpace::Unified, rhsBuf, rhsSpace, rhsCount);
            break;
        }

        release();
        buf_ = newBuf;
        count_ = newCount;
    }
};

template <typename T>
class GLPointBufferView {
public:
    explicit GLPointBufferView(PointBuffer<T>& buffer)
        : buffer_(buffer)
    {
        // Create SSBO
        glGenBuffers(1, &glBuffer_);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBuffer_);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            buffer_.size() * sizeof(T),
            nullptr,
            GL_DYNAMIC_DRAW
        );
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Register with CUDA
        cudaError_t err =
            cudaGraphicsGLRegisterBuffer(
                &cudaRes_,
                glBuffer_,
                cudaGraphicsRegisterFlagsNone
            );

        if (err != cudaSuccess) {
            throw std::runtime_error(
                "cudaGraphicsGLRegisterBuffer failed"
            );
        }
    }

    ~GLPointBufferView() {
        if (cudaRes_) {
            cudaGraphicsUnregisterResource(cudaRes_);
        }
        if (glBuffer_) {
            glDeleteBuffers(1, &glBuffer_);
        }
    }

    GLPointBufferView(const GLPointBufferView&) = delete;
    GLPointBufferView& operator=(const GLPointBufferView&) = delete;

    class CudaMapping {
    public:
        T* data() noexcept { return ptr_; }

        ~CudaMapping() {
            cudaGraphicsUnmapResources(1, &res_, 0);
        }

    private:
        friend class GLPointBufferView;
        CudaMapping(cudaGraphicsResource* res, T* ptr)
            : res_(res), ptr_(ptr) {}

        cudaGraphicsResource* res_;
        T* ptr_;
    };

    CudaMapping mapCuda() {
        cudaGraphicsMapResources(1, &cudaRes_, 0);

        void* ptr = nullptr;
        size_t bytes = 0;
        cudaGraphicsResourceGetMappedPointer(
            &ptr, &bytes, cudaRes_
        );

        return CudaMapping(cudaRes_, static_cast<T*>(ptr));
    }

    GLuint glBuffer() const noexcept { return glBuffer_; }
    size_t size() const noexcept { return buffer_.size(); }

private:
    PointBuffer<T>& buffer_;              // non-owning
    GLuint glBuffer_{0};
    cudaGraphicsResource* cudaRes_{nullptr};
};
