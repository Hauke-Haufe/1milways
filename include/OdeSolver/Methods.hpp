#pragma once
#include <unordered_map>
#include <functional>
#include <cstdint>

#include "half4.cuh"
#include "Euler.cuh"
#include "OdeSolver/FIXCONSTANT.cuh"

int NumSteps[]= {
    EULER_STEPS
};

enum SolverMethod{
    EULER = 1
};

enum SolverFlags : uint32_t {
    FLAG_DEFAULT  = 0,  //PM MEM
    FLAG_P2 = 1 << 0,
    FLAG_TM_MEM = 1 << 1
};

inline SolverFlags operator|(SolverFlags a, SolverFlags b) {
    return static_cast<SolverFlags>  (
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

struct SolverKey {
    SolverMethod method;        
    SolverFlags flags;  

    SolverKey() = default;
    SolverKey(SolverMethod m, SolverFlags f) : method(m), flags(f) {};

    bool operator==(const SolverKey &other) const noexcept {
        return method == other.method && flags == other.flags;
    }
};

struct SolverKeyHash {
    std::size_t operator()(const SolverKey& k) const noexcept {
        return std::hash<int>()(static_cast<int>(k.method)) ^
               (std::hash<uint32_t>()(static_cast<uint32_t>(k.flags)) << 1);
    }
};

using SolverLauncher = void(*)(dim3, dim3, half4*, float4*, int, int, int, float, int);

extern std::unordered_map<SolverKey, SolverLauncher, SolverKeyHash> kernelMap;





