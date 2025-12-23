#include <glad/gl.h>
#include "OdeSolver/Methods.hpp"
#include "PointBuffer/PointBuffer.h"

constexpr int MEMLAYOUT_SWITCH_VAL = 100000000; // Value for automatic memory Layout Switch 

class ParticleSimulation{

    public:

    ParticleSimulation(PointBuffer<float4> ic, unsigned int trailLenght, SolverMethod methods);

    ~ParticleSimulation();

    void draw() const;
    void generate(half4* circOut, double dt, int subSteps);

    inline uint64_t getTrailLenght(){return trailLenght_;}
    inline uint64_t getNumPoints(){return stateBuf_.size();}
    inline uint64_t head() {return head_;}

    private:

    PointBuffer<float4> stateBuf_;

    uint64_t trailLenght_;
    uint64_t head_;
    int numSteps_;

    SolverLauncher kernel_;
};