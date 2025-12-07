#include <glad/gl.h>
#include "OdeSolver/Methods.hpp"
#include "PointBuffer/PointBuffer.h"

enum class IVPConifg{
    SPERE, 
    LINE, 
    PLANE
};

constexpr int MEMLAYOUT_SWITCH_VAL = 100000000; // Value for automatic memory Layout Switch 

class Particles{

    public:

    Particles(PointBuffer points, unsigned int trailLenght, SolverMethod methods);

    ~Particles();

    void draw() const;
    void generate(double dt, int subSteps);

    private:

    PointBuffer stateBuf_;

    unsigned int trailLenght_;
    unsigned int head_;
    int numSteps_;

    GLuint pob_;
    cudaGraphicsResource *cupob_;
    SolverLauncher kernel_;
};