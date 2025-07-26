#include <glad/gl.h>
#include "OdeSolver/methods.cuh"

class Particles{

    public:

    Particles(unsigned int numParticles, unsigned int lenTrajectories, SolverMethod method);
    ~Particles();
    void draw() const;

    private:

    unsigned int  numParticles_;
    unsigned int lenTrajectories;
    
};