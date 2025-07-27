#include <glad/gl.h>
#include "OdeSolver/Methods.hpp"

enum class IVPConifg{
    SPERE, 
    LINE, 
    PLANE
};

constexpr int MEMLAYOUT_SWITCH_VAL = 100000000; // Value for automatic memory Layout Switch 

class Particles{

    public:

    //Constructor for 3d ODE IVP Systems with predifined Intial Values
    Particles(unsigned int numParticles, unsigned int trailLenght, SolverMethod method, IVPConifg iv);
    //Constructor for 3d ODE IVP Systems with given Inital Values
    Particles(unsigned int numParticles, unsigned int trailLenght, SolverMethod method, float4* iv);
    ~Particles();

    void draw() const;
    void generate(double dt, int subSteps);


    private:

    void initIVP(IVPConifg confi, float4*);

    unsigned int  numParticles_;
    unsigned int trailLenght_;
    unsigned int head_;
    int numSteps_;

    GLuint pob_;
    cudaGraphicsResource *cupob_;
    float4* statebuf_;
    SolverLauncher kernel_;

};