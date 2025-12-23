#include "ParticleSimulation.h"
#include "Shader.h"
#include "Camera.h"

class ParticleRenderer {
public:
    explicit ParticleRenderer(ParticleSimulation& ps, Shader shader);

    void step(double dt, int subSteps);
    void draw(const Camera& cam);

private:
    ParticleSimulation& sim_;

    size_t numParticles_;
    size_t trailLength_;
    size_t capacity_;

    PointBuffer<half4> trails_;
    GLPointBufferView<half4> glView_;

    Shader lineShader_;   // you already have this
};