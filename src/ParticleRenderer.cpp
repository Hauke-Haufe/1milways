#include "ParticleRenderer.h"
#include "PointBuffer/PointBuffer.h"

ParticleRenderer::ParticleRenderer(ParticleSimulation& ps, Shader shader)
    : sim_(ps)
    , numParticles_(ps.getNumPoints())
    , trailLength_(ps.getTrailLenght())
    , capacity_(numParticles_ * trailLength_)
    , trails_(capacity_, MemSpace::Unified)
    , glView_(trails_),
    lineShader_(shader)
{
    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        glView_.glBuffer()
    );
}

void ParticleRenderer::step(double dt, int subSteps){

    auto mapping = glView_.mapCuda();
    sim_.generate(
        mapping.data(), 
        dt,
        subSteps
    );
}

void ParticleRenderer::draw(const Camera& cam)
{
    lineShader_.use();

    lineShader_.setMat4("VP", cam.getProjectionMatrix());
    lineShader_.setUInt("trailLength", trailLength_);
    lineShader_.setUInt("numParticles", numParticles_);
    lineShader_.setUInt("head", sim_.head()); 

    glDrawArraysInstanced(
        GL_LINE_STRIP,
        0,
        static_cast<GLsizei>(trailLength_),
        static_cast<GLsizei>(numParticles_)
    );

}
