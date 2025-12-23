#include "Renderer.h"
#include "Shader.h"
#include "ICGenerator/InitialConditions.h"
#include "Particles.h"

int main() {
    auto &renderer = Renderer::get();
    if (!renderer.init(800, 600, "Clean OpenGL")) return -1;

    Shader shader("shaders/basic.vert", "shaders/basic.frag");

    float3 start; 
    float3 end;

    PointBuffer iv = OneD::genLine(1000000, start, end);
    Particles part(std::move(iv), 64, SolverMethod::EULER);

    while (!renderer.shouldClose()) {
        renderer.beginFrame();
        shader.use();
        renderer.endFrame();
    }

    renderer.shutdown();
}


