#include "Renderer.h"
#include "Shader.h"

int main() {
    auto &renderer = Renderer::get();
    if (!renderer.init(800, 600, "Clean OpenGL")) return -1;

    Shader shader("shaders/basic.vert", "shaders/basic.frag");

    while (!renderer.shouldClose()) {
        renderer.beginFrame();
        shader.use();
        renderer.endFrame();
    }

    renderer.shutdown();
}


