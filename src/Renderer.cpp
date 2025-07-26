#include "Renderer.h"
#include <iostream>

Renderer& Renderer::get() {
    static Renderer instance;
    return instance;
}

bool Renderer::init(int width, int height, const char* title) {
    if (!glfwInit()) return false;
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(window);
    if (gladLoadGL(glfwGetProcAddress) == 0) { 
        std::cerr << "Failed to load OpenGL\n"; 
        return false;
    }
    return true;
}

void Renderer::beginFrame() { glClear(GL_COLOR_BUFFER_BIT); }
void Renderer::endFrame() { glfwSwapBuffers(window); glfwPollEvents(); }
bool Renderer::shouldClose() { return glfwWindowShouldClose(window); }
void Renderer::shutdown() { glfwDestroyWindow(window); glfwTerminate(); }