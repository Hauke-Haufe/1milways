#pragma once
#include <glad/gl.h>
#include <GLFW/glfw3.h>

class Renderer {
public:
    static Renderer& get();
    bool init(int width, int height, const char* title);
    void beginFrame();
    void endFrame();
    bool shouldClose();
    void shutdown();
    
private:
    Renderer() = default;
    GLFWwindow* window = nullptr;
};