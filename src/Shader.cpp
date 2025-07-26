#include "Shader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/gtc/type_ptr.hpp> // optional for setUniform mat4

std::string Shader::readFile(const std::string &path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

unsigned int Shader::compileShader(unsigned int type, const std::string &source) {
    unsigned int shader = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, 512, nullptr, info);
        std::cerr << "Shader compile error: " << info << "\n";
    }
    return shader;
}

Shader::Shader(const std::string &vertexPath, const std::string &fragmentPath) {
    std::string vCode = readFile(vertexPath);
    std::string fCode = readFile(fragmentPath);
    unsigned int vShader = compileShader(GL_VERTEX_SHADER, vCode);
    unsigned int fShader = compileShader(GL_FRAGMENT_SHADER, fCode);
    ID = glCreateProgram();
    glAttachShader(ID, vShader);
    glAttachShader(ID, fShader);
    glLinkProgram(ID);
    glDeleteShader(vShader);
    glDeleteShader(fShader);
}

void Shader::use() const { glUseProgram(ID); }

void Shader::setUniform(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::setUniform(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

