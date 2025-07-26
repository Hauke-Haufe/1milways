#pragma once
#include <string>
#include <glad/gl.h>

class Shader {
public:
    unsigned int ID;
    Shader(const std::string &vertexPath, const std::string &fragmentPath);
    void use() const;
    void setUniform(const std::string &name, float value) const;
    void setUniform(const std::string &name, int value) const;

private:
    std::string readFile(const std::string &path);
    unsigned int compileShader(unsigned int type, const std::string &source);
};