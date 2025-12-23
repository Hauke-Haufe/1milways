#pragma once
#include <string>
#include <glad/gl.h>
#include <glm/mat4x4.hpp> 

class Shader {
public:
    unsigned int ID;
    Shader(const std::string &vertexPath, const std::string &fragmentPath);
    void use() const;

    void setUInt(const std::string &name, uint16_t value) const;
    void setFloat(const std::string &name, float value) const;
    void setMat4(const std::string &name, glm::mat4) const;

private:
    std::string readFile(const std::string &path);
    unsigned int compileShader(unsigned int type, const std::string &source);
};