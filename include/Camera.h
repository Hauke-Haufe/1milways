#include <glad/gl.h>
#include <glm/mat4x4.hpp>

class Camera{

public:
    Camera(glm::mat4 vp);

    void setViewPoint(glm::mat4 vp);
    void updateViewPoint(glm::mat4 trans);

    glm::mat4 getProjectionMatrix() const;

private:
    glm::mat4 viewPoint_;
};