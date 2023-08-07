//
// Created by durst on 8/7/23.
//

#ifndef CSKNOW_CAMERA_FUNCTIONS_H
#define CSKNOW_CAMERA_FUNCTIONS_H

#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/gtc/quaternion.hpp>

static glm::mat4 makePerspectiveMatrix(float vfov, float aspect, float near) {
    float half_tan = tan(glm::radians(vfov) / 2.f);

    return glm::mat4(1.f / (aspect * half_tan), 0.f,                      0.f,                       0.f,
                     0.f,            1.f / half_tan,       0.f,                       0.f,
                     0.f,            0.f,                      0,                         -1.f,
                     0.f,            0.f,                      near,                      0.f);

}

static inline glm::mat4 makeViewMatrix(const glm::vec3 &position,
                                       const glm::vec3 &fwd,
                                       const glm::vec3 &up,
                                       const glm::vec3 &right) {
    glm::mat4 v(1.f);
    v[0][0] = right.x;
    v[1][0] = right.y;
    v[2][0] = right.z;
    v[0][1] = up.x;
    v[1][1] = up.y;
    v[2][1] = up.z;
    v[0][2] = -fwd.x;
    v[1][2] = -fwd.y;
    v[2][2] = -fwd.z;
    v[3][0] = -glm::dot(right, position);
    v[3][1] = -glm::dot(up, position);
    v[3][2] = glm::dot(fwd, position);

    return v;
}

#endif //CSKNOW_CAMERA_FUNCTIONS_H
