//
// Created by durst on 10/21/22.
//

#include "bots/analysis/vis_geometry.h"
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

CellBits getCellsInFOV(const VisPoints & visPoints, const Vec3 & pos, const Vec2 & viewAngle) {
    CellBits result;
    // https://github.com/perilouswithadollarsign/cstrike15_src/blob/f82112a2388b841d72cb62ca48ab1846dfcc11c8/mathlib/camera.cpp
    // https://old.reddit.com/r/GlobalOffensive/comments/e4q8sd/whats_the_default_csgo_fov_horizontal_or_vertical/
    // http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
    // https://thxforthefish.com/posts/reverse_z/

    glm::mat4 Projection = makePerspectiveMatrix(verticalFOV, aspectRatio, 0.001f);
    // Camera matrix
    // quaternion = way to represent a rotation
    glm::quat rotation =
        glm::angleAxis(glm::radians(static_cast<float>(viewAngle.x)), glm::vec3{0.f, 0.f, 1.f}) *
        glm::angleAxis(glm::radians(static_cast<float>(viewAngle.y)), glm::vec3{0.f, 1.f, 0.f});
    glm::mat4 View = makeViewMatrix(pos.toGLM(),
                                    rotation * glm::vec3(1, 0, 0),
                                    rotation * glm::vec3(0, 0, 1),
                                    rotation * glm::vec3(0, -1, 0));
    glm::mat4 projMat = Projection * View;

    for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
        glm::vec3 glmCellPos = cellVisPoint.topCenter.toGLM();
        glm::vec4 homogenousCellPos{glmCellPos.x, glmCellPos.y, glmCellPos.z, 1};
        glm::vec4 cellPosScreenSpace = projMat * homogenousCellPos;
        glm::vec3 projCellPosScreenSpace = glm::vec3(cellPosScreenSpace) / cellPosScreenSpace.w;
        if (projCellPosScreenSpace.x >= -1.f && projCellPosScreenSpace.x <= 1.f &&
            projCellPosScreenSpace.y >= -1.f && projCellPosScreenSpace.y <= 1.f &&
            projCellPosScreenSpace.z >= 0.f) {
            result.set(cellVisPoint.cellId, true);
        }
    }

    return result;
}

bool getCellsInFOV(const vector<CellVisPoint> & cellVisPoints, const Vec3 & pos, const Vec2 & viewAngle) {
    // https://github.com/perilouswithadollarsign/cstrike15_src/blob/f82112a2388b841d72cb62ca48ab1846dfcc11c8/mathlib/camera.cpp
    // https://old.reddit.com/r/GlobalOffensive/comments/e4q8sd/whats_the_default_csgo_fov_horizontal_or_vertical/
    // http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
    // https://thxforthefish.com/posts/reverse_z/

    glm::mat4 Projection = makePerspectiveMatrix(verticalFOV, aspectRatio, 0.001f);
    // Camera matrix
    // quaternion = way to represent a rotation
    glm::quat rotation =
        glm::angleAxis(glm::radians(static_cast<float>(viewAngle.x)), glm::vec3{0.f, 0.f, 1.f}) *
        glm::angleAxis(glm::radians(static_cast<float>(viewAngle.y)), glm::vec3{0.f, 1.f, 0.f});
    glm::mat4 View = makeViewMatrix(pos.toGLM(),
                                    rotation * glm::vec3(1, 0, 0),
                                    rotation * glm::vec3(0, 0, 1),
                                    rotation * glm::vec3(0, -1, 0));
    glm::mat4 projMat = Projection * View;

    bool result = true;
    for (const auto & cellVisPoint : cellVisPoints) {
        glm::vec3 glmCellPos = cellVisPoint.topCenter.toGLM();
        glm::vec4 homogenousCellPos{glmCellPos.x, glmCellPos.y, glmCellPos.z, 1};
        glm::vec4 cellPosScreenSpace = projMat * homogenousCellPos;
        glm::vec3 projCellPosScreenSpace = glm::vec3(cellPosScreenSpace) / cellPosScreenSpace.w;
        result &= (projCellPosScreenSpace.x >= -1.f && projCellPosScreenSpace.x <= 1.f &&
            projCellPosScreenSpace.y >= -1.f && projCellPosScreenSpace.y <= 1.f &&
            projCellPosScreenSpace.z >= 0.f);
    }
    return result;
}
