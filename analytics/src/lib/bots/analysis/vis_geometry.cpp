//
// Created by durst on 10/21/22.
//

#include "bots/analysis/vis_geometry.h"
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/gtc/quaternion.hpp>

static glm::mat4 makePerspectiveMatrix(float hfov, float aspect, float near) {
    float half_tan = tan(glm::radians(hfov) / 2.f);

    return glm::mat4(1.f / half_tan, 0.f,                      0.f,                       0.f, 
                     0.f,            -aspect / half_tan,       0.f,                       0.f,
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
    glm::mat4 Projection = makePerspectiveMatrix(horizontalFOV, aspectRatio, 0.001f);
    // Camera matrix
    // quaternion = way to represent a rotation
    glm::quat rotation =
        glm::angleAxis(glm::radians(static_cast<float>(viewAngle.x)), glm::vec3{0.f, 0.f, 1.f}) *
        glm::angleAxis(glm::radians(static_cast<float>(viewAngle.y)), glm::vec3{0.f, 1.f, 0.f});
    glm::vec3 up = rotation * glm::vec3(0, 0, 1);
    glm::mat4 View = makeViewMatrix(pos.toGLM(),
                                    quat * glm::vec3(1, 0, 0),
                                    quat * glm::vec3(0, 0, 1),
                                    quat * glm::vec3(0, -1, 0));
    glm::vec3 forwardMine = angleVectors(viewAngle).toGLM();
    glm::vec3 forwardBrennan = quat * glm::vec3(1, 0, 0);
                                        /*
    glm::mat4 View = glm::lookAt(
        pos.toGLM(), // Camera is at (4,3,3), in World Space
        (pos + angleVectors(viewAngle)).toGLM(), // and looks at the origin
        up  // Head is up (set to 0,-1,0 to look upside-down)
    );
                                         */
    glm::mat4 projMat = Projection * View;
    //16659->16673

    for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
        glm::vec3 glmCellPos = cellVisPoint.topCenter.toGLM();
        glm::vec4 homogenousCellPos{glmCellPos.x, glmCellPos.y, glmCellPos.z, 1};
        glm::vec4 cellPosScreenSpace = projMat * homogenousCellPos;
        glm::vec3 projCellPosScreenSpace = glm::vec3(cellPosScreenSpace) / cellPosScreenSpace.w;
        glm::vec4 justTranslate = View * homogenousCellPos;
        if (cellVisPoint.cellId == 16117 || cellVisPoint.cellId == 16149) {//16673) {
            int x = 1;
            (void) x;
        }
        if (projCellPosScreenSpace.x >= -1.f && projCellPosScreenSpace.x <= 1.f &&
            projCellPosScreenSpace.y >= -1.f && projCellPosScreenSpace.y <= 1.f &&
            projCellPosScreenSpace.z >= 0.f) {
            result.set(cellVisPoint.cellId, true);
        }
    }

    return result;
}
