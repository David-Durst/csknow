//
// Created by durst on 10/21/22.
//

#include "bots/analysis/vis_geometry.h"
#include "bots/analysis/camera_fov.h"

bool getPointInFOV(const Vec3 & targetPos, const Vec3 & sourcePos, const Vec2 & viewAngle) {
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
    glm::mat4 View = makeViewMatrix(sourcePos.toGLM(),
                                    rotation * glm::vec3(1, 0, 0),
                                    rotation * glm::vec3(0, 0, 1),
                                    rotation * glm::vec3(0, -1, 0));
    glm::mat4 projMat = Projection * View;

    glm::vec3 glmCellPos = targetPos.toGLM();
    glm::vec4 homogenousCellPos{glmCellPos.x, glmCellPos.y, glmCellPos.z, 1};
    glm::vec4 cellPosScreenSpace = projMat * homogenousCellPos;
    glm::vec3 projCellPosScreenSpace = glm::vec3(cellPosScreenSpace) / cellPosScreenSpace.w;
    return (projCellPosScreenSpace.x >= -1.f && projCellPosScreenSpace.x <= 1.f &&
            projCellPosScreenSpace.y >= -1.f && projCellPosScreenSpace.y <= 1.f &&
            projCellPosScreenSpace.z >= 0.f);
}