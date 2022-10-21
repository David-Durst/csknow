//
// Created by durst on 10/21/22.
//

#include "bots/analysis/vis_geometry.h"
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi

CellBits getCellsInFOV(const VisPoints & visPoints, const Vec3 & pos, const Vec2 & viewAngle) {
    CellBits result;
    glm::mat4 Projection = glm::perspective(verticalFOV, aspectRatio, 0.1f, 100000.f);
    glm::mat4 View = glm::translate(glm::mat4(1.0f), pos.toGLM());
    View = glm::rotate(View, viewAngle.toGLM().y, glm::vec3(-1.0f, 0.0f, 0.0f));
    View = glm::rotate(View, viewAngle.toGLM().x, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projMat = Projection * View;

    for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
        glm::vec3 glmCellPos = cellVisPoint.topCenter.toGLM();
        glm::vec4 homogenousCellPos{glmCellPos.x, glmCellPos.y, glmCellPos.z, 1};
        glm::vec4 cellPosScreenSpace = projMat * homogenousCellPos;
        glm::vec3 projCellPosScreenSpace = glm::vec3(cellPosScreenSpace) / cellPosScreenSpace.w;
        if (projCellPosScreenSpace.x >= -1.f && projCellPosScreenSpace.x <= 1.f &&
            projCellPosScreenSpace.y >= -1.f && projCellPosScreenSpace.y <= 1.f &&
            projCellPosScreenSpace.z > 0) {
            result.set(cellVisPoint.cellId, true);
        }
    }

    return result;
}
