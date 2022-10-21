//
// Created by durst on 10/21/22.
//

#include "bots/analysis/vis_geometry.h"

CellBits getCellsInFOV(const VisPoints & visPoints, const Vec3 & pos, const Vec2 & viewAngle) {
    Vec3 viewDir = angleVectors(viewAngle);

    Vec2 minViewAngle = viewAngle - (FOV / 2);
    Vec2 maxViewAngle = viewAngle + (FOV / 2);
    Vec3 maxViewDir = angleVectors(maxViewAngle);
    Vec3 minViewUnitDir = minViewDir / computeMagnitude(minViewDir);
    Vec3 maxViewUnitDir = maxViewDir / computeMagnitude(maxViewDir);

}
