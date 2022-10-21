//
// Created by durst on 10/21/22.
//

#ifndef CSKNOW_VIS_GEOMETRY_H
#define CSKNOW_VIS_GEOMETRY_H

#include "bots/analysis/load_save_vis_points.h"
#include "geometry.h"

constexpr double horizontalFOV = 90.;
constexpr double verticalFOV = horizontalFOV * 9. / 16.;
constexpr Vec2 FOV{horizontalFOV, verticalFOV};

CellBits getCellsInFOV(const VisPoints & visPoints, const Vec3 & pos, const Vec2 & viewAngle);

#endif //CSKNOW_VIS_GEOMETRY_H
