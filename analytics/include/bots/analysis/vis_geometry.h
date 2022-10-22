//
// Created by durst on 10/21/22.
//

#ifndef CSKNOW_VIS_GEOMETRY_H
#define CSKNOW_VIS_GEOMETRY_H

#include "bots/analysis/load_save_vis_points.h"
#include "geometry.h"

constexpr float aspectRatio = 16. / 9.;
constexpr float horizontalFOV = 104.;
constexpr float verticalFOV = 74.;

CellBits getCellsInFOV(const VisPoints & visPoints, const Vec3 & pos, const Vec2 & viewAngle);

#endif //CSKNOW_VIS_GEOMETRY_H
