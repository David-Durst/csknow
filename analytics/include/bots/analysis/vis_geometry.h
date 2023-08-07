//
// Created by durst on 10/21/22.
//

#ifndef CSKNOW_VIS_GEOMETRY_H
#define CSKNOW_VIS_GEOMETRY_H

#include "geometry.h"

constexpr float aspectRatio = 16. / 9.;
constexpr float verticalFOV = 74.;

bool getPointInFOV(const Vec3 & targetPos, const Vec3 & sourcePos, const Vec2 & viewAngle);

#endif //CSKNOW_VIS_GEOMETRY_H
