//
// Created by durst on 8/7/23.
//

#ifndef CSKNOW_VIS_CELL_GEOMETRY_H
#define CSKNOW_VIS_CELL_GEOMETRY_H

#include "bots/analysis/load_save_vis_points.h"
#include "bots/analysis/vis_geometry.h"

CellBits getCellsInFOV(const VisPoints & visPoints, const Vec3 & pos, const Vec2 & viewAngle);
bool getCellsInFOV(const vector<CellVisPoint> & cellVisPoints, const Vec3 & pos, const Vec2 & viewAngle);

#endif //CSKNOW_VIS_CELL_GEOMETRY_H
