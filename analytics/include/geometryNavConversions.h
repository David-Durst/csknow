//
// Created by durst on 2/21/22.
//

#ifndef CSKNOW_GEOMETRYNAVCONVERSIONS_H
#define CSKNOW_GEOMETRYNAVCONVERSIONS_H
#include "navmesh/nav_file.h"
#include "geometry.h"

static nav_mesh::vec3_t vec3Conv(Vec3 vec) {
    return {static_cast<float>(vec.x), static_cast<float>(vec.y), static_cast<float>(vec.z)};
}

static Vec3 vec3tConv(nav_mesh::vec3_t vec) {
    return {static_cast<double>(vec.x), static_cast<double>(vec.y), static_cast<double>(vec.z)};
}

static AABB areaToAABB(const nav_mesh::nav_area & area) {
    return {vec3tConv(area.get_min_corner()), vec3tConv(area.get_max_corner())};
}

#endif //CSKNOW_GEOMETRYNAVCONVERSIONS_H
