//
// Created by durst on 2/21/22.
//

#ifndef CSKNOW_GEOMETRYNAVCONVERSIONS_H
#define CSKNOW_GEOMETRYNAVCONVERSIONS_H
#include "navmesh/nav_file.h"
#include "geometry.h"

typedef uint32_t AreaId;
typedef uint32_t CellId;
typedef uint16_t PlaceIndex;

static nav_mesh::vec3_t vec3Conv(Vec3 vec) {
    return {static_cast<float>(vec.x), static_cast<float>(vec.y), static_cast<float>(vec.z)};
}

static Vec3 vec3tConv(nav_mesh::vec3_t vec) {
    return {static_cast<double>(vec.x), static_cast<double>(vec.y), static_cast<double>(vec.z)};
}

static AABB areaToAABB(const nav_mesh::nav_area & area) {
    return {vec3tConv(area.get_min_corner()), vec3tConv(area.get_max_corner())};
}

static inline __attribute__((always_inline))
double computeDistance(AreaId a1, AreaId a2, const nav_mesh::nav_file & navFile) {
    Vec3 v1 = vec3tConv(navFile.get_area_by_id_fast(a1).get_center()),
        v2 = vec3tConv(navFile.get_area_by_id_fast(a2).get_center());
    double xDistance = v1.x - v2.x;
    double yDistance = v1.y - v2.y;
    double zDistance = v1.z - v2.z;
    return sqrt(xDistance * xDistance + yDistance * yDistance + zDistance * zDistance);
}

#endif //CSKNOW_GEOMETRYNAVCONVERSIONS_H
