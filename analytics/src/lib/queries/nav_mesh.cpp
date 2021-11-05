#include "queries/nav_mesh.h"

MapMeshResult queryMapMesh(nav_mesh::nav_file & navFile) {
    MapMeshResult result;
    for (const auto & navArea : navFile.m_areas) {
        result.placeName.push_back(navFile.m_places[navArea.m_place]);
        result.coordinate.push_back(AABB{
                {navArea.m_nw_corner.x, navArea.m_nw_corner.y, navArea.m_nw_corner.z},
                {navArea.m_se_corner.x, navArea.m_se_corner.y, navArea.m_se_corner.z}
        });
    }
    return result;
}
