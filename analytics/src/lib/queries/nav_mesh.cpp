#include "queries/nav_mesh.h"

MapMeshResult queryMapMesh(nav_mesh::nav_file & navFile) {
    MapMeshResult result;
    int64_t i = 0;
    for (const auto & navArea : navFile.m_areas) {
        result.areaId.push_back(navArea.m_id);
        result.id.push_back(i);
        i++;
        if (navArea.m_place < navFile.m_place_count) {
            result.placeName.push_back(navFile.m_places[navArea.m_place]);
        }
        else {
            result.placeName.push_back("");
        }
        result.coordinate.push_back(AABB{
                {navArea.m_nw_corner.x, navArea.m_nw_corner.y, navArea.m_nw_corner.z},
                {navArea.m_se_corner.x, navArea.m_se_corner.y, navArea.m_se_corner.z}
        });
        vector<int64_t> connectionAreaIds;
        for (const auto & connection : navArea.m_connections) {
            connectionAreaIds.push_back(connection.id);
        }
        result.connectionAreaIds.push_back(connectionAreaIds);
    }
    result.size = navFile.m_area_count;
    return result;
}
