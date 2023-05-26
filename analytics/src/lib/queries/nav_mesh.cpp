#include "queries/nav_mesh.h"

MapMeshResult::OverlappingResult MapMeshResult::overlappingAreas(Vec3 pos) const {
    OverlappingResult result;
    for (size_t i = 0; i < coordinate.size(); i++) {
        if (pointInRegion(coordinate[i], pos)) {
            result.overlappingIn3D.push_back(areaId[i]);
        }
        if (pointInRegion2D(coordinate[i], pos)) {
            result.overlappingIn2D.push_back(areaId[i]);
        }
    }
    return result;
}

MapMeshResult queryMapMesh(nav_mesh::nav_file & navFile, const string & queryName) {
    MapMeshResult result(queryName);
    int64_t i = 0;
    for (const auto & navArea : navFile.m_areas) {
        result.areaId.push_back(navArea.m_id);
        result.id.push_back(i);
        result.areaToInternalId.insert({navArea.m_id, i});
        i++;
        if (navArea.m_place < navFile.m_place_count) {
            result.placeName.push_back(navFile.get_place(navArea.m_place));
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

