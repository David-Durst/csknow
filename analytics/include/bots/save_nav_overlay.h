//
// Created by steam on 7/20/22.
//

#ifndef CSKNOW_SAVE_NAV_OVERLAY_H
#define CSKNOW_SAVE_NAV_OVERLAY_H

#include "bots/load_save_vis_points.h"
#define MAX_OVERLAYS 4
#define FADE_SECONDS 0.1
#define MAX_DISTANCE 2500

struct NavAreaData {
    AreaId areaId;
    AABB areaCoordinates;
    Vec3 center;
};

static const array<string, MAX_OVERLAYS> colorScheme{"b", "r", "g", "w"};

class NavFileOverlay {
    const nav_mesh::nav_file & navFile;
    vector<NavAreaData> navAreaData;
    map<AreaId, size_t> areaIdToVectorIndex;
    CSKnowTime lastCallTime = std::chrono::system_clock::from_time_t(0);
    CSKnowTime lastCallTime2 = std::chrono::system_clock::time_point::max();
    string mapsPath;

    void saveOverlays(std::stringstream & stream, Vec3 specPos, const vector<AreaBits> & overlays);

public:
    NavFileOverlay(const nav_mesh::nav_file & navFile) : navFile(navFile) {
        for (const auto & navArea : navFile.m_areas) {
            navAreaData.push_back({navArea.get_id(), {vec3tConv(navArea.get_min_corner()),
                                                      vec3tConv(navArea.get_max_corner())}, vec3tConv(navArea.get_center())});
        }

        areaIdToVectorIndex = navFile.m_area_ids_to_indices;
    }

    void setMapsPath(string mapsPath) { this->mapsPath = mapsPath; }
    void save(const ServerState & state, const vector<AreaBits> & overlays);
};

#endif //CSKNOW_SAVE_NAV_OVERLAY_H
