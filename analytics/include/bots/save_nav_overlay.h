//
// Created by steam on 7/20/22.
//

#ifndef CSKNOW_SAVE_NAV_OVERLAY_H
#define CSKNOW_SAVE_NAV_OVERLAY_H

#include "bots/load_save_vis_points.h"
#define OVERLAY_VERTICAL_BASE 5.
#define OVERLAY_VERTICAL_LENGTH 5.
#define MAX_PLAYERS_PER_OVERLAY 4
#define MAX_OVERLAYS 2
#define FADE_SECONDS 0.1

struct NavAreaData {
    AreaId areaId;
    AABB areaCoordinates;
    Vec3 center;
};

static const array<string, 2*MAX_PLAYERS_PER_OVERLAY> colorScheme{"b", "r", "u", "g", "w", "y", "p", "c"};

class NavFileOverlay {
    vector<NavAreaData> navAreaData;
    map<AreaId, size_t> areaIdToVectorIndex;
    CSKnowTime lastCallTime = std::chrono::system_clock::from_time_t(0);
    CSKnowTime lastCallTime2 = std::chrono::system_clock::time_point::max();
    string mapsPath;

    void saveOverlay(std::stringstream & stream, size_t overlayIndex, size_t numOverlays,
                     const map<CSGOId, AreaBits> & playerToOverlay);

public:
    NavFileOverlay(const nav_mesh::nav_file & navFile) {
        for (const auto & navArea : navFile.m_areas) {
            navAreaData.push_back({navArea.get_id(), {vec3tConv(navArea.get_min_corner()),
                                                      vec3tConv(navArea.get_max_corner())}, vec3tConv(navArea.get_center())});
        }

        areaIdToVectorIndex = navFile.m_area_ids_to_indices;
    }

    void setMapsPath(string mapsPath) { this->mapsPath = mapsPath; }
    void save(const ServerState & state, const vector<map<CSGOId, AreaBits>> & overlays);
};

#endif //CSKNOW_SAVE_NAV_OVERLAY_H
