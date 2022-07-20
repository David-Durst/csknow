//
// Created by steam on 7/20/22.
//

#ifndef CSKNOW_SAVE_NAV_OVERLAY_H
#define CSKNOW_SAVE_NAV_OVERLAY_H

#include "bots/load_save_vis_points.h"
#define OVERLAY_VERTICAL_BASE 5.
#define OVERLAY_VERTICAL_OFFSET 20.
#define OVERLAY_VERTICAL_LENGTH 5.
#define MAX_PLAYERS_PER_OVERLAY 4
#define MAX_OVERLAYS 5

struct NavAreaData {
    AreaId areaId;
    AABB areaCoordinates;
    Vec3 center;
};

static const array<string, MAX_PLAYERS_PER_OVERLAY+1> colorScheme{"b", "r", "u", "g", "w"}; // mtg color abbreviations: black, red, blue, green, white

struct PlayerNavAreaOverlay {
    map<CSGOId, AreaBits> playerToOverlay;
    void saveOverlay(std::ofstream & stream, size_t playerIndex, size_t overlayIndex,
                     size_t numOverlays, const vector<NavAreaData> & navAreaData);
};

class NavFileOverlay {
    vector<NavAreaData> navAreaData;
    vector<PlayerNavAreaOverlay> playerNavAreaOverlay;
    map<AreaId, size_t> areaIdToVectorIndex;

public:
    NavFileOverlay(const nav_mesh::nav_file & navFile) {
        for (const auto & navArea : navFile.m_areas) {
            navAreaData.push_back({navArea.get_id(), {vec3tConv(navArea.get_min_corner()),
                                                      vec3tConv(navArea.get_max_corner())}, vec3tConv(navArea.get_center())});
        }

        areaIdToVectorIndex = navFile.m_area_ids_to_indices;
    }

    void addOverlay();
    void update(const map<CSGOId, AreaBits> & state, size_t overlayIndex);
    void save(string mapsPath, string mapName);
};

#endif //CSKNOW_SAVE_NAV_OVERLAY_H
