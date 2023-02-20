//
// Created by durst on 2/20/23.
//

#ifndef CSKNOW_NEAREST_NAV_AREA_H
#define CSKNOW_NEAREST_NAV_AREA_H
#include "queries/distance_to_places.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

namespace csknow::nearest_nav_area {
    constexpr size_t numNearestAreasPerGridCell = 2;

    typedef std::array<nav_mesh::AreaDistance, numNearestAreasPerGridCell> NearestGridData;

    class NearestNavArea : public QueryResult {
    public:
        std::vector<NearestGridData> nearestAreasGrid;
        AABB playerPositionBounds;
        IVec3 gridDimensions;

        NearestNavArea() {
            variableLength = false;
            nonTemporal = true;
            overlay = true;
        }

        vector<int64_t> filterByForeignKey(int64_t) override {
            return {};
        }

        void oneLineToCSV(int64_t, std::ostream &) override { }

        vector<string> getForeignKeyNames() override {
            return {};
        }

        vector<string> getOtherColumnNames() override {
            return {};
        }

        IVec3 posToGridIndex(Vec3 pos) {
            Vec3 deltaAreaBounds = pos - playerPositionBounds.min;
            return {
                static_cast<int64_t>(deltaAreaBounds.x / CELL_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaAreaBounds.y / CELL_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaAreaBounds.z / CELL_DIM_HEIGHT)
            };
        };

        Vec3 gridIndexToPos(IVec3 gridIndex) {
            return playerPositionBounds.min + Vec3{
                gridIndex.x * CELL_DIM_WIDTH_DEPTH,
                gridIndex.y * CELL_DIM_WIDTH_DEPTH,
                gridIndex.z * CELL_DIM_HEIGHT,
            };
        }

        NearestGridData & gridIndexToNearestAreas(IVec3 gridIndex) {
            return nearestAreasGrid[gridIndex.x * gridDimensions.y * gridDimensions.z +
                                    gridIndex.y * gridDimensions.z + gridIndex.z];
        }

        const NearestGridData & gridIndexToNearestAreas(IVec3 gridIndex) const {
            return nearestAreasGrid[gridIndex.x * gridDimensions.y * gridDimensions.z +
                                    gridIndex.y * gridDimensions.z + gridIndex.z];
        }

        void runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                      const nav_mesh::nav_file & navFile);
    };
}

#endif //CSKNOW_NEAREST_NAV_AREA_H
