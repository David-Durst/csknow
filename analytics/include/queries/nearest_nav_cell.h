//
// Created by durst on 2/20/23.
//

#ifndef CSKNOW_NEAREST_NAV_CELL_H
#define CSKNOW_NEAREST_NAV_CELL_H
#include "queries/distance_to_places.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

namespace csknow::nearest_nav_area {
    constexpr size_t numNearestCellsPerGridEntry = 2;

    typedef std::array<CellIdAndDistance, numNearestCellsPerGridEntry> NearestGridData;

    class NearestNavCell : public QueryResult {
        void load(const string & mapsPath, const string & mapName);
    public:
        std::vector<AABB> gridEntryAABB;
        std::vector<NearestGridData> nearestCellsGrid;
        AABB playerPositionBounds;
        IVec3 gridDimensions;

        NearestNavCell() {
            variableLength = false;
            nonTemporal = true;
            overlay = true;
        }

        vector<int64_t> filterByForeignKey(int64_t) override {
            return {};
        }

        void oneLineToCSV(int64_t index, std::ostream & s) override {
            s << index << ","
                << gridEntryAABB[index].min.toCSV() << ","
                << gridEntryAABB[index].max.toCSV() << ","
                << nearestCellsGrid[index][0].cellId << "," << nearestCellsGrid[index][0].distance
                << nearestCellsGrid[index][1].cellId << "," << nearestCellsGrid[index][1].distance;
        }

        vector<string> getForeignKeyNames() override {
            return {};
        }

        vector<string> getOtherColumnNames() override {
            return {"min x", "min y", "min z", "max x", "max y", "max z", "nearest cell id", "nearest cell distance",
                    "second nearest cell id", "second nearest cell distance"};
        }

        IVec3 posToGridIndex(Vec3 pos) {
            // fit pos into bounds (in case using this on data set not created from)
            Vec3 thresholdedPos = min(max(pos, playerPositionBounds.min), playerPositionBounds.max);
            Vec3 deltaCellBounds = thresholdedPos - playerPositionBounds.min;
            return {
                static_cast<int64_t>(deltaCellBounds.x / CELL_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaCellBounds.y / CELL_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaCellBounds.z / CELL_DIM_HEIGHT)
            };
        };

        Vec3 gridIndexToPos(IVec3 gridIndex) {
            return playerPositionBounds.min + Vec3{
                gridIndex.x * CELL_DIM_WIDTH_DEPTH,
                gridIndex.y * CELL_DIM_WIDTH_DEPTH,
                gridIndex.z * CELL_DIM_HEIGHT,
            };
        }

        AABB & gridIndexToAABB(IVec3 gridIndex) {
            return gridEntryAABB[gridIndex.x * gridDimensions.y * gridDimensions.z +
                                 gridIndex.y * gridDimensions.z + gridIndex.z];
        }

        const AABB & gridIndexToAABB(IVec3 gridIndex) const {
            return gridEntryAABB[gridIndex.x * gridDimensions.y * gridDimensions.z +
                                 gridIndex.y * gridDimensions.z + gridIndex.z];
        }

        NearestGridData & gridIndexToNearestCells(IVec3 gridIndex) {
            return nearestCellsGrid[gridIndex.x * gridDimensions.y * gridDimensions.z +
                                    gridIndex.y * gridDimensions.z + gridIndex.z];
        }

        const NearestGridData & gridIndexToNearestCells(IVec3 gridIndex) const {
            return nearestCellsGrid[gridIndex.x * gridDimensions.y * gridDimensions.z +
                                    gridIndex.y * gridDimensions.z + gridIndex.z];
        }

        void runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                      const VisPoints & visPoints, const string & mapsPath, const string & mapName);
    };
}

#endif //CSKNOW_NEAREST_NAV_CELL_H
