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

namespace csknow::nearest_nav_cell {
    // 9 for same layer, then 1 above and 1 below
    constexpr size_t NUM_NEAREST_CELLS_PER_GRID_ENTRY = 11;
    constexpr double GRID_DIM_WIDTH_DEPTH = CELL_DIM_WIDTH_DEPTH;
    constexpr double GRID_DIM_HEIGHT = CELL_DIM_HEIGHT;

    typedef std::array<CellIdAndDistance, NUM_NEAREST_CELLS_PER_GRID_ENTRY> NearestGridData;

    class NearestNavCell : public QueryResult {
        void load(const string & mapsPath, const string & mapName);
    public:
        std::vector<AABB> gridEntryAABB;
        std::vector<NearestGridData> nearestCellsGrid;
        AABB playerPositionBounds;
        IVec3 gridDimensions;
        const VisPoints & visPoints;

        NearestNavCell(const VisPoints & visPoints) : visPoints(visPoints) {
            variableLength = false;
            nonTemporal = true;
            overlay = true;
            extension = ".near";
        }

        vector<int64_t> filterByForeignKey(int64_t) override {
            return {};
        }

        void oneLineToCSV(int64_t index, std::ostream & s) override {
            s << index << ","
                << gridEntryAABB[index].min.toCSV() << ","
                << gridEntryAABB[index].max.toCSV();
            for (size_t i = 0; i < NUM_NEAREST_CELLS_PER_GRID_ENTRY; i++) {
                s << "," << nearestCellsGrid[index][i].cellId;
                s << "," << nearestCellsGrid[index][i].distance;
            }
            s << std::endl;
        }

        vector<string> getForeignKeyNames() override {
            return {};
        }

        vector<string> getOtherColumnNames() override {
            vector<string> result = {"min x", "min y", "min z", "max x", "max y", "max z"};
            for (size_t i = 0; i < NUM_NEAREST_CELLS_PER_GRID_ENTRY; i++) {
                result.push_back(std::to_string(i) + " nearest cell id");
            }
            return result;
        }

        IVec3 posToGridIndex(Vec3 pos) const {
            // fit pos into bounds
            Vec3 thresholdedPos = min(max(pos, playerPositionBounds.min), playerPositionBounds.max);
            Vec3 deltaCellBounds = thresholdedPos - playerPositionBounds.min;
            return {
                static_cast<int64_t>(deltaCellBounds.x / GRID_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaCellBounds.y / GRID_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaCellBounds.z / GRID_DIM_HEIGHT)
            };
        };

        Vec3 gridIndexToCenterPos(IVec3 gridIndex) const {
            return playerPositionBounds.min + Vec3{
                (gridIndex.x + 0.5) * GRID_DIM_WIDTH_DEPTH,
                (gridIndex.y + 0.5) * GRID_DIM_WIDTH_DEPTH,
                (gridIndex.z + 0.5) * GRID_DIM_HEIGHT
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

        std::vector<CellIdAndDistance> getNearestCells(Vec3 pos) const {
            IVec3 curGridIndex = posToGridIndex(pos);
            const NearestGridData & nearestGridData = gridIndexToNearestCells(curGridIndex);
            // get the nearest grid index other than the cur one
            // take nearest in x/y with same z
            Vec3 curGridCenter = gridIndexToCenterPos(curGridIndex);
            IVec3 otherGridIndex = curGridIndex;
            otherGridIndex.x += pos.x >= curGridCenter.x ? 1 : -1;
            otherGridIndex.y += pos.y >= curGridCenter.y ? 1 : -1;
            const NearestGridData & otherGridData = gridIndexToNearestCells(otherGridIndex);

            std::set<CellId> resultSet;
            std::vector<CellIdAndDistance> result;
            for (const auto & gridData : {nearestGridData, otherGridData}) {
                for (const auto & nearestGridEntry : gridData) {
                    if (resultSet.find(nearestGridEntry.cellId) == resultSet.end()) {
                        resultSet.insert(nearestGridEntry.cellId);
                        result.push_back({
                                             nearestGridEntry.cellId,
                                             vis_point_helpers::get_point_to_aabb_distance(
                                                 pos, visPoints.getCellVisPoints()[nearestGridEntry.cellId].cellCoordinates)
                                         });
                    }
                }

            }

            std::sort(result.begin(), result.end(), [](const CellIdAndDistance & a, const CellIdAndDistance & b) {
                return a.distance < b.distance;
            });

            return result;
        }

        void runQuery(const string & mapsPath, const string & mapName);
    };
}

#endif //CSKNOW_NEAREST_NAV_CELL_H
