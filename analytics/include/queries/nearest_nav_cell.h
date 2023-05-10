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

    struct AreaIdAndDistance {
        CellId areaId;
        double distance;
    };

    class NearestNavCell : public QueryResult {
        void load(const string & mapsPath, const string & mapName);
    public:
        std::vector<AABB> gridEntryAABB;
        std::vector<NearestGridData> nearestCellsGrid;
        AABB areaBounds;
        IVec3 gridDimensions;
        const VisPoints & visPoints;

        explicit NearestNavCell(const VisPoints & visPoints) : visPoints(visPoints) {
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
            Vec3 thresholdedPos = min(max(pos, areaBounds.min), areaBounds.max);
            Vec3 deltaCellBounds = thresholdedPos - areaBounds.min;
            return {
                static_cast<int64_t>(deltaCellBounds.x / GRID_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaCellBounds.y / GRID_DIM_WIDTH_DEPTH),
                static_cast<int64_t>(deltaCellBounds.z / GRID_DIM_HEIGHT)
            };
        };

        Vec3 gridIndexToCenterPos(IVec3 gridIndex) const {
            return areaBounds.min + Vec3{
                (gridIndex.x + 0.5) * GRID_DIM_WIDTH_DEPTH,
                (gridIndex.y + 0.5) * GRID_DIM_WIDTH_DEPTH,
                (gridIndex.z + 0.5) * GRID_DIM_HEIGHT
            };
        }

        Vec3 gridIndexToMinPos(IVec3 gridIndex) const {
            return areaBounds.min + Vec3{
                gridIndex.x * GRID_DIM_WIDTH_DEPTH,
                gridIndex.y * GRID_DIM_WIDTH_DEPTH,
                gridIndex.z * GRID_DIM_HEIGHT
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

        // true if distance metric is cells, false if distance metric is areas
        // grid is fully dense, cells are regular but missing in areas where no nav area
        std::vector<CellIdAndDistance> getNearestCells(Vec3 pos) const;
        // NEED TO FIX, 1807 DOESN'T APPEAR AS TOO SMALL FOR CELL, SO CAN"T GET IT
        // NEED A BETTER FIX
        AreaId getNearestArea(Vec3 pos) const;

        void runQuery(const string & mapsPath, const string & mapName);
    };
}

#endif //CSKNOW_NEAREST_NAV_CELL_H
