//
// Created by durst on 12/29/21.
//

#ifndef CSKNOW_LOAD_COVER_H
#define CSKNOW_LOAD_COVER_H

#include "load_data.h"
#include "geometry.h"

struct GridIndex {
    vector<int64_t> sortedIds;
    vector<int64_t> minIdIndex, numIds;
    Vec3 cellSizes;
    IVec3 numCells;
    Vec3 minValues;
    Vec3 maxValues;
    Vec3 * values;

    IVec3 getCellCoordinates(Vec3 v) const {
        return {
            (int64_t) std::floor((v.x - minValues.x) / cellSizes.x),
            (int64_t) std::floor((v.y - minValues.y) / cellSizes.y),
            (int64_t) std::floor((v.z - minValues.z) / cellSizes.z)
        };
    }

    int64_t getCellIndex(IVec3 v) const {
        return v.x * numCells.y * numCells.z + v.y * numCells.z + v.z;
    }

    int64_t getNearest(Vec3 * origins, Vec3 curPosition, Vec3 & result) const {
        int64_t resultIndex = -1;
        double minDistance = -1;
        IVec3 centerCoord = getCellCoordinates(curPosition);
        IVec3 minCoord = max({0,0,0}, centerCoord - 1);
        IVec3 maxCoord = min(numCells - 1, centerCoord + 1);
        for (int64_t xCoord = minCoord.x; xCoord <= maxCoord.x; xCoord++) {
            for (int64_t yCoord = minCoord.y; yCoord <= maxCoord.y; yCoord++) {
                for (int64_t zCoord = minCoord.z; zCoord <= maxCoord.z; zCoord++) {
                    int64_t curCoord = getCellIndex({xCoord, yCoord, zCoord});
                    for (int64_t idIndex = minIdIndex[curCoord];
                        idIndex < minIdIndex[curCoord] + numIds[curCoord];
                        idIndex++) {
                        double newDistance = computeDistance(curPosition, origins[sortedIds[idIndex]]);
                        if (resultIndex == -1 || newDistance < minDistance) {
                            minDistance = newDistance;
                            resultIndex = sortedIds[idIndex];
                            result = origins[resultIndex];
                        }
                    }
                }
            }
        }
        return resultIndex;
    }
};

struct GridComparator {
    GridIndex & gridIndex;

    GridComparator(GridIndex & gridIndex) : gridIndex(gridIndex) {}

    bool operator() (const int64_t& lhs_index, const int64_t& rhs_index) const
    {
        Vec3 lhs = gridIndex.values[lhs_index];
        Vec3 rhs = gridIndex.values[rhs_index];
        return gridIndex.getCellIndex(gridIndex.getCellCoordinates(lhs)) <
               gridIndex.getCellIndex(gridIndex.getCellCoordinates(rhs));
    }
};

typedef AABB * AABBIndex;

class CoverOrigins : public ColStore {
public:
    Vec3 * origins;
    GridIndex originsGrid;
    RangeIndex coverEdgesPerOrigin;
    AABBIndex coverEdgeBoundsPerOrigin;

    void init(int64_t rows, int64_t numFiles) {
        ColStore::init(rows, numFiles, {});
        origins = (Vec3 *) malloc(rows * sizeof(Vec3));
        originsGrid.values = origins;
        coverEdgesPerOrigin = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        coverEdgeBoundsPerOrigin = (AABB *) malloc(rows * sizeof(AABB));
        originsGrid.cellSizes = {20.0, 20.0, 10000.0};
    }

    CoverOrigins() { };
    ~CoverOrigins() {
        if (!beenInitialized){
            return;
        }

        free(origins);
        free(coverEdgesPerOrigin);
    }

    CoverOrigins(const CoverOrigins& other) = delete;
    CoverOrigins& operator=(const CoverOrigins& other) = delete;
};

class CoverEdges : public ColStore {
public:
    int64_t * originId;
    int64_t * clusterId;
    AABB * aabbs;

    void init(int64_t rows, int64_t numFiles) {
        ColStore::init(rows, numFiles, {});
        originId = (int64_t *) malloc(rows * sizeof(int64_t));
        clusterId = (int64_t *) malloc(rows * sizeof(int64_t));
        aabbs = (AABB *) malloc(rows * sizeof(AABB));
    }

    CoverEdges() { };
    ~CoverEdges() {
        if (!beenInitialized){
            return;
        }

        free(originId);
        free(clusterId);
        free(aabbs);
    }

    CoverEdges(const CoverEdges& other) = delete;
    CoverEdges& operator=(const CoverEdges& other) = delete;
};

void loadCover(CoverOrigins & origins, CoverEdges & edges, string dataPath);

void buildCoverIndex(CoverOrigins & origins, CoverEdges & edges);

#endif //CSKNOW_LOAD_COVER_H
