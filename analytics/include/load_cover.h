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

    IVec3 getCellCoordinates(Vec3 v) {
        return {
            (int64_t) std::floor((v.x - minValues.x) / cellSizes.x),
            (int64_t) std::floor((v.y - minValues.y) / cellSizes.y),
            (int64_t) std::floor((v.z - minValues.z) / cellSizes.z)
        };
    }

    int64_t getCellIndex(IVec3 v) {
        return v.x * numCells.y * numCells.z + v.y * numCells.z + v.z;
    }
};

class CoverOrigins : public ColStore {
public:
    Vec3 * origins;
    GridIndex originsGrid;
    RangeIndex coverEdgesPerOrigin;

    void init(int64_t rows, int64_t numFiles) {
        ColStore::init(rows, numFiles, {});
        origins = (Vec3 *) malloc(rows * sizeof(Vec3));
        coverEdgesPerOrigin = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        originsGrid.cellSizes = {20.0, 20.0, 20.0};
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
