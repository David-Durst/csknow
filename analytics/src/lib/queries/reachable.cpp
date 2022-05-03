//
// Created by durst on 1/11/22.
//

#include "queries/reachable.h"

ReachableResult queryReachable(const MapMeshResult & mapMeshResult) {
    ReachableResult result;
    int64_t i = 0;
    result.numAreas = mapMeshResult.size;
    result.coordinate = mapMeshResult.coordinate;
    result.distanceMatrix.resize(result.numAreas * result.numAreas, std::numeric_limits<double>::max());

    // setup initial connections in reachability matrix
    for (int64_t i = 0; i < result.numAreas; i++) { Vec3 curCenter = getCenter(mapMeshResult.coordinate[i]);
        for (int64_t j = 0; j < mapMeshResult.connectionAreaIds[i].size(); j++) {
            int64_t destId = mapMeshResult.areaToInternalId.at(mapMeshResult.connectionAreaIds[i][j]);
            Vec3 destCenter = getCenter(mapMeshResult.coordinate[destId]);
            result.distanceMatrix[i * result.numAreas + destId] = computeDistance(curCenter, destCenter);
        }
        result.distanceMatrix[i * result.numAreas + i] = 0;
    }

    // then compute nearest adjacency matrix with Warshall's Algorithm
    bool changed_value = true;
    while (changed_value) {
        changed_value = false;
        for (int64_t i = 0; i < result.numAreas; i++) {
            for (int64_t j = 0; j < result.numAreas; j++) {
                double old_value = result.distanceMatrix[i * result.numAreas + j];
                for (int64_t k = 0; k < result.numAreas; k++) {
                    result.distanceMatrix[i * result.numAreas + j] = std::min(
                            result.distanceMatrix[i * result.numAreas + j],
                            result.distanceMatrix[i * result.numAreas + k] + result.distanceMatrix[k * result.numAreas + j]
                            );
                }
                if (result.distanceMatrix[i * result.numAreas + j] != old_value) {
                    changed_value = true;
                }
            }
        }
    }

    // replace all max limits with -1 as not reachable
    for (int64_t i = 0; i < result.numAreas; i++) {
        for (int64_t j = 0; j < result.numAreas; j++) {
            if (result.distanceMatrix[i * result.numAreas + j] == std::numeric_limits<double>::max()) {
                result.distanceMatrix[i * result.numAreas + j] = -1;
            }
        }
    }

    result.size = mapMeshResult.size;
    return result;
}
