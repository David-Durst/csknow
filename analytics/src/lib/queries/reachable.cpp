//
// Created by durst on 1/11/22.
//

#include "queries/reachable.h"
#include <filesystem>

ReachableResult queryReachable(const VisPoints & visPoints, const MapMeshResult & mapMeshResult, const string & overlayLabelsQuery,
                               const string & mapsPath, const string & mapName) {
    ReachableResult result(overlayLabelsQuery, visPoints);
    string reachableFileName = mapName + ".reach";
    string reachableFilePath = mapsPath + "/" + reachableFileName;

    if (std::filesystem::exists(reachableFilePath)) {
        result.load(mapsPath, mapName);
    }
    else {
        result.numAreas = mapMeshResult.size;
        result.coordinate = mapMeshResult.coordinate;
        result.distanceMatrix.resize(result.numAreas * result.numAreas, std::numeric_limits<double>::max());

        // setup initial connections in reachability matrix
        for (int64_t i = 0; i < result.numAreas; i++) { Vec3 curCenter = getCenter(mapMeshResult.coordinate[i]);
            for (int64_t j = 0; j < static_cast<int64_t>(mapMeshResult.connectionAreaIds[i].size()); j++) {
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
        result.save(mapsPath, mapName);
    }
    result.computeCellDistances();
    return result;
}

void ReachableResult::load(const string & mapsPath, const string & mapName) {
    string reachableFileName = mapName + ".reach";
    string reachableFilePath = mapsPath + "/" + reachableFileName;

    if (std::filesystem::exists(reachableFilePath)) {
        std::ifstream fsReach(reachableFilePath);
        string reachBuf;
        size_t index = 0;
        getline(fsReach, reachBuf); // skip first line
        while (getline(fsReach, reachBuf)) {
            stringstream reachStream(reachBuf);
            string value;

            getline(reachStream, value, ','); // skip index

            getline(reachStream, value, ',');
            coordinate.push_back({});
            coordinate[index].min.x = std::stod(value);

            getline(reachStream, value, ',');
            coordinate[index].min.y = stod(value);

            getline(reachStream, value, ',');
            coordinate[index].min.z = stod(value);

            getline(reachStream, value, ',');
            coordinate[index].max.x = stod(value);

            getline(reachStream, value, ',');
            coordinate[index].max.y = stod(value);

            getline(reachStream, value, ',');
            coordinate[index].max.z = stod(value);

            while (getline(reachStream, value, ',')) {
                distanceMatrix.push_back(stod(value));
            }
            index++;
        }
        size = static_cast<int64_t>(coordinate.size());
        numAreas = size;
        if (index * index != distanceMatrix.size()) {
            throw std::runtime_error("number of distances isn't square of number of nav mesh areas");
        }
    }
    else {
        throw std::runtime_error("no reachability file");
    }
}

void ReachableResult::computeCellDistances() {
    std::cout << "startign cell distances" << std::endl;
    double maxDistance = -1*std::numeric_limits<double>::max(),
        minDistance = std::numeric_limits<double>::max();
    size_t numCells = visPoints.getCellVisPoints().size();
#pragma omp parallel for
    for (size_t i = 0; i < visPoints.getAreaVisPoints().size(); i++) {
        const auto & srcAreaVisPoint = visPoints.getAreaVisPoints()[i];
        for (const auto & dstAreaVisPoint : visPoints.getAreaVisPoints()) {
            double distance = getDistance(srcAreaVisPoint.areaId, srcAreaVisPoint.areaId, visPoints);
            if (distance > maxDistance) {
                maxDistance = distance;
            }
            if (distance < minDistance && distance >= 0.) {
                minDistance = distance;
            }
        }
    }

    for (size_t i = 0; i < numCells; i++) {
        scaledCellDistanceMatrix.push_back(vector<uint8_t>(numCells, 0));
    }

#pragma omp parallel for
    for (size_t i = 0; i < visPoints.getAreaVisPoints().size(); i++) {
        const auto & srcAreaVisPoint = visPoints.getAreaVisPoints()[i];
        for (const auto & dstAreaVisPoint : visPoints.getAreaVisPoints()) {
            double distance = getDistance(srcAreaVisPoint.areaId, srcAreaVisPoint.areaId, visPoints);
            for (const auto & srcCellId : srcAreaVisPoint.cells) {
                for (const auto & dstCellId : dstAreaVisPoint.cells) {
                    if (distance < 0) {
                        scaledCellDistanceMatrix[srcCellId][dstCellId] = 255;
                    }
                    else {
                        scaledCellDistanceMatrix[srcCellId][dstCellId] =
                            static_cast<uint8_t>(255 * (distance - minDistance) / (maxDistance - minDistance));
                    }
                }
            }
        }
    }
    std::cout << "ending cell distances" << std::endl;
}
