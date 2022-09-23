//
// Created by durst on 7/24/22.
//

#include "queries/distance_to_places.h"
#include <filesystem>

void setupBasics(DistanceToPlacesResult & result, const nav_mesh::nav_file & navFile,
                 const ReachableResult & reachableResult) {
    result.coordinate = reachableResult.coordinate;
    for (size_t i = 0; i < navFile.m_places.size(); i++) {
        result.placeNameToIndex[navFile.get_place(i)] = i;
        result.places.push_back(navFile.get_place(i));
    }
    for (const auto & area : navFile.m_areas) {
        result.placeToArea[navFile.get_place(area.m_place)].push_back(area.get_id());
    }
    for (size_t i = 0; i < navFile.m_areas.size(); i++) {
        result.areaIndexToId.push_back(navFile.m_areas[i].get_id());
        result.areaToPlace.push_back(navFile.m_areas[i].m_place);
    }
    result.numAreas = reachableResult.size;
    result.numPlaces = static_cast<int64_t>(navFile.m_places.size());

    result.closestDistanceMatrix.resize(result.numAreas * result.numPlaces, NOT_CLOSEST_DISTANCE);
    result.medianDistanceMatrix.resize(result.numAreas * result.numPlaces, NOT_CLOSEST_DISTANCE);
    result.closestAreaIndexMatrix.resize(result.numAreas * result.numPlaces, INVALID_ID);
    result.medianAreaIndexMatrix.resize(result.numAreas * result.numPlaces, INVALID_ID);
}

[[maybe_unused]]
DistanceToPlacesResult queryDistanceToPlaces(const nav_mesh::nav_file & navFile,
                                             const ReachableResult & reachableResult) {
    DistanceToPlacesResult result;
    setupBasics(result, navFile, reachableResult);

    for (int64_t i = 0; i < result.numAreas; i++) {
        for (int64_t j = 0; j < result.numPlaces; j++) {
            struct AreaDistance {
                int64_t areaIndex;
                double distance;
            };
            const vector<AreaId> & areaIds = result.placeToArea[result.places[j]];
            vector<AreaDistance> areaDistances;
            for (size_t k = 0; k < areaIds.size(); k++) {
                int64_t newAreaIndex = static_cast<int64_t>(navFile.m_area_ids_to_indices.find(areaIds[k])->second);
                double newDistance = reachableResult.getDistance(i, newAreaIndex);
                if (newDistance != NOT_CLOSEST_DISTANCE) {
                    areaDistances.push_back({newAreaIndex, newDistance});
                }
            }
            if (!areaDistances.empty()) {
                std::sort(areaDistances.begin(), areaDistances.end(),
                          [](const AreaDistance & a, const AreaDistance & b) { return a.distance < b.distance; });
                result.closestDistanceMatrix[i * result.numPlaces + j] = areaDistances[0].distance;
                result.closestAreaIndexMatrix[i * result.numPlaces + j] = areaDistances[0].areaIndex;
                int median = std::max(0, static_cast<int>(static_cast<double>(areaDistances.size())/2.) - 1);
                result.medianDistanceMatrix[i * result.numPlaces + j] = areaDistances[median].distance;
                result.medianAreaIndexMatrix[i * result.numPlaces + j] = areaDistances[median].areaIndex;
            }
        }
    }

    result.size = reachableResult.size;
    return result;
}

void DistanceToPlacesResult::load(const string & mapsPath, const string & mapName, const nav_mesh::nav_file & navFile,
                                  const ReachableResult & reachableResult) {
    string distToPlacesFileName = mapName + extension;
    string distToPlacesFilePath = mapsPath + "/" + distToPlacesFileName;

    if (std::filesystem::exists(distToPlacesFilePath)) {
        setupBasics(*this, navFile, reachableResult);

        std::ifstream fsDist(distToPlacesFilePath);
        string distBuf;
        size_t index = 0;
        getline(fsDist, distBuf); // skip first line
        while (getline(fsDist, distBuf)) {
            stringstream distStream(distBuf);
            string value;

            getline(distStream, value, ','); // skip index

            getline(distStream, value, ',');
            // read coordinates table but don't use it, already getting it from reachable result
            /*
            coordinate.push_back({});
            coordinate[index].min.x = std::stod(value);
             */

            getline(distStream, value, ',');
            //coordinate[index].min.y = stod(value);

            getline(distStream, value, ',');
            //coordinate[index].min.z = stod(value);

            getline(distStream, value, ',');
            //coordinate[index].max.x = stod(value);

            getline(distStream, value, ',');
            //coordinate[index].max.y = stod(value);

            getline(distStream, value, ',');
            //coordinate[index].max.z = stod(value);

            int64_t areaIndex = 0;
            while (getline(distStream, value, ',')) {
                double distance = stod(value);
                if (distance != NOT_CLOSEST_DISTANCE) {
                    size_t matrixIndex = index * numPlaces + areaToPlace[areaIndex];
                    if (closestDistanceMatrix[matrixIndex] == NOT_CLOSEST_DISTANCE) {
                        closestDistanceMatrix[matrixIndex] = distance;
                        closestAreaIndexMatrix[matrixIndex] = areaIndex;
                        medianDistanceMatrix[matrixIndex] = distance;
                        medianAreaIndexMatrix[matrixIndex] = areaIndex;
                    }
                    else {
                        if (distance < closestDistanceMatrix[matrixIndex]) {
                            closestDistanceMatrix[matrixIndex] = distance;
                            closestAreaIndexMatrix[matrixIndex] = areaIndex;
                        }
                        else {
                            medianDistanceMatrix[matrixIndex] = distance;
                            medianAreaIndexMatrix[matrixIndex] = areaIndex;
                        }
                    }
                }
                areaIndex++;
            }
            index++;
        }
        size = static_cast<int64_t>(coordinate.size());
        if (index * numPlaces != closestDistanceMatrix.size()) {
            throw std::runtime_error("distance to places matrix wrong size");
        }
    }
    else {
        throw std::runtime_error("no reachability file");
    }
}
