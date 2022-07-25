//
// Created by durst on 7/24/22.
//

#include "queries/distance_to_places.h"
#include <filesystem>

void setupBasics(DistanceToPlacesResult & result, const nav_mesh::nav_file & navFile,
                 const ReachableResult & reachableResult) {
    result.coordinate = reachableResult.coordinate;
    for (size_t i = 0; i < navFile.m_places.size(); i++) {
        result.placeNameToIndex[navFile.m_places[i]] = i;
        result.places.push_back(navFile.get_place(i));
    }
    for (const auto & area : navFile.m_areas) {
        result.placeToArea[navFile.get_place(area.m_place)].push_back(area.get_id());
    }
    for (size_t i = 0; i < navFile.m_areas.size(); i++) {
        result.areaToPlace.push_back(navFile.m_areas[i].m_place);
    }
    result.numAreas = reachableResult.size;
    result.numPlaces = navFile.m_places.size();

    result.distanceMatrix.resize(result.numAreas * result.numPlaces, NOT_CLOSEST_DISTANCE);
    result.closestAreaIndexMatrix.resize(result.numAreas * result.numPlaces, INVALID_ID);
}

DistanceToPlacesResult queryDistanceToPlaces(const nav_mesh::nav_file & navFile,
                                             const ReachableResult & reachableResult) {
    DistanceToPlacesResult result;
    setupBasics(result, navFile, reachableResult);

    for (int64_t i = 0; i < result.numAreas; i++) {
        for (int64_t j = 0; j < result.numPlaces; j++) {
            double minDistance = std::numeric_limits<double>::max();
            int64_t minAreaIndex = INVALID_ID;
            const vector<AreaId> & areaIds = result.placeToArea[result.places[j]];
            for (int64_t k = 0; k < areaIds.size(); k++) {
                int64_t newAreaIndex = navFile.m_area_ids_to_indices.find(areaIds[k])->second;
                double newDistance = reachableResult.getDistance(i, newAreaIndex);
                if (newDistance != NOT_CLOSEST_DISTANCE && newDistance < minDistance) {
                    minDistance = newDistance;
                    minAreaIndex = newAreaIndex;
                }
            }
            result.distanceMatrix[i * result.numPlaces + j] = minDistance;
            result.closestAreaIndexMatrix[i * result.numPlaces + j] = minAreaIndex;
        }
    }

    result.size = reachableResult.size;
    return result;
}

void DistanceToPlacesResult::load(string mapsPath, string mapName, const nav_mesh::nav_file & navFile,
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
            coordinate.push_back({});
            coordinate[index].min.x = std::stod(value);

            getline(distStream, value, ',');
            coordinate[index].min.y = stod(value);

            getline(distStream, value, ',');
            coordinate[index].min.z = stod(value);

            getline(distStream, value, ',');
            coordinate[index].max.x = stod(value);

            getline(distStream, value, ',');
            coordinate[index].max.y = stod(value);

            getline(distStream, value, ',');
            coordinate[index].max.z = stod(value);

            int64_t areaIndex = 0;
            while (getline(distStream, value, ',')) {
                double distance = stod(value);
                if (distance != NOT_CLOSEST_DISTANCE) {
                    distanceMatrix[index * numPlaces + areaToPlace[areaIndex]] = distance;
                    closestAreaIndexMatrix[index * numPlaces + areaToPlace[areaIndex]] = areaIndex;
                }
                areaIndex++;
            }
            index++;
        }
        size = coordinate.size();
        numAreas = size;
        if (index * index != distanceMatrix.size()) {
            throw std::runtime_error("distance to places matrix wrong size");
        }
    }
    else {
        throw std::runtime_error("no reachability file");
    }
}
