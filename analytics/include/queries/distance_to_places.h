//
// Created by durst on 7/24/22.
//

#ifndef CSKNOW_DISTANCE_TO_PLACES_H
#define CSKNOW_DISTANCE_TO_PLACES_H
#define NOT_CLOSEST_DISTANCE -1.
#include "queries/reachable.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class DistanceToPlacesResult : public QueryResult {
public:
    vector<AABB> coordinate;
    vector<AreaId> areaIndexToId;
    map<string, PlaceIndex> placeNameToIndex;
    map<string, vector<AreaId>> placeToArea;
    vector<PlaceIndex> areaToPlace;
    vector<string> places;
    vector<double> closestDistanceMatrix;
    vector<double> medianDistanceMatrix;
    // closest area id for each place
    vector<AreaId> closestAreaIndexMatrix;
    vector<AreaId> medianAreaIndexMatrix;
    int64_t numAreas;
    int64_t numPlaces;

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        return {};
    }

    DistanceToPlacesResult() {
        this->variableLength = false;
        this->nonTemporal = true;
        this->overlay = true;
        this->extension = ".place_dist";
    };

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index;
        ss << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
           << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        for (int i = 0; i < coordinate.size(); i++) {
            PlaceIndex placeIndex = areaToPlace[i];
            if (places[placeIndex] != "" && i == getClosestArea(index, placeIndex)) {
                ss << "," << closestDistanceMatrix[index * numPlaces + placeIndex];
            }
            else if (places[placeIndex] != "" && i == getMedianArea(index, placeIndex)) {
                ss << "," << medianDistanceMatrix[index * numPlaces + placeIndex];
            }
            else {
                ss << "," << NOT_CLOSEST_DISTANCE;
            }
        }
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {};
    }

    vector<string> getOtherColumnNames() {
        vector<string> nameVector = {"min_x", "min_y", "min_z", "max_x", "max_y", "max_z"};
        for (uint64_t i = 0; i < coordinate.size(); i++) {
            nameVector.push_back(std::to_string(i));
        }
        return nameVector;
    }

    double getClosestDistance(int64_t srcArea, PlaceIndex dstPlace) const {
        return closestDistanceMatrix[srcArea * numPlaces + dstPlace];
    }

    double getClosestDistance(AreaId srcAreaId, string dstPlaceName, const nav_mesh::nav_file & navFile) const {
        int64_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second;
        PlaceIndex dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return getClosestDistance(srcArea, dstPlace);
    }

    int64_t getClosestArea(int64_t srcArea, int64_t dstPlace) const {
        return closestAreaIndexMatrix[srcArea * numPlaces + dstPlace];
    }

    AreaId getClosestArea(AreaId srcAreaId, string dstPlaceName, const nav_mesh::nav_file & navFile) const {
        if (dstPlaceName == "") {
            return srcAreaId;
        }
        size_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second,
                dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return areaIndexToId[getClosestArea(srcArea, dstPlace)];
    }

    double getMedianDistance(int64_t srcArea, PlaceIndex dstPlace) const {
        return medianDistanceMatrix[srcArea * numPlaces + dstPlace];
    }

    double getMedianDistance(AreaId srcAreaId, string dstPlaceName, const nav_mesh::nav_file & navFile) const {
        int64_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second;
        PlaceIndex dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return getMedianDistance(srcArea, dstPlace);
    }

    int64_t getMedianArea(int64_t srcArea, int64_t dstPlace) const {
        return medianAreaIndexMatrix[srcArea * numPlaces + dstPlace];
    }

    AreaId getMedianArea(AreaId srcAreaId, string dstPlaceName, const nav_mesh::nav_file & navFile) const {
        if (dstPlaceName == "") {
            return srcAreaId;
        }
        size_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second,
                dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return areaIndexToId[getMedianArea(srcArea, dstPlace)];
    }

    void load(string mapsPath, string mapName, const nav_mesh::nav_file & navFile,
              const ReachableResult & reachableResult);
};

DistanceToPlacesResult queryDistanceToPlaces(const nav_mesh::nav_file & navFile,
                                             const ReachableResult & reachableResult);

#endif //CSKNOW_DISTANCE_TO_PLACES_H
