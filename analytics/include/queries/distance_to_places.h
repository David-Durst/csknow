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
    map<string, AABB> placeToAABB;
    vector<PlaceIndex> areaToPlace;
    vector<string> places;
    vector<double> closestDistanceMatrix;
    vector<double> medianDistanceMatrix;
    // closest area id for each place
    vector<int64_t> closestAreaIndexMatrix;
    vector<int64_t> medianAreaIndexMatrix;
    int64_t numAreas;
    int64_t numPlaces;

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    explicit
    DistanceToPlacesResult(const string & overlayLabelsQuery) {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        extension = ".place_dist";
        numAreas = 0;
        numPlaces = 0;
        this->overlayLabelsQuery = overlayLabelsQuery;
    };

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << index;
        s << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
          << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        for (size_t i = 0; i < coordinate.size(); i++) {
            PlaceIndex placeIndex = areaToPlace[i];
            // -1 for invalid place gets cast to max value
            if (placeIndex < places.size() && !places[placeIndex].empty() &&
                static_cast<int64_t>(i) == getClosestArea(index, placeIndex)) {
                s << "," << closestDistanceMatrix[index * numPlaces + placeIndex];
            }
            else if (placeIndex < places.size() && !places[placeIndex].empty() &&
                static_cast<int64_t>(i) == getMedianArea(index, placeIndex)) {
                s << "," << medianDistanceMatrix[index * numPlaces + placeIndex];
            }
            else {
                s << "," << NOT_CLOSEST_DISTANCE;
            }
        }
        s << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {};
    }

    vector<string> getOtherColumnNames() override {
        vector<string> nameVector = {"min_x", "min_y", "min_z", "max_x", "max_y", "max_z"};
        for (uint64_t i = 0; i < coordinate.size(); i++) {
            nameVector.push_back(std::to_string(i));
        }
        return nameVector;
    }

    [[nodiscard]]
    double getClosestDistance(int64_t srcArea, PlaceIndex dstPlace) const {
        return closestDistanceMatrix[srcArea * numPlaces + dstPlace];
    }

    [[nodiscard]]
    PlaceIndex getClosestValidPlace(int64_t srcArea, const nav_mesh::nav_file & navFile) const {
        PlaceIndex actualPlaceIndex = areaToPlace[srcArea];
        string actualPlaceName = navFile.get_place(actualPlaceIndex);
        if (actualPlaceName == "INVALID") {
            double minDistance = std::numeric_limits<double>::max();
            PlaceIndex closestPlaceIndex = std::numeric_limits<uint16_t>::max();
            for (size_t i = 0; i < places.size(); i++) {
                double curDistance = getClosestDistance(srcArea, i);
                if (curDistance < minDistance) {
                    closestPlaceIndex = i;
                    minDistance = curDistance;
                }
            }
            return closestPlaceIndex;
        }
        else {
            return actualPlaceIndex;
        }
    }

    [[nodiscard]]
    double getClosestDistance(string srcPlaceName, PlaceIndex dstPlace) const {
        double minDistance = std::numeric_limits<double>::max();
        for (const auto & srcAreaId : placeToArea.at(srcPlaceName)) {
            minDistance = std::min(minDistance, getClosestDistance(srcAreaId, dstPlace));
        }
        return minDistance;
    }

    [[nodiscard]]
    double getClosestDistance(AreaId srcAreaId, const string & dstPlaceName, const nav_mesh::nav_file & navFile) const {
        int64_t srcArea = static_cast<int64_t>(navFile.m_area_ids_to_indices.find(srcAreaId)->second);
        PlaceIndex dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return getClosestDistance(srcArea, dstPlace);
    }

    [[nodiscard]]
    int64_t getClosestArea(int64_t srcArea, int64_t dstPlace) const {
        return closestAreaIndexMatrix[srcArea * numPlaces + dstPlace];
    }

    [[nodiscard]]
    AreaId getClosestArea(AreaId srcAreaId, const string & dstPlaceName, const nav_mesh::nav_file & navFile) const {
        if (dstPlaceName.empty()) {
            return srcAreaId;
        }
        size_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second,
                dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return areaIndexToId[getClosestArea(static_cast<int64_t>(srcArea), static_cast<int64_t>(dstPlace))];
    }

    [[nodiscard]]
    double getMedianDistance(int64_t srcArea, PlaceIndex dstPlace) const {
        return medianDistanceMatrix[srcArea * numPlaces + dstPlace];
    }

    [[nodiscard]] [[maybe_unused]]
    double getMedianDistance(AreaId srcAreaId, const string & dstPlaceName, const nav_mesh::nav_file & navFile) const {
        int64_t srcArea = static_cast<int64_t>(navFile.m_area_ids_to_indices.find(srcAreaId)->second);
        PlaceIndex dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return getMedianDistance(srcArea, dstPlace);
    }

    [[nodiscard]]
    int64_t getMedianArea(int64_t srcArea, int64_t dstPlace) const {
        return medianAreaIndexMatrix[srcArea * numPlaces + dstPlace];
    }

    [[nodiscard]]
    AreaId getMedianArea(AreaId srcAreaId, const string & dstPlaceName, const nav_mesh::nav_file & navFile) const {
        if (dstPlaceName.empty()) {
            return srcAreaId;
        }
        size_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second,
                dstPlace = placeNameToIndex.find(dstPlaceName)->second;
        return areaIndexToId[getMedianArea(static_cast<int64_t>(srcArea), static_cast<int64_t>(dstPlace))];
    }

    [[nodiscard]]
    bool placeValid(PlaceIndex placeIndex) const {
        return placeIndex < places.size();
    }

    void writeAreasPerPlace(const string & path) const {
        std::ofstream ofile(path);
        ofile << "place name, num areas" << std::endl;
        for (const auto & [placeName, areas] : placeToArea) {
            ofile << placeName << "," << areas.size() << std::endl;
        }
        ofile.close();
    }

    void load(const string & mapsPath, const string & mapName, const nav_mesh::nav_file & navFile,
              const ReachableResult & reachableResult);
};

[[maybe_unused]]
DistanceToPlacesResult queryDistanceToPlaces(const nav_mesh::nav_file & navFile,
                                             const ReachableResult & reachableResult,
                                             const string & overlayLabelsQuery,
                                             const string & mapsPath, const string & mapName);

#endif //CSKNOW_DISTANCE_TO_PLACES_H
