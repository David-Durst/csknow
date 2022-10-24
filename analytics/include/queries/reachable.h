//
// Created by durst on 1/11/22.
//

#ifndef CSKNOW_REACHABLE_H
#define CSKNOW_REACHABLE_H
#include "queries/nav_mesh.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class ReachableResult : public QueryResult {
public:
    vector<AABB> coordinate;
    vector<double> distanceMatrix;
    const VisPoints & visPoints;
    vector<vector<uint8_t>> scaledCellDistanceMatrix;
    int64_t numAreas;

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    ReachableResult(const string & overlayLabelsQuery, const VisPoints & visPoints) : visPoints(visPoints) {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        extension = ".reach";
        numAreas = 0;
        this->overlayLabelsQuery = overlayLabelsQuery;
    };

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index;
        ss << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
           << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        for (size_t i = 0; i < coordinate.size(); i++) {
            ss << "," << distanceMatrix[index * coordinate.size() + i];
        }
        ss << std::endl;
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
    double getDistance(int64_t src, int64_t dst) const {
        return distanceMatrix[src * numAreas + dst];
    }

    [[nodiscard]]
    double getDistance(AreaId srcId, AreaId dstId, const nav_mesh::nav_file & navFile) const {
        size_t src = navFile.m_area_ids_to_indices.find(srcId)->second,
            dst = navFile.m_area_ids_to_indices.find(dstId)->second;
        return getDistance(static_cast<int64_t>(src), static_cast<int64_t>(dst));
    }

    [[nodiscard]]
    double getDistance(AreaId srcId, AreaId dstId, const VisPoints & visPoints) const {
        size_t src = visPoints.areaIdToIndex(srcId),
            dst = visPoints.areaIdToIndex(dstId);
        return getDistance(static_cast<int64_t>(src), static_cast<int64_t>(dst));
    }

    void load(const string & mapsPath, const string & mapName);
    void computeCellDistances();
};

[[maybe_unused]] ReachableResult queryReachable(const VisPoints & visPoints, const MapMeshResult & mapMeshResult,
                                                const string & overlayLabelsQuery,
                                                const string & mapsPath, const string & mapName);

#endif //CSKNOW_REACHABLE_H
