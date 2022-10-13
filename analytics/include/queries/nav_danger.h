//
// Created by steam on 7/26/22.
//

#ifndef CSKNOW_NAV_DANGER_H
#define CSKNOW_NAV_DANGER_H
#include "queries/nav_mesh.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class NavDangerResult : public QueryResult {
public:
    vector<AABB> coordinate;
    vector<bool> dangerMatrix;
    int64_t numAreas;

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    explicit
    NavDangerResult(const string & overlayLabelsQuery) {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        numAreas = INVALID_ID;
        this->overlayLabelsQuery = overlayLabelsQuery;
    };

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index;
        ss << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
           << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        for (size_t i = 0; i < coordinate.size(); i++) {
            // convert to 0 if visible as using similar range as distance (0 if closer/visible, 1 if farther/not visible)
            ss << "," << (dangerMatrix[index * coordinate.size() + i] ? 0 : 1);
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
    bool getDanger(int64_t srcArea, int64_t dstArea) const {
        return dangerMatrix[srcArea * numAreas + dstArea];
    }

    [[nodiscard]] [[maybe_unused]]
    bool getDanger(AreaId srcAreaId, AreaId dstAreaId, const nav_mesh::nav_file & navFile) const {
        size_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second,
            dstArea = navFile.m_area_ids_to_indices.find(dstAreaId)->second;
        return getDanger(static_cast<int64_t>(srcArea), static_cast<int64_t>(dstArea));
    }

    [[nodiscard]]
    AreaBits getDanger(int64_t srcArea) const {
        AreaBits result;
        for (size_t i = 0; i < coordinate.size(); i++) {
            result.set(i, dangerMatrix[srcArea * coordinate.size() + i]);
        }
        return result;
    }

    [[nodiscard]] [[maybe_unused]]
    AreaBits getDanger(AreaId srcAreaId, const nav_mesh::nav_file & navFile) const {
        size_t srcArea = navFile.m_area_ids_to_indices.find(srcAreaId)->second;
        return getDanger(static_cast<int64_t>(srcArea));
    }
};

NavDangerResult queryNavDanger(const VisPoints & visPoints, const string & overlayLabelsQuery);
#endif //CSKNOW_NAV_DANGER_H
