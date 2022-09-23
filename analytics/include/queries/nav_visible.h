//
// Created by durst on 1/11/22.
//

#ifndef CSKNOW_NAV_VISIBLE_H
#define CSKNOW_NAV_VISIBLE_H
#include "queries/nav_mesh.h"
#include "bots/analysis/load_save_vis_points.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class NavVisibleResult : public QueryResult {
public:
    vector<AABB> coordinate;
    vector<bool> visibleMatrix;
    int64_t numAreas;

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    NavVisibleResult() {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        numAreas = INVALID_ID;
    };

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index;
        ss << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
           << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        for (size_t i = 0; i < coordinate.size(); i++) {
            // convert to 0 if visible as using similar range as distance (0 if closer/visible, 1 if farther/not visible)
            ss << "," << (visibleMatrix[index * coordinate.size() + i] ? 0 : 1);
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

    [[maybe_unused]]
    bool getVisible(int64_t src, int64_t dst) {
        return visibleMatrix[src * numAreas + dst];
    }
};

NavVisibleResult queryNavVisible(const VisPoints & visPoints);

#endif //CSKNOW_NAV_VISIBLE_H
