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
    bool area;
    vector<AABB> coordinate;
    vector<bool> visibleMatrix;
    int64_t numPoints;
    const VisPoints & visPoints;

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    explicit
    NavVisibleResult(const string & overlayLabelsQuery, bool area, const VisPoints & visPoints) :
        area(area), visPoints(visPoints) {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        numPoints = INVALID_ID;
        this->overlayLabelsQuery = overlayLabelsQuery;
    };

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index;
        ss << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
           << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        if (area) {
            ss << "," << bitsetToBase64(visPoints.getVisPoints()[index].visibleFromCurPoint);
        }
        else {
            ss << "," << bitsetToBase64(visPoints.getCellVisPoints()[index].visibleFromCurPoint);
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
        return visibleMatrix[src * numPoints + dst];
    }
};

NavVisibleResult queryNavVisible(const VisPoints & visPoints, const string & overlayLabelsQuery, bool area);

#endif //CSKNOW_NAV_VISIBLE_H
