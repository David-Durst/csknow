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
    NavVisibleResult(const string & overlayLabelsQuery, bool area, const VisPoints & visPoints,
                     const string & mapName) :
        area(area), visPoints(visPoints) {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        numPoints = INVALID_ID;
        this->overlayLabelsQuery = overlayLabelsQuery;
        haveBlob = true;
        blobFileName = visPoints.getVisFileName(mapName, area, true);
        blobBytesPerRow = area ? visPoints.getAreaVisPoints().front().visibleFromCurPoint.getInternal().size() :
            visPoints.getCellVisPoints().front().visibleFromCurPoint.getInternal().size();
    };

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {};
    }

    vector<string> getOtherColumnNames() override {
        return {};
    }

    [[maybe_unused]]
    bool getVisible(int64_t src, int64_t dst) {
        return visibleMatrix[src * numPoints + dst];
    }
};

#endif //CSKNOW_NAV_VISIBLE_H
