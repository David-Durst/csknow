//
// Created by durst on 10/2/22.
//

#ifndef CSKNOW_NAV_CELLS_H
#define CSKNOW_NAV_CELLS_H

#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/query.h"
#include "bots/analysis/load_save_vis_points.h"
#include "geometry.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

class MapCellsResult : public QueryResult {
public:
    vector<int64_t> id;
    vector<int64_t> areaId;
    vector<string> placeName;
    vector<AABB> coordinate;
    vector<vector<int64_t>> connectionAreaIds;

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    MapCellsResult(const string & queryName) {
        variableLength = false;
        nonTemporal = true;
        overlay = true;
        overlayLabelsQuery = queryName;
    };

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << id[index] << "," << placeName[index] << "," << areaId[index]
          << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
          << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z << ",";
        for (size_t i = 0; i < connectionAreaIds[index].size(); i++) {
            if (i > 0) {
                s << ";";
            }
            s << connectionAreaIds[index][i];
        }
        s << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {};
    }

    vector<string> getOtherColumnNames() override {
        return {"placeName", "areaId", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "connectionAreaIds"};
    }
};

MapCellsResult queryMapCells(const VisPoints & visPoints, const nav_mesh::nav_file & navFile, const string & queryName);
#endif //CSKNOW_NAV_CELLS_H
