//
// Created by durst on 11/4/21.
//

#ifndef CSKNOW_NAV_MESH_H
#define CSKNOW_NAV_MESH_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "navmesh/nav_file.h"
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

class MapMeshResult : public QueryResult {
public:
    vector<int64_t> id;
    vector<int64_t> areaId;
    vector<string> placeName;
    vector<AABB> coordinate;
    vector<vector<int64_t>> connectionAreaIds;
    map<int64_t, int64_t> areaToInternalId;

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        return {};
    }

    MapMeshResult() {
        this->variableLength = false;
        this->nonTemporal = true;
        this->overlay = true;
        this->overlayLabels = true;
    };

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << id[index] << ",";
        // skip empty strings that just have null terminator
        if (placeName[index].size() > 1) {
            int lastNotNull = placeName[index].size() - 1;
            if (placeName[index][lastNotNull] == '\0') {
                lastNotNull--;
            }
            ss.write(placeName[index].c_str(), lastNotNull + 1);
        }
        ss << "," << areaId[index] << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
            << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z << ",";
        for (int i = 0; i < connectionAreaIds[index].size(); i++) {
            if (i > 0) {
                ss << ";";
            }
            ss << connectionAreaIds[index][i];
        }
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {};
    }

    vector<string> getOtherColumnNames() {
        return {"placeName", "areaId", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "connectionAreaIds"};
    }
};

MapMeshResult queryMapMesh(nav_mesh::nav_file & navFile);

#endif //CSKNOW_NAV_MESH_H
