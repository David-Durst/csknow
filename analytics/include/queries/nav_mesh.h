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

class MapMeshResult : public QueryResult {
public:
    vector<int64_t> id;
    vector<string> placeName;
    vector<AABB> coordinate;

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        return {};
    }

    MapMeshResult() {
        this->variableLength = false;
        this->allTicks = true;
    };

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << id[index] << "," << placeName[index] << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
            << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {};
    }

    vector<string> getOtherColumnNames() {
        return {"placeName", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"};
    }
};

MapMeshResult queryMapMesh(nav_mesh::nav_file & navFile);

#endif //CSKNOW_NAV_MESH_H
