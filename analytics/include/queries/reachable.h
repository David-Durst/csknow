//
// Created by durst on 1/11/22.
//

#ifndef CSKNOW_REACHABLE_H
#define CSKNOW_REACHABLE_H
#include "queries/nav_mesh.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class ReachableResult : public QueryResult {
public:
    vector<AABB> coordinate;
    vector<double> distanceMatrix;
    int64_t numAreas;

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        return {};
    }

    ReachableResult() {
        this->variableLength = false;
        this->allTicks = true;
        this->extension = ".reach";
    };

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index;
        ss << "," << coordinate[index].min.x << "," << coordinate[index].min.y << "," << coordinate[index].min.z
           << "," << coordinate[index].max.x << "," << coordinate[index].max.y << "," << coordinate[index].max.z;
        for (int i = 0; i < coordinate.size(); i++) {
            ss << "," << distanceMatrix[index * coordinate.size() + i];
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

    double getDistance(int64_t src, int64_t dst) {
        return distanceMatrix[src * numAreas + dst];
    }

    void load(string mapsPath, string mapName);
};

ReachableResult queryReachable(const MapMeshResult & mapMeshResult);

#endif //CSKNOW_REACHABLE_H
