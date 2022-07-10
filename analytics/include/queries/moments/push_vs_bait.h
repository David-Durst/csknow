//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_PUSH_VS_BAIT_H
#define CSKNOW_PUSH_VS_BAIT_H
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

class PushVsBaitResult : public QueryResult {
public:
    vector<int64_t> roundId;
    vector<int64_t> playerAtTickId;
    vector<int64_t> playerId;
    vector<double> posX;
    vector<double> posY;
    vector<double> posZ;
    // cluster data per PositionsAndWallViews tick
    vector<int> clusterId;


    PushVsBaitResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        return {};
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << playerAtTickId[index] << "," << posX[index] << "," << posY[index] << "," << posZ[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"player at tick id"};
    }

    vector<string> getOtherColumnNames() {
        return {"pos x", "pos y", "pos z"};
    }
};


PushVsBaitResult queryPushVsBait(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick);

#endif //CSKNOW_PUSH_VS_BAIT_H
