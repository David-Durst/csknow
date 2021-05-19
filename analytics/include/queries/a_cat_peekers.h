//
// Created by durst on 5/18/21.
//

#ifndef CSKNOW_A_CAT_PEEKERS_H
#define CSKNOW_A_CAT_PEEKERS_H

#include "queries/query.h"
#include "load_data.h"
#include "geometry.h"
#include <string>
#include <vector>
using std::string;
using std::vector;

class ACatPeekers : public QueryResult {
public:
    vector<int64_t> playerAtTickId;
    vector<double> posX;
    vector<double> posY;
    vector<double> posZ;
    vector<double> viewX;
    vector<double> viewY;
    vector<double> wallX;
    vector<double> wallY;
    vector<double> wallZ;

    vector<AABB> walls;
    vector<int64_t> wallId;
    ACatPeekers(vector<AABB> walls) {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
        this->walls = walls;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        return {};
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << playerAtTickId[index] << "," << posX[index] << "," << posY[index] << "," << posZ[index]
            << "," << viewX[index] << "," << viewY[index] << "," << wallId[index] << ","
            << wallX[index] << "," << wallY[index] << "," << wallZ[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"player at tick id"};
    }

    vector<string> getOtherColumnNames() {
        return {"pos x", "pos y", "pos z", "view x", "wall id", "wall x", "wall y", "wall z"};
    }
};


ACatPeekers queryACatPeekers(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick);


#endif //CSKNOW_A_CAT_PEEKERS_H
