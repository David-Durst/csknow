//
// Created by durst on 12/30/21.
//

#ifndef CSKNOW_NEAREST_ORIGIN_H
#define CSKNOW_NEAREST_ORIGIN_H
#include "load_data.h"
#include "load_cover.h"
#include "query.h"
#include <string>
#include <map>
using std::string;
using std::map;

class NearestOriginResult : public QueryResult {
public:
    vector<RangeIndexEntry> nearestOriginPerRound;
    vector<int64_t> tickId;
    vector<int64_t> playerAtTickId;
    vector<int64_t> playerId;
    vector<int64_t> originId;

    NearestOriginResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        // no indexes on results
        vector<int64_t> result;
        for (int64_t i = nearestOriginPerRound[otherTableIndex].minId; i <= nearestOriginPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << index << "," << tickId[index] << "," << playerAtTickId[index] << "," << playerId[index] << ","
          << originId[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id",
                "player at tick id", "player id",
                "origin id"};
    }

    vector<string> getOtherColumnNames() override {
        return {};
    }
};

[[maybe_unused]]
NearestOriginResult queryNearestOrigin(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                       const CoverOrigins & coverOrigins);

#endif //CSKNOW_NEAREST_ORIGIN_H
