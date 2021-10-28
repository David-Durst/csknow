#ifndef CSKNOW_LOOKING_H
#define CSKNOW_LOOKING_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
#include <string>
#include <map>
using std::string;
using std::map;

class LookingResult : public QueryResult {
public:
    vector<RangeIndexEntry> lookersPerRound;
    vector<int64_t> tickId;
    vector<int64_t> lookerPlayerAtTickId;
    vector<int64_t> lookerPlayerId;
    vector<int64_t> lookedAtPlayerAtTickId;
    vector<int64_t> lookedAtPlayerId;

    LookingResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        vector<int64_t> result;
        for (int i = lookersPerRound[otherTableIndex].minId; i <= lookersPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << tickId[index] << "," << lookerPlayerAtTickId[index] << "," << lookerPlayerId[index] << ","
           << lookedAtPlayerAtTickId[index] << "," << lookedAtPlayerId[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id",
                "looker player at tick id", "looker player id",
                "looked at player at tick id", "looked at player id"};
    }

    vector<string> getOtherColumnNames() {
        return {};
    }
};

LookingResult queryLookers(const Games & games, const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick);

#endif //CSKNOW_LOOKING_H
