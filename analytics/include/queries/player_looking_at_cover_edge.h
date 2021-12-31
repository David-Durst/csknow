//
// Created by durst on 12/31/21.
//

#ifndef CSKNOW_PLAYER_LOOKING_AT_COVER_EDGE_H
#define CSKNOW_PLAYER_LOOKING_AT_COVER_EDGE_H
#include "load_data.h"
#include "load_cover.h"
#include "queries/nearest_origin.h"
#include "query.h"
#include <string>
#include <map>
using std::string;
using std::map;

class PlayerLookingAtCoverEdgeResult : public QueryResult {
public:
    vector<RangeIndexEntry> playerLookingAtCoverEdgePerRound;
    vector<int64_t> tickId;
    vector<int64_t> curPlayerAtTickId;
    vector<int64_t> curPlayerId;
    vector<int64_t> lookerPlayerAtTickId;
    vector<int64_t> lookerPlayerId;
    vector<int64_t> nearestOriginId;

    PlayerLookingAtCoverEdgeResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        vector<int64_t> result;
        for (int i = playerLookingAtCoverEdgePerRound[otherTableIndex].minId;
             i <= playerLookingAtCoverEdgePerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << tickId[index] << "," << curPlayerAtTickId[index] << "," << curPlayerId[index] << ","
           << lookerPlayerAtTickId[index] << "," << lookerPlayerId[index] << ","
           << nearestOriginId[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id",
                "cur player at tick id", "cur player id",
                "looker player at tick id", "looker player id",
                "nearest origin id"};
    }

    vector<string> getOtherColumnNames() {
        return {};
    }
};

PlayerLookingAtCoverEdgeResult
queryPlayerLookingAtCoverEdge(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                              const CoverOrigins & coverOrigins, const CoverEdges & coverEdges,
                              const NearestOriginResult & nearestOriginResult);


#endif //CSKNOW_PLAYER_LOOKING_AT_COVER_EDGE_H
