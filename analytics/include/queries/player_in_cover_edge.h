//
// Created by durst on 12/30/21.
//

#ifndef CSKNOW_PLAYER_IN_COVER_EDGE_H
#define CSKNOW_PLAYER_IN_COVER_EDGE_H
#include "load_data.h"
#include "load_cover.h"
#include "queries/nearest_origin.h"
#include "query.h"
#include <string>
#include <map>
using std::string;
using std::map;

class PlayerInCoverEdgeResult : public QueryResult {
public:
    vector<RangeIndexEntry> playerInCoverEdgePerRound;
    vector<int64_t> tickId;
    vector<int64_t> lookerPlayerAtTickId;
    vector<int64_t> lookerPlayerId;
    vector<int64_t> lookedAtPlayerAtTickId;
    vector<int64_t> lookedAtPlayerId;
    vector<int64_t> nearestOriginId;

    PlayerInCoverEdgeResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        vector<int64_t> result;
        for (int i = playerInCoverEdgePerRound[otherTableIndex].minId; i <= playerInCoverEdgePerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << tickId[index] << "," << lookerPlayerAtTickId[index] << "," << lookerPlayerId[index] << ","
           << lookedAtPlayerAtTickId[index] << "," << lookedAtPlayerId[index] << ","
           << nearestOriginId[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id",
                "looker player at tick id", "looker player id",
                "looked at player at tick id", "looked at player id",
                "nearest origin id"};
    }

    vector<string> getOtherColumnNames() {
        return {};
    }
};

PlayerInCoverEdgeResult queryPlayerInCoverEdge(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                               const CoverOrigins & coverOrigins, const CoverEdges & coverEdges,
                                               const NearestOriginResult & nearestOriginResult);

#endif //CSKNOW_PLAYER_IN_COVER_EDGE_H
