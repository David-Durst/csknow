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
    vector<int64_t> coverEdgeId;

    PlayerInCoverEdgeResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        // no indexes on results
        vector<int64_t> result;
        for (int64_t i = playerInCoverEdgePerRound[otherTableIndex].minId; i <= playerInCoverEdgePerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index << "," << tickId[index] << "," << lookerPlayerAtTickId[index] << "," << lookerPlayerId[index] << ","
           << lookedAtPlayerAtTickId[index] << "," << lookedAtPlayerId[index] << ","
           << nearestOriginId[index] << "," << coverEdgeId[index] << std::endl;
    }

    [[nodiscard]]
    vector<string> getForeignKeyNames() override {
        return {"tick id",
                "looker player at tick id", "looker player id",
                "looked at player at tick id", "looked at player id",
                "nearest origin id", "cover edge id"};
    }

    [[nodiscard]]
    vector<string> getOtherColumnNames() override {
        return {};
    }
};

PlayerInCoverEdgeResult queryPlayerInCoverEdge(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                               const CoverOrigins & coverOrigins, const CoverEdges & coverEdges,
                                               const NearestOriginResult & nearestOriginResult);

#endif //CSKNOW_PLAYER_IN_COVER_EDGE_H
