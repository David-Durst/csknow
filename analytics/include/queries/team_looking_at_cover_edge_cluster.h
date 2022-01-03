//
// Created by durst on 12/31/21.
//

#ifndef CSKNOW_TEAM_LOOKING_AT_COVER_EDGE_CLUSTER_H
#define CSKNOW_TEAM_LOOKING_AT_COVER_EDGE_CLUSTER_H
#include "load_data.h"
#include "load_cover.h"
#include "queries/nearest_origin.h"
#include "query.h"
#include <string>
#include <map>
using std::string;
using std::map;

class TeamLookingAtCoverEdgeCluster : public QueryResult {
public:
    vector<RangeIndexEntry> teamLookingAtCoverEdgeClusterPerRound;
    vector<int64_t> tickId;
    vector<int64_t> curPlayerAtTickId;
    vector<int64_t> curPlayerId;
    vector<int64_t> lookerPlayerAtTickId;
    vector<int64_t> lookerPlayerId;
    vector<int64_t> nearestOriginId;
    vector<int64_t> coverEdgeClusterId;

    TeamLookingAtCoverEdgeCluster() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        vector<int64_t> result;
        for (int i = teamLookingAtCoverEdgeClusterPerRound[otherTableIndex].minId;
             i <= teamLookingAtCoverEdgeClusterPerRound[otherTableIndex].maxId; i++) {
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
           << nearestOriginId[index] << "," << coverEdgeClusterId[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id",
                "cur player at tick id", "cur player id",
                "looker player at tick id", "looker player id",
                "nearest origin id", "cover edge cluster id"};
    }

    vector<string> getOtherColumnNames() {
        return {};
    }
};

TeamLookingAtCoverEdgeCluster
queryTeamLookingAtCoverEdgeCluster(const Games & games, const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                   const CoverOrigins & coverOrigins, const CoverEdges & coverEdges,
                                   const NearestOriginResult & nearestOriginResult);


#endif //CSKNOW_TEAM_LOOKING_AT_COVER_EDGE_CLUSTER_H
