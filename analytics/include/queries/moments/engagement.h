//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_ENGAGEMENT_H
#define CSKNOW_ENGAGEMENT_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "bots/analysis/load_save_vis_points.h"
#include "navmesh/nav_file.h"
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

enum class EngagementRole {
    Engaged,
    NotEngaged,
    NUM_ENGAGEMENT_ROLES
};

class AggressionEventResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> startTickId;
    vector<int64_t> endTickId;
    vector<int64_t> tickLength;
    vector<vector<int64_t>> playerId;
    vector<vector<EngagementRole>> role;


    AggressionEventResult() {
        variableLength = true;
        startTickColumn = 0;
        perEventLengthColumn = 2;
        havePlayerLabels = true;
        playerLabels = {"E", "N"};
        playersToLabelColumn = 0;
        playerLabelIndicesColumn = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        vector<int64_t> result;
        for (int i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index] << ",";

        vector<string> tmp;
        for (int64_t pId : playerId[index]) {
            tmp.push_back(std::to_string(pId));
        }
        commaSeparateList(ss, tmp, ";");
        ss << ",";

        tmp.clear();
        for (AggressionRole r : role[index]) {
            tmp.push_back(std::to_string(enumAsInt(r)));
        }
        commaSeparateList(ss, tmp, ";");
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"start tick id", "end tick id", "length"};
    }

    vector<string> getOtherColumnNames() {
        return {"player ids", "roles"};
    }
};


AggressionEventResult queryAggressionRoles(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                           const PlayerAtTick & playerAtTick,
                                           const nav_mesh::nav_file & navFile, const VisPoints & visPoints, const ReachableResult & reachableResult);

#endif //CSKNOW_ENGAGEMENT_H
