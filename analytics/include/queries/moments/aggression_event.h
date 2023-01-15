//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_AGGRESSION_EVENT_H
#define CSKNOW_AGGRESSION_EVENT_H
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
#include "queries/reachable.h"
#include "geometry.h"
#include "enum_helpers.h"
#define NOT_VISIBLE_END_SECONDS 5.
#define MAX_BAITER_DISTANCE 750.
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

enum class AggressionRole {
    Pusher,
    Baiter,
    Lurker,
    NUM_AGGRESSION_ROLES [[maybe_unused]]
};

class AggressionEventResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> startTickId;
    vector<int64_t> endTickId;
    vector<int64_t> tickLength;
    vector<vector<int64_t>> playerId;
    vector<vector<AggressionRole>> role;


    AggressionEventResult() {
        variableLength = true;
        startTickColumn = 0;
        perEventLengthColumn = 2;
        havePlayerLabels = true;
        playerLabels = {"P", "B", "L"};
        playersToLabelColumn = 0;
        playerLabelIndicesColumn = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        // Normally would segfault, but this query is slow so I don't run for all rounds in debug cases
        if (otherTableIndex >= static_cast<int64_t>(rowIndicesPerRound.size())) {
            return result;
        }
        for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index] << ",";

        vector<string> tmp;
        for (int64_t pId : playerId[index]) {
            tmp.push_back(std::to_string(pId));
        }
        commaSeparateList(s, tmp, ";");
        s << ",";

        tmp.clear();
        for (AggressionRole r : role[index]) {
            tmp.push_back(std::to_string(enumAsInt(r)));
        }
        commaSeparateList(s, tmp, ";");
        s << std::endl;
    }

    [[nodiscard]]
    vector<string> getForeignKeyNames() override {
        return {"start tick id", "end tick id", "length"};
    }

    [[nodiscard]]
    vector<string> getOtherColumnNames() override {
        return {"player ids", "roles"};
    }
};


AggressionEventResult queryAggressionRoles(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                           const PlayerAtTick & playerAtTick,
                                           const nav_mesh::nav_file & navFile, const VisPoints & visPoints, const ReachableResult & reachableResult);

#endif //CSKNOW_AGGRESSION_EVENT_H
