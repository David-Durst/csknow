//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_NON_ENGAGEMENT_TRAJECTORY_H
#define CSKNOW_NON_ENGAGEMENT_TRAJECTORY_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
#include "queries/moments/engagement.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

/*
#define STOPPING_PERIOD_SECONDS 1.0
#define STOPPED_AABB_MAX_SIZE_2D PLAYER_WIDTH*PLAYER_WIDTH
#define STOPPED_AABB_HEIGHT PLAYER_HEIGHT
#define STARTED_AABB_MAX_SIZE_2D PLAYER_WIDTH*PLAYER_WIDTH
#define STARTED_AABB_HEIGHT PLAYER_HEIGHT
// https://old.reddit.com/r/GlobalOffensive/comments/a28h8r/movement_speed_chart/
#define STOPPED_SPEED_THRESHOLD 10.0
#define START_SPEED_THRESHOLD 50.0
 */

class NonEngagementTrajectoryResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> startTickId;
    vector<int64_t> endTickId;
    vector<int64_t> tickLength;
    vector<int64_t> playerId;
    IntervalIndex trajectoriesPerTick;

    NonEngagementTrajectoryResult() {
        variableLength = true;
        startTickColumn = 0;
        perEventLengthColumn = 2;
        havePlayerLabels = true;
        playerLabels = {"T"};
        playersToLabelColumn = 0;
        playerLabelIndicesColumn = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index] << ","
            << playerId[index] << "," << 0;

        ss << std::endl;
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


NonEngagementTrajectoryResult queryNonEngagementTrajectory(const Rounds & rounds, const Ticks & ticks,
                                                           const PlayerAtTick & playerAtTick,
                                                           const EngagementResult & engagementResult);

#endif //CSKNOW_NON_ENGAGEMENT_TRAJECTORY_H
