//
// Created by durst on 9/26/22.
//

#ifndef CSKNOW_ENGAGEMENT_AIM_H
#define CSKNOW_ENGAGEMENT_AIM_H

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
#include "queries/moments/engagement.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::array;
using std::map;

constexpr int NUM_TICKS = 6;

class EngagementAimResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> tickId;
    vector<int64_t> engagementId;
    vector<int64_t> attackerPlayerId;
    vector<int64_t> victimPlayerId;
    vector<array<Vec2, NUM_TICKS>> deltaViewAngle;
    vector<array<double, NUM_TICKS>> eyeToHeadDistance;


    EngagementAimResult() {
        startTickColumn = 0;
        ticksPerEvent = 1;
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
        ss << index << "," << tickId[index] << "," << engagementId[index] << ","
           << attackerPlayerId[index] << "," << victimPlayerId[index];

        for (size_t i = 0; i < NUM_TICKS; i++) {
            ss << "," << deltaViewAngle[index][i].x << "," << deltaViewAngle[index][i].y
               << "," << eyeToHeadDistance[index][i];
        }

        ss << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id", "engagement id", "attacker player id", "victim player id"};
    }

    vector<string> getOtherColumnNames() override {
        vector<string> result;
        for (int i = 0; i < NUM_TICKS; i++) {
            result.push_back("delta view angle x (t - " + std::to_string(i) + ")");
            result.push_back("delta view angle y (t - " + std::to_string(i) + ")");
            result.push_back("eye-to-eye distance (t - " + std::to_string(i) + ")");
        }
        return result;
    }
};


EngagementAimResult queryEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick,
                                       const EngagementResult & engagementResult);

#endif //CSKNOW_ENGAGEMENT_AIM_H
