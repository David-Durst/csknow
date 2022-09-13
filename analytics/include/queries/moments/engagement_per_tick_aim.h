//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_ENGAGEMENT_PER_TICK_AIM_H
#define CSKNOW_ENGAGEMENT_PER_TICK_AIM_H
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
using std::map;

class EngagementPerTickAimResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> tickId;
    vector<int64_t> engagementId;
    vector<int64_t> attackerPlayerId;
    vector<int64_t> victimPlayerId;
    vector<double> secondsToHit;
    vector<Vec2> deltaViewAngle;
    vector<double> rawViewAngleSpeed;
    vector<int64_t> gameTickNumber;


    EngagementPerTickAimResult() {
        this->startTickColumn = 0;
        this->eventIdColumn = 1;
        this->ticksPerEvent = 1;
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
        ss << index << "," << tickId[index] << "," << engagementId[index] << ","
            << attackerPlayerId[index] << "," << victimPlayerId[index] << "," << secondsToHit[index] << ","
            << deltaViewAngle[index].x << "," << deltaViewAngle[index].y << "," << rawViewAngleSpeed[index] << ","
            << gameTickNumber[index];
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "engagement id", "attacker player id", "victim player id"};
    }

    vector<string> getOtherColumnNames() {
        return {"seconds to hit",  "delta view angle x", "delta view angle y", "raw view angle speed", "game tick number"};
    }
};


EngagementPerTickAimResult queryEngagementPerTickAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick, const WeaponFire & first, const Hurt & hurt,
                                       const EngagementResult & engagementResult);

#endif //CSKNOW_ENGAGEMENT_PER_TICK_AIM_H
