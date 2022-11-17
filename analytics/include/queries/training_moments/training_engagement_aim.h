//
// Created by durst on 9/26/22.
//

#ifndef CSKNOW_TRAINING_ENGAGEMENT_AIM_H
#define CSKNOW_TRAINING_ENGAGEMENT_AIM_H

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
#include "queries/moments/engagement_per_tick_aim.h"
#include "queries/moments/fire_history.h"

using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::array;
using std::map;

constexpr int PAST_AIM_TICKS = 12;
constexpr int FUTURE_AIM_TICKS = 6;
constexpr int CUR_AIM_TICK = 1;
constexpr int TOTAL_AIM_TICKS = PAST_AIM_TICKS + FUTURE_AIM_TICKS + CUR_AIM_TICK;

enum class AimWeaponType {
    Pistol = 0,
    SMG,
    Heavy,
    AR,
    Sniper,
    Unknown,
    AIM_WEAPON_TYPE_COUNT [[maybe_unused]]
};

class TrainingEngagementAimResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> roundId;
    vector<int64_t> tickId;
    vector<int64_t> demoTickId;
    vector<int64_t> gameTickId;
    vector<int64_t> engagementId;
    vector<int64_t> attackerPlayerId;
    vector<int64_t> victimPlayerId;
    vector<array<Vec2, TOTAL_AIM_TICKS>> attackerViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> idealViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> deltaRelativeFirstHitHeadViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> deltaRelativeCurHeadViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> recoilAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> deltaRelativeCurHeadRecoilAdjustedViewAngle;
    vector<array<int16_t, TOTAL_AIM_TICKS>> numShotsFired;
    vector<array<int16_t, TOTAL_AIM_TICKS>> ticksSinceLastFire;
    vector<array<int16_t, TOTAL_AIM_TICKS>> ticksSinceLastHoldingAttack;
    vector<array<int16_t, TOTAL_AIM_TICKS>> ticksUntilNextFile;
    vector<array<int16_t, TOTAL_AIM_TICKS>> ticksUntilNextHoldingAttack;
    vector<array<bool, TOTAL_AIM_TICKS>> enemyVisible;
    vector<array<Vec2, TOTAL_AIM_TICKS>> enemyMinViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> enemyMaxViewAngle;
    vector<array<Vec3, TOTAL_AIM_TICKS>> attackerEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> victimEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> attackerVel;
    vector<array<Vec3, TOTAL_AIM_TICKS>> victimVel;
    vector<array<double, TOTAL_AIM_TICKS>> eyeToHeadDistance;
    vector<AimWeaponType> weaponType;
    vector<double> distanceNormalization;


    TrainingEngagementAimResult() {
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
        ss << index << "," << roundId[index] << "," << tickId[index] << ","
           << demoTickId[index] << "," << gameTickId[index] << ","
           << engagementId[index] << "," << attackerPlayerId[index] << "," << victimPlayerId[index] << ",";

        ss << numShotsFired[index] << "," << ticksSinceLastFire[index] << ","
           << lastShotFiredTickId[index];

        for (size_t i = 0; i < TOTAL_AIM_TICKS; i++) {
            ss << "," << deltaViewAngle[index][i].x << "," << deltaViewAngle[index][i].y
               << "," << recoilAngle[index][i].x << "," << recoilAngle[index][i].y
               << "," << deltaViewAngleRecoilAdjusted[index][i].x << "," << deltaViewAngleRecoilAdjusted[index][i].y
               << "," << deltaPosition[index][i].x << "," << deltaPosition[index][i].y << "," << deltaPosition[index][i].z
               << "," << eyeToHeadDistance[index][i];
        }

        ss << "," << enumAsInt(weaponType[index]);

        ss << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"round id", "tick id", "demo tick id", "game tick id",
                "engagement id", "attacker player id", "victim player id"};
    }

    vector<string> getOtherColumnNames() override {
        vector<string> result;
        result.push_back("num shots fired");
        result.push_back("ticks since last fire");
        result.push_back("last fire tick id");
        for (int i = -1*PAST_AIM_TICKS; i <= FUTURE_AIM_TICKS; i++) {
            result.push_back("delta view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("recoil angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("recoil angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta view angle recoil adjusted x (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta view angle recoil adjusted y (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta position x (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta position y (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta position z (t" + toSignedIntString(i, true) + ")");
            result.push_back("eye-to-head distance (t" + toSignedIntString(i, true) + ")");
        }
        result.push_back("weapon type");
        return result;
    }

    void analyzeRollingWindowDifferences(const Rounds & rounds, const Ticks & ticks,
                                         const EngagementPerTickAimResult & engagementPerTickAimResult) const {
        int64_t numDifferencesDueToWindowRange = 0;
        for (int64_t i = 0, j = 0; i < engagementPerTickAimResult.size; i++) {
            if (engagementPerTickAimResult.tickId[i] == tickId[j]) {
                j++;
            }
            else {
                int64_t distanceToRoundEnd = rounds.endTick[ticks.roundId[engagementPerTickAimResult.tickId[i]]] -
                    engagementPerTickAimResult.tickId[i];
#if false
                std::cout << "per tick i " << i << " per tick aim tick id " << engagementPerTickAimResult.tickId[i]
                          << " per tick aim engagmenet id " << engagementPerTickAimResult.engagementId[i]
                          << " distance to round end " << distanceToRoundEnd
                          << " training j " << j
                          << " training tick id " << tickId[j]
                          << " training engagement id " << engagementId[j] << std::endl;
#endif //false
                if (distanceToRoundEnd < FUTURE_AIM_TICKS) {
                    numDifferencesDueToWindowRange++;
                }
            }
        }
        std::cout << "missing ticks: " << engagementPerTickAimResult.size - size
            << " missing ticks due to end of round/window range " << numDifferencesDueToWindowRange << std::endl;
    }
};


TrainingEngagementAimResult queryTrainingEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                       const PlayerAtTick & playerAtTick, const WeaponFire & weaponFire,
                                                       const EngagementResult & engagementResult,
                                                       const csknow::fire_history::FireHistoryResult & fireHistoryResult);

#endif //CSKNOW_TRAINING_ENGAGEMENT_AIM_H
