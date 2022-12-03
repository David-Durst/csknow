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

const int64_t MAX_TICKS_SINCE_LAST_FIRE_ATTACK = 100;

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
    vector<int64_t> gameTime;
    vector<int64_t> engagementId;
    vector<int64_t> attackerPlayerId;
    vector<int64_t> victimPlayerId;
    vector<array<Vec2, TOTAL_AIM_TICKS>> attackerViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> idealViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> deltaRelativeFirstHeadViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> deltaRelativeCurHeadViewAngle;
    vector<array<bool, TOTAL_AIM_TICKS>> hitVictim;
    vector<array<float, TOTAL_AIM_TICKS>> recoilIndex;
    vector<array<Vec2, TOTAL_AIM_TICKS>> scaledRecoilAngle;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksSinceLastFire;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksSinceLastHoldingAttack;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksUntilNextFire;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksUntilNextHoldingAttack;
    vector<array<bool, TOTAL_AIM_TICKS>> victimVisible;
    vector<array<bool, TOTAL_AIM_TICKS>> victimAlive;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeFirstHeadMinViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeFirstHeadMaxViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeFirstHeadCurHeadViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeCurHeadMinViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeCurHeadMaxViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeCurHeadCurHeadViewAngle;
    vector<array<Vec3, TOTAL_AIM_TICKS>> attackerEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> victimEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> attackerVel;
    vector<array<Vec3, TOTAL_AIM_TICKS>> victimVel;
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
           << demoTickId[index] << "," << gameTickId[index] << "," << gameTime[index] << ","
           << engagementId[index] << "," << attackerPlayerId[index] << "," << victimPlayerId[index];

        for (size_t i = 0; i < TOTAL_AIM_TICKS; i++) {
            ss << "," << attackerViewAngle[index][i].toCSV() << "," << idealViewAngle[index][i].toCSV()
               << "," << deltaRelativeFirstHeadViewAngle[index][i].toCSV()
               << "," << deltaRelativeCurHeadViewAngle[index][i].toCSV()
               << "," << boolToInt(hitVictim[index][i])
               << "," << recoilIndex[index][i]
               << "," << scaledRecoilAngle[index][i].toCSV()
               << "," << ticksSinceLastFire[index][i] << "," << ticksSinceLastHoldingAttack[index][i]
               << "," << ticksUntilNextFire[index][i] << "," << ticksUntilNextHoldingAttack[index][i]
               << "," << boolToInt(victimVisible[index][i])
               << "," << boolToInt(victimAlive[index][i])
               << "," << victimRelativeFirstHeadMinViewAngle[index][i].toCSV()
               << "," << victimRelativeFirstHeadMaxViewAngle[index][i].toCSV()
               << "," << victimRelativeFirstHeadCurHeadViewAngle[index][i].toCSV()
               << "," << victimRelativeCurHeadMinViewAngle[index][i].toCSV()
               << "," << victimRelativeCurHeadMaxViewAngle[index][i].toCSV()
               << "," << victimRelativeCurHeadCurHeadViewAngle[index][i].toCSV()
               << "," << attackerEyePos[index][i].toCSV()
               << "," << victimEyePos[index][i].toCSV()
               << "," << attackerVel[index][i].toCSV()
               << "," << victimVel[index][i].toCSV();
        }

        ss << "," << enumAsInt(weaponType[index]);

        ss << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"round id", "tick id", "demo tick id", "game tick id", "game time",
                "engagement id", "attacker player id", "victim player id"};
    }

    vector<string> getOtherColumnNames() override {
        vector<string> result;
        for (int i = -1*PAST_AIM_TICKS; i <= FUTURE_AIM_TICKS; i++) {
            result.push_back("attacker view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("ideal view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("ideal view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta relative first head view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta relative first head view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta relative cur head view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta relative cur head view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("hit victim (t"+ toSignedIntString(i, true) + ")");
            result.push_back("recoil index (t" + toSignedIntString(i, true) + ")");
            result.push_back("scaled recoil angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("scaled recoil angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks since last fire (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks since last holding attack (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks until next fire (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks until next holding attack (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim visible (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim alive (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative first head min view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative first head min view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative first head max view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative first head max view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative first head cur head view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative first head cur head view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative cur head min view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative cur head min view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative cur head max view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative cur head max view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative cur head cur head view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim relative cur head cur head view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker eye pos x (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker eye pos y (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker eye pos z (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim eye pos x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim eye pos y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim eye pos z (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker vel x (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker vel y (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker vel z (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim vel x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim vel y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim vel z (t" + toSignedIntString(i, true) + ")");
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
                                                       const PlayerAtTick & playerAtTick,
                                                       const EngagementResult & engagementResult,
                                                       const csknow::fire_history::FireHistoryResult & fireHistoryResult,
                                                       const VisPoints & visPoints);

#endif //CSKNOW_TRAINING_ENGAGEMENT_AIM_H
