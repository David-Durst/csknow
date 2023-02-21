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
#include "queries/moments/fire_history.h"
#include "queries/nearest_nav_cell.h"

using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::array;
using std::map;

constexpr int PAST_AIM_TICKS = 13;
constexpr int FUTURE_AIM_TICKS = 13;
constexpr int CUR_AIM_TICK = 1;
constexpr int TOTAL_AIM_TICKS = PAST_AIM_TICKS + FUTURE_AIM_TICKS + CUR_AIM_TICK;

const int64_t MAX_TICKS_SINCE_LAST_FIRE_ATTACK = 100;

enum class AimWeaponType {
    Pistol = 0,
    SMG,
    Heavy,
    AK,
    M4A1,
    AROther,
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
    vector<array<bool, TOTAL_AIM_TICKS>> holdingAttack;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksSinceLastFire;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksSinceLastHoldingAttack;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksUntilNextFire;
    vector<array<int64_t, TOTAL_AIM_TICKS>> ticksUntilNextHoldingAttack;
    vector<array<bool, TOTAL_AIM_TICKS>> victimVisible;
    vector<array<bool, TOTAL_AIM_TICKS>> victimVisibleYet;
    vector<array<bool, TOTAL_AIM_TICKS>> victimAlive;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimMinViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimMaxViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimCurHeadViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeFirstHeadMinViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeFirstHeadMaxViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeFirstHeadCurHeadViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeCurHeadMinViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeCurHeadMaxViewAngle;
    vector<array<Vec2, TOTAL_AIM_TICKS>> victimRelativeCurHeadCurHeadViewAngle;
    vector<array<Vec3, TOTAL_AIM_TICKS>> attackerEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> victimEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> deltaEyePos;
    vector<array<Vec3, TOTAL_AIM_TICKS>> attackerVel;
    vector<array<Vec3, TOTAL_AIM_TICKS>> victimVel;
    vector<array<float, TOTAL_AIM_TICKS>> attackerDuckAmount;
    vector<array<float, TOTAL_AIM_TICKS>> attackerNextPrimaryAttack;
    vector<array<float, TOTAL_AIM_TICKS>> attackerNextSecondaryAttack;
    vector<array<float, TOTAL_AIM_TICKS>> attackerGameTime;
    vector<array<DemoEquipmentType, TOTAL_AIM_TICKS>> weaponId;
    vector<AimWeaponType> weaponType;


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

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << index << "," << roundId[index] << "," << tickId[index] << ","
          << demoTickId[index] << "," << gameTickId[index] << "," << gameTime[index] << ","
          << engagementId[index] << "," << attackerPlayerId[index] << "," << victimPlayerId[index];

        for (size_t i = 0; i < TOTAL_AIM_TICKS; i++) {
            s << "," << attackerViewAngle[index][i].toCSV() << "," << idealViewAngle[index][i].toCSV()
              << "," << deltaRelativeFirstHeadViewAngle[index][i].toCSV()
              << "," << deltaRelativeCurHeadViewAngle[index][i].toCSV()
              << "," << boolToInt(hitVictim[index][i])
              << "," << recoilIndex[index][i]
               << "," << scaledRecoilAngle[index][i].toCSV()
               << "," << boolToInt(holdingAttack[index][i])
               << "," << ticksSinceLastFire[index][i] << "," << ticksSinceLastHoldingAttack[index][i]
               << "," << ticksUntilNextFire[index][i] << "," << ticksUntilNextHoldingAttack[index][i]
               << "," << boolToInt(victimVisible[index][i])
               << "," << boolToInt(victimVisibleYet[index][i])
               << "," << boolToInt(victimAlive[index][i])
               << "," << victimMinViewAngle[index][i].toCSV()
               << "," << victimMaxViewAngle[index][i].toCSV()
               << "," << victimCurHeadViewAngle[index][i].toCSV()
               << "," << victimRelativeFirstHeadMinViewAngle[index][i].toCSV()
               << "," << victimRelativeFirstHeadMaxViewAngle[index][i].toCSV()
               << "," << victimRelativeFirstHeadCurHeadViewAngle[index][i].toCSV()
               << "," << victimRelativeCurHeadMinViewAngle[index][i].toCSV()
               << "," << victimRelativeCurHeadMaxViewAngle[index][i].toCSV()
               << "," << victimRelativeCurHeadCurHeadViewAngle[index][i].toCSV()
               << "," << attackerEyePos[index][i].toCSV()
               << "," << victimEyePos[index][i].toCSV()
               << "," << deltaEyePos[index][i].toCSV()
               << "," << attackerVel[index][i].toCSV()
               << "," << victimVel[index][i].toCSV()
               << "," << attackerDuckAmount[index][i]
               << "," << attackerNextPrimaryAttack[index][i]
               << "," << attackerNextSecondaryAttack[index][i]
               << "," << attackerGameTime[index][i]
               << "," << enumAsInt(weaponId[index][i]);
        }

        s << "," << enumAsInt(weaponType[index]);

        s << std::endl;
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
            result.push_back("holding attack (t"+ toSignedIntString(i, true) + ")");
            result.push_back("ticks since last fire (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks since last holding attack (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks until next fire (t" + toSignedIntString(i, true) + ")");
            result.push_back("ticks until next holding attack (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim visible (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim visible yet (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim alive (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim min view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim min view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim max view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim max view angle y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim cur head view angle x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim cur head view angle y (t" + toSignedIntString(i, true) + ")");
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
            result.push_back("delta eye pos x (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta eye pos y (t" + toSignedIntString(i, true) + ")");
            result.push_back("delta eye pos z (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker vel x (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker vel y (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker vel z (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim vel x (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim vel y (t" + toSignedIntString(i, true) + ")");
            result.push_back("victim vel z (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker duck amount (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker next primary attack (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker next secondary attack (t" + toSignedIntString(i, true) + ")");
            result.push_back("attacker game time (t" + toSignedIntString(i, true) + ")");
            result.push_back("weapon id (t" + toSignedIntString(i, true) + ")");
        }
        result.push_back("weapon type");
        return result;
    }

    void toHDF5Inner(HighFive::File & file) override {

        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(roundId.size()));

        file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/demo tick id", demoTickId, hdf5FlatCreateProps);
        file.createDataSet("/data/game tick id", gameTickId, hdf5FlatCreateProps);
        file.createDataSet("/data/game time", gameTime, hdf5FlatCreateProps);
        file.createDataSet("/data/engagement id", engagementId, hdf5FlatCreateProps);
        file.createDataSet("/data/attacker player id", attackerPlayerId, hdf5FlatCreateProps);
        file.createDataSet("/data/victim player id", victimPlayerId, hdf5FlatCreateProps);

        HighFive::DataSetCreateProps hdf5NestedCreateProps;
        hdf5NestedCreateProps.add(HighFive::Deflate(6));
        hdf5NestedCreateProps.add(HighFive::Chunking({roundId.size(), PAST_AIM_TICKS + CUR_AIM_TICK + FUTURE_AIM_TICKS}));
        //file.createDataSet("/data/attacker duck amount", attackerDuckAmount, hdf5NestedCreateProps);
        int startOffset = -1 * PAST_AIM_TICKS;
        saveTemporalArrayOfVec2VectorsToHDF5(attackerViewAngle, file, startOffset,
                                             "attacker view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(idealViewAngle, file, startOffset,
                                             "ideal view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(deltaRelativeFirstHeadViewAngle, file, startOffset,
                                             "delta relative first head view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(deltaRelativeCurHeadViewAngle, file, startOffset,
                                             "delta relative cur head view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(hitVictim, file, startOffset,
                                         "hit victim", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(recoilIndex, file, startOffset,
                                         "recoil index", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(scaledRecoilAngle, file, startOffset,
                                             "scaled recoil angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(holdingAttack, file, startOffset,
                                         "holding attack", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(ticksSinceLastFire, file, startOffset,
                                         "ticks since last fire", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(ticksSinceLastHoldingAttack, file, startOffset,
                                         "ticks since last holding attack", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(ticksUntilNextFire, file, startOffset,
                                         "ticks until next fire", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(ticksUntilNextHoldingAttack, file, startOffset,
                                         "ticks until next holding attack", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(victimVisible, file, startOffset,
                                         "victim visible", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(victimVisibleYet, file, startOffset,
                                         "victim visible yet", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(victimAlive, file, startOffset,
                                         "victim alive", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimMinViewAngle, file, startOffset,
                                             "victim min view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimMaxViewAngle, file, startOffset,
                                             "victim max view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimCurHeadViewAngle, file, startOffset,
                                             "victim cur head view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimRelativeFirstHeadMinViewAngle, file, startOffset,
                                             "victim relative first head min view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimRelativeFirstHeadMaxViewAngle, file, startOffset,
                                             "victim relative first head max view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimRelativeFirstHeadCurHeadViewAngle, file, startOffset,
                                             "victim relative first head cur head view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimRelativeCurHeadMinViewAngle, file, startOffset,
                                             "victim relative cur head min view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimRelativeCurHeadMaxViewAngle, file, startOffset,
                                             "victim relative cur head max view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec2VectorsToHDF5(victimRelativeCurHeadCurHeadViewAngle, file, startOffset,
                                             "victim relative cur head cur head view angle", hdf5NestedCreateProps);
        saveTemporalArrayOfVec3VectorsToHDF5(attackerEyePos, file, startOffset,
                                             "attacker eye pos", hdf5NestedCreateProps);
        saveTemporalArrayOfVec3VectorsToHDF5(victimEyePos, file, startOffset,
                                             "victim eye pos", hdf5NestedCreateProps);
        saveTemporalArrayOfVec3VectorsToHDF5(deltaEyePos, file, startOffset,
                                             "delta eye pos", hdf5NestedCreateProps);
        saveTemporalArrayOfVec3VectorsToHDF5(attackerVel, file, startOffset,
                                             "attacker vel", hdf5NestedCreateProps);
        saveTemporalArrayOfVec3VectorsToHDF5(victimVel, file, startOffset,
                                             "victim vel", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(attackerDuckAmount, file, startOffset,
                                         "attacker duck amount", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(attackerNextPrimaryAttack, file, startOffset,
                                         "attacker next primary attack", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(attackerNextSecondaryAttack, file, startOffset,
                                         "attacker next secondary attack", hdf5NestedCreateProps);
        saveTemporalArrayOfVectorsToHDF5(attackerGameTime, file, startOffset,
                                         "attacker game time", hdf5NestedCreateProps);
        saveTemporalVectorOfEnumsToHDF5(weaponId, file, startOffset,
                                        "weapon id", hdf5NestedCreateProps);

        file.createDataSet("/data/weapon type", vectorOfEnumsToVectorOfInts(weaponType), hdf5FlatCreateProps);
        //file.createDataSet<int64_t>("id", )
        //return {"round id", "tick id", "demo tick id", "game tick id", "game time",
        //        "engagement id", "attacker player id", "victim player id"};
        //throw std::runtime_error("HDFS saving not implemented for this query yet");
    }
};


TrainingEngagementAimResult queryTrainingEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                       const PlayerAtTick & playerAtTick,
                                                       const EngagementResult & engagementResult,
                                                       const csknow::fire_history::FireHistoryResult & fireHistoryResult,
                                                       const VisPoints & visPoints,
                                                       const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell,
                                                       bool parallelize = true);

#endif //CSKNOW_TRAINING_ENGAGEMENT_AIM_H
