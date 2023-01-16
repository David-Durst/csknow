//
// Created by durst on 9/28/22.
//

#ifndef CSKNOW_INFERENCE_ENGAGEMENT_AIM_H
#define CSKNOW_INFERENCE_ENGAGEMENT_AIM_H

#include "queries/training_moments/training_engagement_aim.h"

constexpr int WARMUP_TICKS = 0;

struct EngagementAimTickData {
    // general float encoded
    bool hitVictim;
    float recoilIndex;
    int64_t ticksSinceLastFire;
    int64_t ticksSinceLastHoldingAttack;
    bool victimVisible;
    bool victimVisibleYet;
    bool victimAlive;
    Vec3 attackerEyePos;
    Vec3 victimEyePos;
    Vec3 attackerVel;
    Vec3 victimVel;
    // angle encoded
    Vec2 idealViewAngle;
    Vec2 deltaRelativeFirstHeadViewAngle;
    Vec2 scaledRecoilAngle;
    Vec2 victimRelativeFirstHeadMinViewAngle;
    Vec2 victimRelativeFirstHeadMaxViewAngle;
    Vec2 victimRelativeFirstHeadCurHeadViewAngle;
    bool holdingAttack;
    int warmupTicksUsed;
    // extras
    Vec2 attackerViewAngle;
    Vec2 deltaRelativeCurHeadViewAngle;

    string toString() const {
        std::ostringstream ss;
        ss << "hit victim: " << boolToString(hitVictim)
            << ", recoil index: " << recoilIndex
            << ", ticks since last fire: " << ticksSinceLastFire
            << ", ticks since last holding attack: " << ticksSinceLastHoldingAttack
            << ", victim visible: " << boolToString(victimVisible)
            << ", victim visible yet: " << boolToString(victimVisibleYet)
            << ", victim alive: " << boolToString(victimAlive)
            << ", attacker eye pos: " << attackerEyePos.toString()
            << ", victim eye pos: " << victimEyePos.toString()
            << ", attacker vel: " << attackerVel.toString()
            << ", victim vel: " << victimVel.toString()
            << ", ideal view angle: " << idealViewAngle.toString()
            << ", delta relative first head view angle: " << deltaRelativeFirstHeadViewAngle.toString()
            << ", scaled recoil angle: " << scaledRecoilAngle.toString()
            << ", victim relative first head min view angle: " << victimRelativeFirstHeadMinViewAngle.toString()
            << ", victim relative first head max view angle: " << victimRelativeFirstHeadMaxViewAngle.toString()
            << ", victim relative first head cur head view angle: " << victimRelativeFirstHeadCurHeadViewAngle.toString()
            << ", holding attack: " << boolToString(holdingAttack)
            << ", attacker view angle: " << attackerViewAngle.toString()
            << ", delta relative cur head view angle: " << deltaRelativeCurHeadViewAngle.toString()
            << ", warmup ticks used: " << warmupTicksUsed;
        return ss.str();
    }
    
    string toCSV(bool header) const {
        std::ostringstream ss;
        if (header) {
            ss << "hit victim"
               << ",recoil index"
               << ",ticks since last fire"
               << ",ticks since last holding attack"
               << ",victim visible"
               << ",victim visible yet"
               << ",victim alive"
               << ",attacker eye pos x,attacker eye pos y,attacker eye pos z"
               << ",victim eye pos x,victim eye pos y,victim eye pos z"
               << ",attacker vel x,attacker vel y,attacker vel z"
               << ",victim vel x,victim vel y,victim vel z"
               << ",ideal view angle x,ideal view angle y"
               << ",delta relative first head view angle x,delta relative first head view angle y"
               << ",scaled recoil angle x,scaled recoil angle y"
               << ",victim relative first head min view angle x,victim relative first head min view angle y"
               << ",victim relative first head max view angle x,victim relative first head max view angle y"
               << ",victim relative first head cur head view angle x,victim relative first head cur head view angle y"
               << ",holding attack"
               << ",attacker view angle x,attacker view angle y"
               << ",delta relative cur head view angle x,delta relative cur head view angle y"
               << ",warmup ticks used"
               << std::endl;
        }
        ss << boolToString(hitVictim)
           << "," << recoilIndex
           << "," << ticksSinceLastFire
           << "," << ticksSinceLastHoldingAttack
           << "," << boolToString(victimVisible)
           << "," << boolToString(victimVisibleYet)
           << "," << boolToString(victimAlive)
           << "," << attackerEyePos.toCSV()
           << "," << victimEyePos.toCSV()
           << "," << attackerVel.toCSV()
           << "," << victimVel.toCSV()
           << "," << idealViewAngle.toCSV()
           << "," << deltaRelativeFirstHeadViewAngle.toCSV()
           << "," << scaledRecoilAngle.toCSV()
           << "," << victimRelativeFirstHeadMinViewAngle.toCSV()
           << "," << victimRelativeFirstHeadMaxViewAngle.toCSV()
           << "," << victimRelativeFirstHeadCurHeadViewAngle.toCSV()
           << "," << boolToString(holdingAttack)
           << "," << attackerViewAngle.toCSV()
           << "," << deltaRelativeCurHeadViewAngle.toCSV()
           << "," << warmupTicksUsed;
        return ss.str();
    }
};


class InferenceEngagementAimResult : public QueryResult {
public:
    const TrainingEngagementAimResult & trainingEngagementAimResult;
    vector<Vec2> predictedDeltaRelativeFirstHeadViewAngle;


    explicit InferenceEngagementAimResult(const TrainingEngagementAimResult & trainingEngagementAimResult) :
        trainingEngagementAimResult(trainingEngagementAimResult) {
        startTickColumn = 0;
        eventIdColumn = 1;
        ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        if (size == 0) {
            return {};
        }
        vector<int64_t> result;
        for (int64_t i = trainingEngagementAimResult.rowIndicesPerRound[otherTableIndex].minId;
             i <= trainingEngagementAimResult.rowIndicesPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << index << "," << trainingEngagementAimResult.tickId[index] << ","
          << trainingEngagementAimResult.engagementId[index] << ","
          << trainingEngagementAimResult.attackerPlayerId[index] << ","
          << trainingEngagementAimResult.victimPlayerId[index] << ","
          << predictedDeltaRelativeFirstHeadViewAngle[index].x << ","
          << predictedDeltaRelativeFirstHeadViewAngle[index].y;

        s << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id", "engagement id", "attacker player id", "victim player id"};
    }

    vector<string> getOtherColumnNames() override {
        return {"predicted delta view angle x", "predicted delta view angle y"};
    }

    void runQuery(const Rounds & rounds, const string & modelsDir, const EngagementResult & engagementResult);
};

#endif //CSKNOW_INFERENCE_ENGAGEMENT_AIM_H
