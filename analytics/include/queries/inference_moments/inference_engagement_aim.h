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

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << index << "," << trainingEngagementAimResult.tickId[index] << ","
            << trainingEngagementAimResult.engagementId[index] << ","
            << trainingEngagementAimResult.attackerPlayerId[index] << ","
            << trainingEngagementAimResult.victimPlayerId[index] << ","
            << predictedDeltaRelativeFirstHeadViewAngle[index].x << ","
            << predictedDeltaRelativeFirstHeadViewAngle[index].y;

        ss << std::endl;
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
