//
// Created by durst on 9/28/22.
//

#include "queries/inference_moments/inference_engagement_aim.h"
#include "queries/rolling_window.h"
#include "file_helpers.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

struct EngagementAimInferenceTickData {
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

void updatePriorData(EngagementAimInferenceTickData & priorData,
                     const TrainingEngagementAimResult & trainingEngagementAimResult,
                     size_t engagementAimId, size_t tickNum) {
    priorData.hitVictim =
        trainingEngagementAimResult.hitVictim[engagementAimId][tickNum];
    priorData.recoilIndex =
        trainingEngagementAimResult.recoilIndex[engagementAimId][tickNum];
    priorData.ticksSinceLastFire =
        trainingEngagementAimResult.ticksSinceLastFire[engagementAimId][tickNum];
    priorData.ticksSinceLastHoldingAttack =
        trainingEngagementAimResult.ticksSinceLastHoldingAttack[engagementAimId][tickNum];
    priorData.victimVisible =
        trainingEngagementAimResult.victimVisible[engagementAimId][tickNum];
    priorData.victimVisibleYet =
        trainingEngagementAimResult.victimVisibleYet[engagementAimId][tickNum];
    priorData.victimAlive =
        trainingEngagementAimResult.victimAlive[engagementAimId][tickNum];
    priorData.attackerEyePos =
        trainingEngagementAimResult.attackerEyePos[engagementAimId][tickNum];
    priorData.victimEyePos =
        trainingEngagementAimResult.victimEyePos[engagementAimId][tickNum];
    priorData.attackerVel =
        trainingEngagementAimResult.attackerVel[engagementAimId][tickNum];
    priorData.victimVel =
        trainingEngagementAimResult.victimVel[engagementAimId][tickNum];
    priorData.idealViewAngle =
        trainingEngagementAimResult.idealViewAngle[engagementAimId][tickNum];
    priorData.deltaRelativeFirstHeadViewAngle =
        trainingEngagementAimResult.deltaRelativeCurHeadViewAngle[engagementAimId][tickNum];
    priorData.scaledRecoilAngle =
        trainingEngagementAimResult.scaledRecoilAngle[engagementAimId][tickNum];
    priorData.victimRelativeFirstHeadMinViewAngle =
        trainingEngagementAimResult.victimRelativeCurHeadMinViewAngle[engagementAimId][tickNum];
    priorData.victimRelativeFirstHeadMaxViewAngle =
        trainingEngagementAimResult.victimRelativeFirstHeadMaxViewAngle[engagementAimId][tickNum];
    priorData.victimRelativeFirstHeadCurHeadViewAngle =
        trainingEngagementAimResult.victimRelativeFirstHeadCurHeadViewAngle[engagementAimId][tickNum];
    priorData.holdingAttack =
        trainingEngagementAimResult.holdingAttack[engagementAimId][tickNum];
}

void InferenceEngagementAimResult::runQuery(const Rounds & rounds, const string & modelsDir,
                                            const EngagementResult & engagementResult) {
    fs::path modelPath = fs::path(modelsDir) / fs::path("engagement_aim_model") /
        fs::path("script_model.pt");

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        size = 0;
        std::cerr << "error loading engagement aim model\n";
        return;
    }

    std::atomic<int64_t> roundsProcessed = 0;

    predictedDeltaRelativeFirstHeadViewAngle.resize(trainingEngagementAimResult.size);
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        auto options = torch::TensorOptions().dtype(at::kFloat);
        // NUM_TICKS stores cur tick and prior ticks in window, shrink by 1 for just prior ticks
        map<int64_t, array<EngagementAimInferenceTickData, PAST_AIM_TICKS>> activeEngagementsPriorTickData;
        for (int64_t engagementAimId = trainingEngagementAimResult.rowIndicesPerRound[roundIndex].minId;
             engagementAimId <= trainingEngagementAimResult.rowIndicesPerRound[roundIndex].maxId;
             engagementAimId++) {
            int64_t engagementId = trainingEngagementAimResult.engagementId[engagementAimId];
            array<EngagementAimInferenceTickData, PAST_AIM_TICKS> &priorData = activeEngagementsPriorTickData[engagementId];

            // add old deltas if in engagement's first tick, otherwise use delta prior earlier predictions
            const RangeIndexEntry &engagementTickRange =
                engagementResult.engagementsPerTick.eventToInterval.at(engagementId);
            if (trainingEngagementAimResult.tickId[engagementAimId] == engagementTickRange.minId) {
                for (size_t priorTickNum = 0; priorTickNum < PAST_AIM_TICKS; priorTickNum++) {
                    updatePriorData(priorData[priorTickNum], trainingEngagementAimResult,
                                    engagementAimId, priorTickNum);
                }
                // only need the last prior data entry's warmup tracker
                priorData[PAST_AIM_TICKS - 1].warmupTicksUsed = 0;
            }

            if (priorData[PAST_AIM_TICKS - 1].warmupTicksUsed >= WARMUP_TICKS) {
                // Create a vector of inputs.
                std::vector<torch::jit::IValue> inputs;
                std::vector<float> rowCPP;
                // all but cur tick are inputs
                // seperate different input types
                for (size_t priorDeltaNum = 0; priorDeltaNum < priorData.size(); priorDeltaNum++) {
                    rowCPP.push_back(static_cast<float>(boolToInt(priorData[priorDeltaNum].hitVictim)));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].recoilIndex));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].ticksSinceLastFire));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].ticksSinceLastHoldingAttack));
                    rowCPP.push_back(static_cast<float>(boolToInt(priorData[priorDeltaNum].victimVisible)));
                    rowCPP.push_back(static_cast<float>(boolToInt(priorData[priorDeltaNum].victimVisibleYet)));
                    rowCPP.push_back(static_cast<float>(boolToInt(priorData[priorDeltaNum].victimAlive)));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].attackerEyePos.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].attackerEyePos.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].attackerEyePos.z));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimEyePos.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimEyePos.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimEyePos.z));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].attackerVel.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].attackerVel.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].attackerVel.z));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimVel.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimVel.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimVel.z));
                }
                for (size_t priorDeltaNum = 0; priorDeltaNum < priorData.size(); priorDeltaNum++) {
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].idealViewAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaRelativeFirstHeadViewAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].scaledRecoilAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimRelativeFirstHeadMinViewAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimRelativeFirstHeadMaxViewAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimRelativeFirstHeadCurHeadViewAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].idealViewAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaRelativeFirstHeadViewAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].scaledRecoilAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimRelativeFirstHeadMinViewAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimRelativeFirstHeadMaxViewAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].victimRelativeFirstHeadCurHeadViewAngle.y));
                }
                for (size_t priorDeltaNum = 0; priorDeltaNum < priorData.size(); priorDeltaNum++) {
                    rowCPP.push_back(static_cast<float>(boolToInt(priorData[priorDeltaNum].holdingAttack)));
                }
                rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.weaponType[engagementAimId]));
                torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())},
                                                       options).clone();
                inputs.push_back(rowPT);

                // Execute the model and turn its output into a tensor.
                at::Tensor output = module.forward(inputs).toTuple()->elements()[1].toTensor();
                predictedDeltaRelativeFirstHeadViewAngle[engagementAimId] = {
                        static_cast<double>(output[0][0].item<float>()),
                        static_cast<double>(output[0][output[0].size(0) / 2].item<float>())
                };
            }
            else {
                predictedDeltaRelativeFirstHeadViewAngle[engagementAimId] =
                        trainingEngagementAimResult.deltaRelativeFirstHeadViewAngle[engagementAimId][PAST_AIM_TICKS];
            }

            // if last tick for engagement, remove it from actives. Otherwise rotate the current prediction into prior deltas
            if (trainingEngagementAimResult.tickId[engagementAimId] == engagementTickRange.maxId) {
                activeEngagementsPriorTickData.erase(engagementId);
            } else {
                for (size_t priorDeltaNum = 1; priorDeltaNum < PAST_AIM_TICKS; priorDeltaNum++) {
                    priorData[priorDeltaNum - 1] = priorData[priorDeltaNum];
                }
                updatePriorData(priorData[PAST_AIM_TICKS - 1], trainingEngagementAimResult,
                                engagementAimId, PAST_AIM_TICKS);
                priorData[PAST_AIM_TICKS - 1].deltaRelativeFirstHeadViewAngle =
                    predictedDeltaRelativeFirstHeadViewAngle.back();
                priorData[PAST_AIM_TICKS - 1].warmupTicksUsed++;
            }
        }
        roundsProcessed++;
        printProgress(roundsProcessed, rounds.size);
    }
    size = trainingEngagementAimResult.size;
}
