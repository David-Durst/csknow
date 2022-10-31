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
    Vec2 deltaViewAngle;
    Vec2 recoilAngle;
    Vec2 deltaViewAngleRecoilAdjusted;
    Vec3 deltaPosition;
    double eyeToHeadDistance;
    int warmupTicksUsed;
};

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

    predictedDeltaViewAngle.resize(trainingEngagementAimResult.size);
    normalizedPredictedDeltaViewAngle.resize(trainingEngagementAimResult.size);
#pragma omp parallel for
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
                    priorData[priorTickNum].deltaViewAngle =
                        trainingEngagementAimResult.deltaViewAngle[engagementAimId][priorTickNum];
                    priorData[priorTickNum].recoilAngle =
                        trainingEngagementAimResult.recoilAngle[engagementAimId][priorTickNum];
                    priorData[priorTickNum].deltaViewAngleRecoilAdjusted =
                        trainingEngagementAimResult.deltaViewAngleRecoilAdjusted[engagementAimId][priorTickNum];
                    priorData[priorTickNum].eyeToHeadDistance =
                        trainingEngagementAimResult.eyeToHeadDistance[engagementAimId][priorTickNum];
                    priorData[priorTickNum].deltaPosition =
                        trainingEngagementAimResult.deltaPosition[engagementAimId][priorTickNum];
                }
                // only need the last prior data entry's warmup tracker
                priorData[PAST_AIM_TICKS - 1].warmupTicksUsed = 0;
            }

            if (priorData[PAST_AIM_TICKS - 1].warmupTicksUsed >= WARMUP_TICKS) {
                // Create a vector of inputs.
                std::vector<torch::jit::IValue> inputs;
                std::vector<float> rowCPP;
                // all but cur tick are inputs
                for (size_t priorDeltaNum = 0; priorDeltaNum < priorData.size(); priorDeltaNum++) {
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaViewAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaViewAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].recoilAngle.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].recoilAngle.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaViewAngleRecoilAdjusted.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaViewAngleRecoilAdjusted.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaPosition.x));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaPosition.y));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].deltaPosition.z));
                    rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].eyeToHeadDistance));
                }
                rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.numShotsFired[engagementAimId]));
                rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.ticksSinceLastFire[engagementAimId]));
                rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.weaponType[engagementAimId]));
                torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())},
                                                       options).clone();
                inputs.push_back(rowPT);

                // Execute the model and turn its output into a tensor.
                at::Tensor output = module.forward(inputs).toTensor();
                predictedDeltaViewAngle[engagementAimId] = {
                        static_cast<double>(output[0][output[0].size(0) / 2].item<float>()),
                        static_cast<double>(output[0][output[0].size(0) / 2 + 1].item<float>())
                };
            }
            else {
                predictedDeltaViewAngle[engagementAimId] =
                        trainingEngagementAimResult.deltaViewAngle[engagementAimId][PAST_AIM_TICKS];
            }
            normalizedPredictedDeltaViewAngle[engagementAimId] = predictedDeltaViewAngle[engagementAimId] /
                trainingEngagementAimResult.distanceNormalization[engagementAimId];

            // if last tick for engagement, remove it from actives. Otherwise rotate the current prediction into prior deltas
            if (trainingEngagementAimResult.tickId[engagementAimId] == engagementTickRange.maxId) {
                activeEngagementsPriorTickData.erase(engagementId);
            } else {
                for (size_t priorDeltaNum = 1; priorDeltaNum < PAST_AIM_TICKS; priorDeltaNum++) {
                    priorData[priorDeltaNum - 1] = priorData[priorDeltaNum];
                }
                priorData[PAST_AIM_TICKS - 1].deltaViewAngle = predictedDeltaViewAngle.back();
                priorData[PAST_AIM_TICKS - 1].recoilAngle =
                    trainingEngagementAimResult.recoilAngle[engagementAimId][PAST_AIM_TICKS];
                priorData[PAST_AIM_TICKS - 1].deltaViewAngleRecoilAdjusted =
                    priorData[PAST_AIM_TICKS - 1].deltaViewAngle +
                    priorData[PAST_AIM_TICKS - 1].recoilAngle * WEAPON_RECOIL_SCALE;
                priorData[PAST_AIM_TICKS - 1].deltaPosition =
                    trainingEngagementAimResult.deltaPosition[engagementAimId][PAST_AIM_TICKS];
                priorData[PAST_AIM_TICKS - 1].eyeToHeadDistance =
                    trainingEngagementAimResult.eyeToHeadDistance[engagementAimId][PAST_AIM_TICKS];
                priorData[PAST_AIM_TICKS - 1].warmupTicksUsed++;
            }
        }
        roundsProcessed++;
        printProgress(roundsProcessed, rounds.size);
    }
    size = trainingEngagementAimResult.size;
}
