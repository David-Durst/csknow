//
// Created by durst on 9/28/22.
//

#include "queries/inference_moments/inference_engagement_aim.h"
#include "queries/rolling_window.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

struct EngagementAimInferenceTickData {
    Vec2 deltaViewAngle;
    Vec2 recoilAngle;
    Vec2 deltaViewAngleRecoilAdjusted;
    double eyeToHeadDistance;
    AimWeaponType weaponType;
};

void InferenceEngagementAimResult::runQuery(const string & modelsDir, const EngagementResult & engagementResult) {
    fs::path modelPath = fs::path(modelsDir) / fs::path("engagement_aim_model") /
        fs::path("script_model.pt");

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading engagement aim model\n";
        return;
    }


    auto options = torch::TensorOptions().dtype(at::kFloat);
    // NUM_TICKS stores cur tick and prior ticks in window, shrink by 1 for just prior ticks
    /*
    map<int64_t, array<EngagementAimInferenceTickData, PAST_AIM_TICKS>> activeEngagementsPriorTickData;
    for (int64_t engagementAimId = 0; engagementAimId < trainingEngagementAimResult.size; engagementAimId++) {
        int64_t engagementId = trainingEngagementAimResult.engagementId[engagementAimId];
        array<EngagementAimInferenceTickData, PAST_AIM_TICKS> & priorData = activeEngagementsPriorTickData[engagementId];

        // add old deltas if in engagement's first tick, otherwise use delta prior earlier predictions
        const RangeIndexEntry & engagementTickRange =
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
                priorData[priorTickNum].weaponType =
                    trainingEngagementAimResult.weaponType[engagementAimId];
            }
        }

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
            rowCPP.push_back(static_cast<float>(priorData[priorDeltaNum].eyeToHeadDistance));
        }
        torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())}, options).clone();
        inputs.push_back(rowPT);

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        predictedDeltaViewAngle.push_back({
            static_cast<double>(output[0][output[0].size(1) / 2].item<float>()),
            static_cast<double>(output[0][output[0].size(1) / 2 + 1].item<float>())
        });
        normalizedPredictedDeltaViewAngle.push_back({
            predictedDeltaViewAngle.back().x / trainingEngagementAimResult.distanceNormalization[engagementAimId],
            predictedDeltaViewAngle.back().y / trainingEngagementAimResult.distanceNormalization[engagementAimId]
        });

        // if last tick for engagement, remove it from actives. Otherwise rotate the current prediction into prior deltas
        if (trainingEngagementAimResult.tickId[engagementAimId] == engagementTickRange.maxId) {
            activeEngagementsPriorTickData.erase(engagementId);
        }
        else {
            for (size_t priorDeltaNum = 1; priorDeltaNum < activeEngagementsPriorTickData.size(); priorDeltaNum++) {
                activeEngagementsPriorTickData[priorDeltaNum - 1] = activeEngagementsPriorTickData[priorDeltaNum];
            }
            priorData[PAST_AIM_TICKS - 1].deltaViewAngle = predictedDeltaViewAngle.back();
            priorData[PAST_AIM_TICKS - 1].deltaViewAngle =
                trainingEngagementAimResult.deltaViewAngle[engagementAimId][PAST_AIM_TICKS];
            priorData[PAST_AIM_TICKS - 1].recoilAngle =
                trainingEngagementAimResult.recoilAngle[engagementAimId][PAST_AIM_TICKS];
            priorData[PAST_AIM_TICKS - 1].deltaViewAngleRecoilAdjusted =
                trainingEngagementAimResult.deltaViewAngleRecoilAdjusted[engagementAimId][PAST_AIM_TICKS];
            priorData[PAST_AIM_TICKS - 1].eyeToHeadDistance =
                trainingEngagementAimResult.eyeToHeadDistance[engagementAimId][PAST_AIM_TICKS];
            priorData[PAST_AIM_TICKS - 1].weaponType =
                trainingEngagementAimResult.weaponType[engagementAimId];
        }
    }
     */
}
