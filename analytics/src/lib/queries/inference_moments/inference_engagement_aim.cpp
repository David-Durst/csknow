//
// Created by durst on 9/28/22.
//

#include "queries/inference_moments/inference_engagement_aim.h"
#include "queries/rolling_window.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

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
    map<int64_t, array<Vec2, PAST_AIM_TICKS>> activeEngagementsPriorDeltas;
    for (int64_t engagementAimId = 0; engagementAimId < trainingEngagementAimResult.size; engagementAimId++) {
        int64_t engagementId = trainingEngagementAimResult.engagementId[engagementAimId];
        array<Vec2, PAST_AIM_TICKS> & priorDeltas = activeEngagementsPriorDeltas[engagementId];

        // add old deltas if in engagement's firs tick, otherwise use delta prior earlier predictions
        const RangeIndexEntry & engagementTickRange =
            engagementResult.engagementsPerTick.eventToInterval.at(engagementId);
        if (trainingEngagementAimResult.tickId[engagementAimId] == engagementTickRange.minId) {
            for (size_t priorTickNum = 0; priorTickNum < NUM_TICKS - 1; priorTickNum++) {
                priorDeltas[priorTickNum] = trainingEngagementAimResult.deltaViewAngle[engagementAimId][priorTickNum + 1];
            }
        }

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        std::vector<float> rowCPP;
        // all but cur tick are inputs
        for (size_t priorDeltaNum = 0; priorDeltaNum < priorDeltas.size(); priorDeltaNum++) {
            rowCPP.push_back(static_cast<float>(priorDeltas[priorDeltaNum].x));
            rowCPP.push_back(static_cast<float>(priorDeltas[priorDeltaNum].y));
            rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.eyeToHeadDistance[engagementAimId][priorDeltaNum+1]));
        }
        torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())}, options).clone();
        inputs.push_back(rowPT);

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        predictedDeltaViewAngle.push_back({
            static_cast<double>(output[0][2].item<float>()),
            static_cast<double>(output[0][3].item<float>())
        });
        normalizedPredictedDeltaViewAngle.push_back({
            predictedDeltaViewAngle.back().x / trainingEngagementAimResult.distanceNormalization[engagementAimId],
            predictedDeltaViewAngle.back().y / trainingEngagementAimResult.distanceNormalization[engagementAimId]
        });

        // if last tick for engagement, remove it from actives. Otherwise rotate the current prediction into prior deltas
        if (trainingEngagementAimResult.tickId[engagementAimId] == engagementTickRange.maxId) {
            activeEngagementsPriorDeltas.erase(engagementId);
        }
        else {
            for (size_t priorDeltaNum = priorDeltas.size() - 1; priorDeltaNum > 0; priorDeltaNum--) {
                priorDeltas[priorDeltaNum] = priorDeltas[priorDeltaNum - 1];
            }
            priorDeltas[0] = predictedDeltaViewAngle.back();
        }
    }
     */
}
