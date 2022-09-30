//
// Created by durst on 9/28/22.
//

#include "queries/inference_moments/inference_engagement_aim.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

InferenceEngagementAimResult queryInferenceEngagementAimResult(const string & modelsDir,
                                                               const EngagementResult & engagementResult,
                                                               const TrainingEngagementAimResult & trainingEngagementAimResult) {
    InferenceEngagementAimResult result(trainingEngagementAimResult);
    result.size = trainingEngagementAimResult.size;

    fs::path modelPath = fs::path(modelsDir) / fs::path("engagement_aim_model") /
        fs::path("script_model.pt");

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading engagement aim model\n";
        return result;
    }


    auto options = torch::TensorOptions().dtype(at::kFloat);
    /*
    for (const auto & [engagementId, tickIdRange] :
        engagementResult.engagementsPerTick.eventToInterval) {
        for (int64_t tickId = tickIdRange.minId; tickId <= tickIdRange.maxId; tickId++) {

        }
    }
     */
    for (int64_t i = 0; i < result.trainingEngagementAimResult.size; i++) {
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        std::vector<float> rowCPP;
        // all but cur tick are inputs
        for (size_t j = 1; j < NUM_TICKS; j++) {
            rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.deltaViewAngle[i][j].x));
            rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.deltaViewAngle[i][j].y));
            rowCPP.push_back(static_cast<float>(trainingEngagementAimResult.eyeToHeadDistance[i][j]));
        }
        torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())}, options).clone();
        inputs.push_back(rowPT);

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        result.predictedDeltaViewAngle.push_back({
            static_cast<double>(output[0][2].item<float>()),
            static_cast<double>(output[0][3].item<float>())
        });
    }

    return result;
}
