//
// Created by durst on 9/28/22.
//

#include "queries/inference_moments/inference_engagement_aim.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

InferenceEngagementAimResult queryInferenceEngagementAimResult(const string & modelsDir,
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
        std::cerr << "error loading the model\n";
    }

    std::cout << "ok\n";

    return result;
}
