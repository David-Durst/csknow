//
// Created by durst on 9/28/22.
//

#include "queries/inference_moments/inference_engagement_aim.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

struct StandardScalarParams {
    string columnName;
    double stdDev, mean;

    double apply(double input) {
        return (input - mean) / stdDev;
    }

    double invert(double output) {
        return (output * stdDev) + mean;
    }
};

InferenceEngagementAimResult queryInferenceEngagementAimResult(const string & modelsDir,
                                                               const TrainingEngagementAimResult & trainingEngagementAimResult) {
    InferenceEngagementAimResult result(trainingEngagementAimResult);
    result.size = trainingEngagementAimResult.size;

    fs::path modelPath = fs::path(modelsDir) / fs::path("engagement_aim_model") /
        fs::path("script_model.pt");
    fs::path transformsPath = fs::path(modelsDir) / fs::path("engagement_aim_model") /
        fs::path("transforms.csv");

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading engagement aim model\n";
        return result;
    }

    vector<StandardScalarParams> inputTransformParams, outputTransformParams;
    std::ifstream transformsFileStream(transformsPath);

    if (std::filesystem::exists(transformsPath)) {
        string transformsFileBuf;
        bool inputLine = true;
        while (getline(transformsFileStream, transformsFileBuf)) {
            stringstream transformsLineStream(transformsFileBuf);
            string transformsLineBuf;
            while (getline(transformsLineStream, transformsLineBuf, ',')) {
                stringstream transformColStream(transformsFileBuf);
                string transformColBuf;

                getline(transformColStream, transformColBuf, ';');
                if (transformColBuf != "standard-scaler") {
                    std::cerr << "invalid scalar type" << std::endl;
                }

                string colName;
                getline(transformColStream, colName, ';');

                double stdDev, mean;
                getline(transformColStream, transformColBuf, ';');
                stdDev = std::stod(transformColBuf);
                getline(transformColStream, transformColBuf, ';');
                mean = std::stod(transformColBuf);

                if (inputLine) {
                    inputTransformParams.push_back({colName, stdDev, mean});
                }
                else {
                    outputTransformParams.push_back({colName, stdDev, mean});
                }
            }
            inputLine = false;
        }
    }
    else {
        throw std::runtime_error("no valid transforms");
    }

    auto options = torch::TensorOptions().dtype(at::kFloat);
    for (int64_t i = 0; i < result.trainingEngagementAimResult.size; i++) {

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        std::vector<float> rowCPP;
        // all but cur tick are inputs
        for (size_t j = 1; j < NUM_TICKS; j++) {
            rowCPP.push_back(static_cast<float>(
                inputTransformParams[3*(j-1)].apply(trainingEngagementAimResult.deltaViewAngle[i][j].x)));
            rowCPP.push_back(static_cast<float>(
                inputTransformParams[3*(j-1)+1].apply(trainingEngagementAimResult.deltaViewAngle[i][j].y)));
            rowCPP.push_back(static_cast<float>(
                inputTransformParams[3*(j-1)+2].apply(trainingEngagementAimResult.eyeToHeadDistance[i][j])));
        }
        torch::Tensor rowPT = torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())}, options).clone();
        inputs.push_back(rowPT);

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        result.predictedDeltaViewAngle.push_back({
            static_cast<double>(output[0][0].item<float>()),
            static_cast<double>(output[0][1].item<float>())
        });
        int x = 1;
        (void) x;
    }

    return result;
}
