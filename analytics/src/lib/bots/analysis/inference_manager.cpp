//
// Created by durst on 4/13/23.
//

#include "bots/analysis/inference_manager.h"
#include "bots/analysis/learned_models.h"

using namespace torch::indexing;

namespace csknow::inference_manager {

    InferenceManager::InferenceManager(const std::string & modelsDir) : valid(true),
        deltaPosModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                      fs::path("delta_pos_script_model.pt")),
        uncertainDeltaPosModelPath(fs::path(modelsDir) / fs::path("uncertain_model") /
                      fs::path("delta_pos_script_model.pt")) {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        torch::jit::getProfilingMode() = false;
        auto tmpDeltaPosModule = torch::jit::load(deltaPosModelPath);
        deltaPosModule = torch::jit::optimize_for_inference(tmpDeltaPosModule);
        auto tmpUncertainDeltaPosModule = torch::jit::load(uncertainDeltaPosModelPath);
        uncertainDeltaPosModule = torch::jit::optimize_for_inference(tmpUncertainDeltaPosModule);
    }

    void InferenceManager::setCurClients(const vector<ServerState::Client> & clients) {
        set<CSGOId> curClients;
        for (const auto & client : clients) {
            if (client.isAlive && client.isBot) {
                curClients.insert(client.csgoId);
                if (playerToInferenceData.find(client.csgoId) == playerToInferenceData.end()) {
                    playerToInferenceData[client.csgoId] = {};
                    playerToInferenceData[client.csgoId].team = client.team;
                    playerToInferenceData[client.csgoId].validUncertainDeltaPosProbabilities = false;
                    playerToInferenceData[client.csgoId].validDeltaPosProbabilities = false;
                }

            }
        }
        vector<CSGOId> oldClients;
        for (const auto & [clientId, _] : playerToInferenceData) {
            if (curClients.find(clientId) == curClients.end()) {
                oldClients.push_back(clientId);
            }
        }
        for (const auto & oldClient : oldClients) {
            playerToInferenceData.erase(oldClient);
        }
    }

    void InferenceManager::recordInputFeatureValues(csknow::feature_store::FeatureStoreResult & featureStoreResult) {
        /*
        orderValues = csknow::inference_latent_order::extractFeatureStoreOrderValues(featureStoreResult, 0);
        placeValues = csknow::inference_latent_place::extractFeatureStorePlaceValues(featureStoreResult, 0);
        areaValues = csknow::inference_latent_area::extractFeatureStoreAreaValues(featureStoreResult, 0);
         */
        deltaPosValues = csknow::inference_delta_pos::extractFeatureStoreDeltaPosValues(featureStoreResult, 0,
                                                                                        teamSaveControlParameters);
    }

    void InferenceManager::runDeltaPosInference(bool uncertainModule) {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor rowPT = torch::from_blob(deltaPosValues.rowCPP.data(),
                                               {1, static_cast<long>(deltaPosValues.rowCPP.size())},
                                               options);
        inputs.push_back(rowPT);
        float overallPush =
                teamSaveControlParameters.getPushModelValue(true, feature_store::DecreaseTimingOption::s5, 0);
        vector<float> similarityArr{overallPush};
        //vector<float> similarityArr{0.f, 0.f};
        torch::Tensor similarityPt = torch::from_blob(similarityArr.data(), {1, 1}, options);
        inputs.push_back(similarityPt);
        vector<float> temperatureArr{teamSaveControlParameters.temperature};
        torch::Tensor temperaturePt = torch::from_blob(temperatureArr.data(), {1, 1}, options);
        inputs.push_back(temperaturePt);

        if (uncertainModule) {
            //throw std::runtime_error("can't use uncertain model right now");
            at::Tensor output;
            if (getUseUncertainModel()) {
                output = uncertainDeltaPosModule.forward(inputs).toTuple()->elements()[1].toTensor();
            }

            for (auto & [csgoId, inferenceData] : playerToInferenceData) {
                playerToInferenceData[csgoId].validUncertainDeltaPosProbabilities = true;
                if (getUseUncertainModel()) {
                    playerToInferenceData[csgoId].uncertainDeltaPosProbabilities =
                            extractFeatureStoreDeltaPosResults(output, deltaPosValues, csgoId, inferenceData.team);
                }
            }
        }
        else {
            at::Tensor output = deltaPosModule.forward(inputs).toTuple()->elements()[1].toTensor();

            for (auto & [csgoId, inferenceData] : playerToInferenceData) {
                playerToInferenceData[csgoId].validDeltaPosProbabilities = true;
                playerToInferenceData[csgoId].deltaPosProbabilities =
                        extractFeatureStoreDeltaPosResults(output, deltaPosValues, csgoId, inferenceData.team);
            }

        }
    }

    //std::chrono::time_point<std::chrono::system_clock> lastInferenceTime;
    vector<float> inferenceTimePerIteration;
    size_t inferenceTimeIndex = 0;
    void InferenceManager::runInferences() {
        if (!valid) {
            //inferenceSeconds = 0;
            return;
        }

        torch::NoGradGuard no_grad;

        auto start = std::chrono::system_clock::now();
        //runEngagementInference(clients);
        //runAggressionInference(clients);
        ranDeltaPosInferenceThisTick = false;
        if (overallModelToRun == 0) {
            runDeltaPosInference(true);
            ranUncertainDeltaPosInference = true;
        }
        if (overallModelToRun == 8) {
            //std::chrono::duration<double> inferenceTime = start - lastInferenceTime;
            //std::cout << "times between inferences " << inferenceTime.count() << std::endl;
            //lastInferenceTime = start;
            runDeltaPosInference(false);
            ranDeltaPosInference = true;
            ranDeltaPosInferenceThisTick = true;
        }
        overallModelToRun = (overallModelToRun + 1) % ticks_per_inference;
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> inferenceTime = end - start;
        double tmpInferenceSeconds = inferenceTime.count();
        if (tmpInferenceSeconds > 1e-4) {
            inferenceSeconds = tmpInferenceSeconds;
        }
        if ((getUseUncertainModel() && overallModelToRun - 1 == 0) || overallModelToRun - 1 == 8) {
            if (inferenceTimePerIteration.size() < 10000) {
                inferenceTimePerIteration.push_back(inferenceTime.count());
            }
            else {
                inferenceTimePerIteration[inferenceTimeIndex] = inferenceTime.count();
                inferenceTimeIndex = (inferenceTimeIndex + 1) % inferenceTimePerIteration.size();
            }
        }
    }

    bool InferenceManager::haveValidData() const {
        return ranDeltaPosInference && ranUncertainDeltaPosInference;
        /*
        if (!ranOrderInference || !ranPlaceInference || !ranAreaInference || !ranDeltaPosInference) {
            return false;
        }
        for (const auto & [_, inferenceData] : playerToInferenceData) {
            if (!inferenceData.validData || inferenceData.orderProbabilities.orderProbabilities.empty() ||
                inferenceData.placeProbabilities.placeProbabilities.empty() ||
                inferenceData.areaProbabilities.areaProbabilities.empty() ||
                inferenceData.deltaPosProbabilities.deltaPosProbabilities.empty()) {
                return false;
            }
        }
        return true;
         */
    }

    void saveInferenceTimeLog(string path) {
        if (inferenceTimePerIteration.size() < 200) {
            return;
        }
        std::fstream inferenceTimeLogFile (path, std::fstream::out);
        inferenceTimeLogFile << "inference time" << std::endl;
        for (const auto & inferenceTime: inferenceTimePerIteration) {
            inferenceTimeLogFile << inferenceTime << std::endl;
        }
    }


}
