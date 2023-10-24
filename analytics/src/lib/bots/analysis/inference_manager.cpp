//
// Created by durst on 4/13/23.
//

#include "bots/analysis/inference_manager.h"
using namespace torch::indexing;

namespace csknow::inference_manager {

    InferenceManager::InferenceManager(const std::string & modelsDir) : valid(true),
        deltaPosModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                      fs::path("delta_pos_script_model.pt")),
        combatDeltaPosModelPath(fs::path(modelsDir) / fs::path("combat_model") /
                      fs::path("delta_pos_script_model.pt")) {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        torch::jit::getProfilingMode() = false;
        auto tmpDeltaPosModule = torch::jit::load(deltaPosModelPath);
        deltaPosModule = torch::jit::optimize_for_inference(tmpDeltaPosModule);
        auto tmpCombatDeltaPosModule = torch::jit::load(combatDeltaPosModelPath);
        combatDeltaPosModule = torch::jit::optimize_for_inference(tmpCombatDeltaPosModule);
    }

    void InferenceManager::setCurClients(const vector<ServerState::Client> & clients) {
        set<CSGOId> curClients;
        for (const auto & client : clients) {
            if (client.isAlive && client.isBot) {
                curClients.insert(client.csgoId);
                if (playerToInferenceData.find(client.csgoId) == playerToInferenceData.end()) {
                    playerToInferenceData[client.csgoId] = {};
                    playerToInferenceData[client.csgoId].team = client.team;
                    playerToInferenceData[client.csgoId].validData = false;
                    playerToInferenceData[client.csgoId].ticksSinceLastInference = max_track_ticks;
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

    void InferenceManager::runDeltaPosInference(bool combatModule) {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor rowPT = torch::from_blob(deltaPosValues.rowCPP.data(),
                                               {1, static_cast<long>(deltaPosValues.rowCPP.size())},
                                               options);
        inputs.push_back(rowPT);
        //float overallPush =
        //        teamSaveControlParameters.getPushModelValue(true, feature_store::DecreaseTimingOption::s5, 0);
        //vector<float> similarityArr{overallPush, overallPush};
        vector<float> similarityArr{0.f, 0.f};
        torch::Tensor similarityPt = torch::from_blob(similarityArr.data(), {1, 2}, options);
        inputs.push_back(similarityPt);
        vector<float> temperatureArr{teamSaveControlParameters.temperature};
        torch::Tensor temperaturePt = torch::from_blob(temperatureArr.data(), {1, 1}, options);
        inputs.push_back(temperaturePt);

        if (combatModule) {
            at::Tensor output = combatDeltaPosModule.forward(inputs).toTuple()->elements()[1].toTensor();

            for (auto & [csgoId, inferenceData] : playerToInferenceData) {
                playerToInferenceData[csgoId].combatDeltaPosProbabilities =
                        extractFeatureStoreDeltaPosResults(output, deltaPosValues, csgoId, inferenceData.team);
            }
        }
        else {
            at::Tensor output = deltaPosModule.forward(inputs).toTuple()->elements()[1].toTensor();

            for (auto & [csgoId, inferenceData] : playerToInferenceData) {
                playerToInferenceData[csgoId].deltaPosProbabilities =
                        extractFeatureStoreDeltaPosResults(output, deltaPosValues, csgoId, inferenceData.team);
            }
        }
    }

    void InferenceManager::runInferences() {
        if (!valid) {
            //inferenceSeconds = 0;
            return;
        }

        torch::NoGradGuard no_grad;
        // sort clients by ticks since max inference
        struct ClientAndTicks {
            CSGOId csgoId;
            size_t ticksSinceLastInference;
        };
        vector<ClientAndTicks> clientsToInfer;

        for (auto & [csgoId, clientInferenceData] : playerToInferenceData) {
            clientInferenceData.ticksSinceLastInference =
                std::min(clientInferenceData.ticksSinceLastInference + 1, max_track_ticks);
            clientsToInfer.push_back({csgoId, clientInferenceData.ticksSinceLastInference});
        }

        std::sort(clientsToInfer.begin(), clientsToInfer.end(),
                  [](const ClientAndTicks & a, const ClientAndTicks & b) {
            return a.ticksSinceLastInference > b.ticksSinceLastInference ||
                (a.ticksSinceLastInference == b.ticksSinceLastInference && a.csgoId < b.csgoId);
        });

        clientsToInfer.resize(std::min(batch_size_per_model, clientsToInfer.size()));

        vector<CSGOId> clients;
        for (const auto & client : clientsToInfer) {
            clients.push_back(client.csgoId);
            playerToInferenceData[client.csgoId].validData = true;
            playerToInferenceData[client.csgoId].ticksSinceLastInference = 0;
        }

        auto start = std::chrono::system_clock::now();
        //runEngagementInference(clients);
        //runAggressionInference(clients);
        if (overallModelToRun == 0) {
            runDeltaPosInference(false);
            ranDeltaPosInference = true;
        }
        if (overallModelToRun == 4) {
            runDeltaPosInference(true);
            ranCombatDeltaPosInference = true;
        }
        overallModelToRun = (overallModelToRun + 1) % 16;
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> inferenceTime = end - start;
        double tmpInferenceSeconds = inferenceTime.count();
        if (tmpInferenceSeconds > 1e-5) {
            inferenceSeconds = tmpInferenceSeconds;
        }
    }

    bool InferenceManager::haveValidData() const {
        return ranDeltaPosInference && ranCombatDeltaPosInference;
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

}
