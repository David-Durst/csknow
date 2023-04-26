//
// Created by durst on 4/13/23.
//

#include "bots/analysis/inference_manager.h"
using namespace torch::indexing;

namespace csknow::inference_manager {

    InferenceManager::InferenceManager(const std::string & modelsDir) : valid(true),
        engagementModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                            fs::path("engagement_script_model.pt")),
        aggressionModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                            fs::path("aggression_script_model.pt")),
        orderModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                       fs::path("order_script_model.pt")),
        placeModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                       fs::path("place_script_model.pt")),
        areaModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                       fs::path("area_script_model.pt")) {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        torch::jit::getProfilingMode() = false;
        auto tmpEngagementModule = torch::jit::load(engagementModelPath);
        engagementModule = torch::jit::optimize_for_inference(tmpEngagementModule);
        auto tmpAggressionModule = torch::jit::load(aggressionModelPath);
        aggressionModule = torch::jit::optimize_for_inference(tmpAggressionModule);
        auto tmpOrderModule = torch::jit::load(orderModelPath);
        orderModule = torch::jit::optimize_for_inference(tmpOrderModule);
        auto tmpPlaceModule = torch::jit::load(placeModelPath);
        placeModule = torch::jit::optimize_for_inference(tmpPlaceModule);
        auto tmpAreaModule = torch::jit::load(areaModelPath);
        areaModule = torch::jit::optimize_for_inference(tmpAreaModule);
    }

    void InferenceManager::setCurClients(const vector<ServerState::Client> & clients) {
        set<CSGOId> curClients;
        for (const auto & client : clients) {
            if (client.isAlive && client.isBot) {
                curClients.insert(client.csgoId);
                if (playerToInferenceData.find(client.csgoId) == playerToInferenceData.end()) {
                    playerToInferenceData[client.csgoId] = {};
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

    void InferenceManager::recordTeamValues(csknow::feature_store::FeatureStoreResult & featureStoreResult) {
        orderValues = csknow::inference_latent_order::extractFeatureStoreOrderValues(featureStoreResult, 0);
        placeValues = csknow::inference_latent_place::extractFeatureStorePlaceValues(featureStoreResult, 0);
        areaValues = csknow::inference_latent_area::extractFeatureStoreAreaValues(featureStoreResult, 0);
    }

    void InferenceManager::recordPlayerValues(csknow::feature_store::FeatureStoreResult &featureStoreResult,
                                              CSGOId playerId) {
        playerToInferenceData[playerId].engagementValues =
            csknow::inference_latent_engagement::extractFeatureStoreEngagementValues(featureStoreResult, 0);
        playerToInferenceData[playerId].aggressionValues =
            csknow::inference_latent_aggression::extractFeatureStoreAggressionValues(featureStoreResult, 0);
    }

    void InferenceManager::runEngagementInference(const vector<CSGOId> & clientsToInfer) {
        if (clientsToInfer.empty()) {
            return;
        }
        vector<csknow::inference_latent_engagement::InferenceEngagementTickValues> values;
        size_t elementsPerClient = playerToInferenceData[clientsToInfer.front()].engagementValues.rowCPP.size();
        std::vector<float> rowCPP;
        for (const auto & csgoId : clientsToInfer) {
            const vector<float> & playerRow = playerToInferenceData[csgoId].engagementValues.rowCPP;
            rowCPP.insert(rowCPP.end(), playerRow.begin(), playerRow.end());
        }
        std::vector<torch::jit::IValue> inputs;

        torch::Tensor rowPT =
            torch::from_blob(rowCPP.data(),
                             {static_cast<long>(clientsToInfer.size()), static_cast<long>(elementsPerClient)}, options);
        inputs.push_back(rowPT);

        at::Tensor output = engagementModule.forward(inputs).toTuple()->elements()[0].toTensor();

        for (size_t i = 0; i < clientsToInfer.size(); i++) {
            at::Tensor playerOutput = output.index({Slice(i, None, i+1), "..."});
            playerToInferenceData[clientsToInfer[i]].engagementProbabilities =
                csknow::inference_latent_engagement::extractFeatureStoreEngagementResults(
                    playerOutput, playerToInferenceData[clientsToInfer[i]].engagementValues);
        }
    }

    void InferenceManager::runAggressionInference(const vector<CSGOId> & clientsToInfer) {
        if (clientsToInfer.empty()) {
            return;
        }
        vector<csknow::inference_latent_aggression::InferenceAggressionTickValues> values;
        size_t elementsPerClient = playerToInferenceData[clientsToInfer.front()].aggressionValues.rowCPP.size();
        std::vector<float> rowCPP;
        for (const auto & csgoId : clientsToInfer) {
            const vector<float> & playerRow = playerToInferenceData[csgoId].aggressionValues.rowCPP;
            rowCPP.insert(rowCPP.end(), playerRow.begin(), playerRow.end());
        }
        std::vector<torch::jit::IValue> inputs;

        torch::Tensor rowPT =
            torch::from_blob(rowCPP.data(),
                             {static_cast<long>(clientsToInfer.size()), static_cast<long>(elementsPerClient)}, options);
        inputs.push_back(rowPT);

        at::Tensor output = aggressionModule.forward(inputs).toTuple()->elements()[0].toTensor();

        for (size_t i = 0; i < clientsToInfer.size(); i++) {
            at::Tensor playerOutput = output.index({Slice(i, None, i+1), "..."});
            playerToInferenceData[clientsToInfer[i]].aggressionProbabilities =
                csknow::inference_latent_aggression::extractFeatureStoreAggressionResults(playerOutput);
        }
    }

    void InferenceManager::runOrderInference() {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor rowPT = torch::from_blob(orderValues.rowCPP.data(),
                                               {1, static_cast<long>(orderValues.rowCPP.size())},
                                               options);

        inputs.push_back(rowPT);

        at::Tensor output = orderModule.forward(inputs).toTuple()->elements()[0].toTensor();

        for (auto & [csgoId, _] : playerToInferenceData) {
            playerToInferenceData[csgoId].orderProbabilities =
                extractFeatureStoreOrderResults(output, orderValues, csgoId);
        }
    }

    void InferenceManager::runPlaceInference() {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor rowPT = torch::from_blob(placeValues.rowCPP.data(),
                                               {1, static_cast<long>(placeValues.rowCPP.size())},
                                               options);

        inputs.push_back(rowPT);

        at::Tensor output = placeModule.forward(inputs).toTuple()->elements()[0].toTensor();

        for (auto & [csgoId, _] : playerToInferenceData) {
            playerToInferenceData[csgoId].placeProbabilities =
                    extractFeatureStorePlaceResults(output, placeValues, csgoId);
        }
    }

    void InferenceManager::runAreaInference() {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor rowPT = torch::from_blob(areaValues.rowCPP.data(),
                                               {1, static_cast<long>(areaValues.rowCPP.size())},
                                               options);

        inputs.push_back(rowPT);

        at::Tensor output = areaModule.forward(inputs).toTuple()->elements()[0].toTensor();

        for (auto & [csgoId, _] : playerToInferenceData) {
            playerToInferenceData[csgoId].areaProbabilities =
                    extractFeatureStoreAreaResults(output, areaValues, csgoId);
        }
    }

    void InferenceManager::runInferences() {
        if (!valid) {
            inferenceSeconds = 0;
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
        runEngagementInference(clients);
        runAggressionInference(clients);
        runOrderInference();
        runPlaceInference();
        runAreaInference();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> inferenceTime = end - start;
        inferenceSeconds = inferenceTime.count();
    }

    bool InferenceManager::haveValidData() const {
        for (const auto & [_, inferenceData] : playerToInferenceData) {
            if (!inferenceData.validData) {
                return false;
            }
        }
        return true;
    }

}
