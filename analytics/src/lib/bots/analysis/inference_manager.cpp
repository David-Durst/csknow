//
// Created by durst on 4/13/23.
//

#include "bots/analysis/inference_manager.h"

namespace csknow::inference_manager {

    InferenceManager::InferenceManager(const std::string & modelsDir) :
        engagementModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                            fs::path("engagement_script_model.pt")),
        aggressionModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                            fs::path("aggression_script_model.pt")),
        orderModelPath(fs::path(modelsDir) / fs::path("latent_model") /
                       fs::path("order_script_model.pt")) {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        torch::jit::getProfilingMode() = false;
        auto tmpEngagementModule = torch::jit::load(engagementModelPath);
        engagementModule = torch::jit::optimize_for_inference(tmpEngagementModule);
        auto tmpAggressionModule = torch::jit::load(aggressionModelPath);
        aggressionModule = torch::jit::optimize_for_inference(tmpAggressionModule);
        auto tmpOrderModule = torch::jit::load(orderModelPath);
        orderModule = torch::jit::optimize_for_inference(tmpOrderModule);
    }

    void InferenceManager::setCurClients(const vector<ServerState::Client> & clients) {
        set<CSGOId> curClients;
        for (const auto & client : clients) {
            curClients.insert(client.csgoId);
            if (playerToInferenceData.find(client.csgoId) == playerToInferenceData.end()) {
                playerToInferenceData[client.csgoId] = {};
                playerToInferenceData[client.csgoId].ticksSinceLastInference = max_track_ticks;
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

    void InferenceManager::recordPlayerValues(csknow::feature_store::FeatureStoreResult &featureStoreResult,
                                              CSGOId playerId) {
        playerToInferenceData[playerId].engagementValues =
            csknow::inference_latent_engagement::extractFeatureStoreEngagementValues(featureStoreResult, 0);
    }

}
