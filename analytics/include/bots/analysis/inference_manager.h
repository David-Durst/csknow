//
// Created by durst on 4/12/23.
//

#ifndef CSKNOW_INFERENCE_MANAGER_H
#define CSKNOW_INFERENCE_MANAGER_H

#include "queries/inference_moments/inference_latent_engagement.h"
#include "bots/load_save_bot_data.h"

namespace csknow::inference_manager {

    constexpr size_t batch_size_per_model = 2;
    constexpr size_t max_track_ticks = 20;

    struct ClientInferenceData {
        size_t ticksSinceLastInference;
        csknow::inference_latent_engagement::InferenceEngagementTickValues engagementValues;
        csknow::inference_latent_engagement::InferenceEngagementTickProbabilities engagementProbabilities;
    };

    class InferenceManager {
        void runEngagementInference(const vector<CSGOId> & clientsToInfer);
    public:
        bool valid;
        double inferenceSeconds;
        torch::TensorOptions options = torch::TensorOptions().dtype(at::kFloat);
        map<CSGOId, ClientInferenceData> playerToInferenceData;

        fs::path engagementModelPath, aggressionModelPath, orderModelPath;
        torch::jit::script::Module engagementModule, aggressionModule, orderModule;
        InferenceManager(const std::string & modelsDir);
        InferenceManager() : valid(false) { };

        void setCurClients(const vector<ServerState::Client> & clients);
        void recordPlayerValues(csknow::feature_store::FeatureStoreResult & featureStoreResult, CSGOId playerId);
        void runInferences();
    };
}

#endif //CSKNOW_INFERENCE_MANAGER_H
