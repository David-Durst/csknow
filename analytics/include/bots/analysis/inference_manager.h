//
// Created by durst on 4/12/23.
//

#ifndef CSKNOW_INFERENCE_MANAGER_H
#define CSKNOW_INFERENCE_MANAGER_H

#include "queries/inference_moments/inference_latent_engagement_helpers.h"
#include "queries/inference_moments/inference_latent_aggression_helpers.h"
#include "queries/inference_moments/inference_latent_order_helpers.h"
#include "queries/inference_moments/inference_latent_place_helpers.h"
#include "queries/inference_moments/inference_latent_area_helpers.h"
#include "bots/load_save_bot_data.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace csknow::inference_manager {

    constexpr size_t batch_size_per_model = 1;
    constexpr size_t max_track_ticks = 20;

    struct ClientInferenceData {
        bool validData;
        size_t ticksSinceLastInference;
        csknow::inference_latent_engagement::InferenceEngagementTickValues engagementValues;
        csknow::inference_latent_engagement::InferenceEngagementTickProbabilities engagementProbabilities;
        csknow::inference_latent_aggression::InferenceAggressionTickValues aggressionValues;
        csknow::inference_latent_aggression::InferenceAggressionTickProbabilities aggressionProbabilities;
        csknow::inference_latent_order::InferenceOrderPlayerAtTickProbabilities orderProbabilities;
        csknow::inference_latent_place::InferencePlacePlayerAtTickProbabilities placeProbabilities;
        csknow::inference_latent_area::InferenceAreaPlayerAtTickProbabilities areaProbabilities;
    };

    class InferenceManager {
        void runEngagementInference(const vector<CSGOId> & clientsToInfer);
        void runAggressionInference(const vector<CSGOId> & clientsToInfer);
        void runOrderInference();
    public:
        bool valid;
        double inferenceSeconds;
        torch::TensorOptions options = torch::TensorOptions().dtype(at::kFloat);
        map<CSGOId, ClientInferenceData> playerToInferenceData;
        csknow::inference_latent_order::InferenceOrderTickValues orderValues;
        csknow::inference_latent_place::InferencePlaceTickValues placeValues;
        csknow::inference_latent_area::InferenceAreaTickValues areaValues;

        fs::path engagementModelPath, aggressionModelPath, orderModelPath;
        torch::jit::script::Module engagementModule, aggressionModule, orderModule;
        InferenceManager(const std::string & modelsDir);
        InferenceManager() : valid(false) { };

        void setCurClients(const vector<ServerState::Client> & clients);
        void recordTeamValues(csknow::feature_store::FeatureStoreResult & featureStoreResult);
        void recordPlayerValues(csknow::feature_store::FeatureStoreResult & featureStoreResult, CSGOId playerId);
        void runInferences();
    };
}

#endif //CSKNOW_INFERENCE_MANAGER_H
