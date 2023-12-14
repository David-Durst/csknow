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
#include "queries/inference_moments/inference_latent_delta_pos_helpers.h"
#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/inference_control_parameters.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace csknow::inference_manager {

    constexpr size_t batch_size_per_model = 1;
    constexpr size_t max_track_ticks = 20;
    constexpr int ticks_per_inference = 16;
    constexpr int ticks_per_seconds = 128;

    struct ClientInferenceData {
        bool validData;
        TeamId team;
        size_t ticksSinceLastInference;
        csknow::inference_delta_pos::InferenceDeltaPosPlayerAtTickProbabilities deltaPosProbabilities,
            combatDeltaPosProbabilities;
    };

    class InferenceManager {
        bool ranDeltaPosInference = false;
        void runDeltaPosInference();
        int overallModelToRun = 0;
    public:
        bool ranDeltaPosInferenceThisTick = false;
        bool valid;
        double inferenceSeconds;
        torch::TensorOptions options = torch::TensorOptions().dtype(at::kFloat);
        map<CSGOId, ClientInferenceData> playerToInferenceData;
        csknow::inference_delta_pos::InferenceDeltaPosTickValues deltaPosValues, combatDeltaPosValues;
        TeamSaveControlParameters teamSaveControlParameters;

        fs::path deltaPosModelPath, combatDeltaPosModelPath;
        torch::jit::script::Module deltaPosModule, combatDeltaPosModule;
        InferenceManager(const std::string & modelsDir);
        InferenceManager() : valid(false) { };

        void setCurClients(const vector<ServerState::Client> & clients);
        void recordInputFeatureValues(csknow::feature_store::FeatureStoreResult & featureStoreResult);
        void runInferences();
        bool haveValidData() const;
    };
}

#endif //CSKNOW_INFERENCE_MANAGER_H
