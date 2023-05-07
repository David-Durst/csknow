//
// Created by durst on 4/13/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ENGAGEMENT_HELPERS_H
#define CSKNOW_INFERENCE_LATENT_ENGAGEMENT_HELPERS_H

#include "bots/load_save_bot_data.h"
#include "bots/analysis/feature_store.h"
#include <torch/script.h>
#include <ATen/Parallel.h>

namespace csknow::inference_latent_engagement {
    struct InferenceEngagementTickValues {
        std::vector<float> rowCPP;
        vector<CSGOId> enemyIds;
        vector<csknow::feature_store::EngagementEnemyState> enemyStates;
    };
    InferenceEngagementTickValues extractFeatureStoreEngagementValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex);
    struct InferenceEngagementTickProbabilities {
        vector<float> enemyProbabilities;
        size_t mostLikelyEnemyNum;
    };
    InferenceEngagementTickProbabilities extractFeatureStoreEngagementResults(
        //const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex, int64_t tickIndex,
        const at::Tensor & output, const InferenceEngagementTickValues & values);
}

#endif //CSKNOW_INFERENCE_LATENT_ENGAGEMENT_HELPERS_H
