//
// Created by durst on 4/13/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AGGRESSION_HELPERS_H
#define CSKNOW_INFERENCE_LATENT_AGGRESSION_HELPERS_H

#include "queries/inference_moments/inference_latent_engagement_helpers.h"

namespace csknow::inference_latent_aggression {
    struct InferenceAggressionTickValues {
        std::vector<float> rowCPP;
        vector<csknow::feature_store::EngagementEnemyState> enemyStates;
    };
    InferenceAggressionTickValues extractFeatureStoreAggressionValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex);
    struct InferenceAggressionTickProbabilities {
        vector<float> aggressionProbabilities;
        feature_store::NearestEnemyState mostLikelyAggression;
    };
    InferenceAggressionTickProbabilities extractFeatureStoreAggressionResults(
        const at::Tensor & output, const InferenceAggressionTickValues & values);
}

#endif //CSKNOW_INFERENCE_LATENT_AGGRESSION_HELPERS_H
