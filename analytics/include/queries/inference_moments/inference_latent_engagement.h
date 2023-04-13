//
// Created by durst on 3/5/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
#define CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
#include "queries/moments/behavior_tree_latent_states.h"

namespace csknow::inference_latent_engagement {
    struct PlayerEngageProb {
        int64_t playerId;
        float prob;
    };

    struct InferenceEngagementTickValues {
        std::vector<float> rowCPP;
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

    class InferenceLatentEngagementResult : public EngagementResult {
    public:
        const PlayerAtTick & playerAtTick;
        vector<array<PlayerEngageProb, feature_store::maxEnemies+1>> playerEngageProbs;

        explicit InferenceLatentEngagementResult(const PlayerAtTick & playerAtTick) : playerAtTick(playerAtTick) { };

        void runQuery(const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates);

    };
}

#endif //CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
