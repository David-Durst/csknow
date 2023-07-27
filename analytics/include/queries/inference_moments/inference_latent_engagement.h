//
// Created by durst on 3/5/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
#define CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
#include "queries/moments/behavior_tree_latent_states.h"
#include "queries/inference_moments/inference_latent_engagement_helpers.h"
#include "feature_store_precommit.h"

namespace csknow::inference_latent_engagement {
    struct PlayerEngageProb {
        int64_t playerId;
        float prob;
    };


    class InferenceLatentEngagementResult : public EngagementResult {
    public:
        const PlayerAtTick & playerAtTick;
        vector<array<PlayerEngageProb, feature_store::max_enemies + 1>> playerEngageProbs;

        explicit InferenceLatentEngagementResult(const PlayerAtTick & playerAtTick) : playerAtTick(playerAtTick) { };

        void runQuery(const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates);

    };
}

#endif //CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
