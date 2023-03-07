//
// Created by durst on 3/5/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
#define CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
#include "queries/moments/behavior_tree_latent_states.h"

namespace csknow::inference_latent_engagement {
    class InferenceLatentEngagementResult : public EngagementResult {
    public:
        const PlayerAtTick & playerAtTick;

        explicit InferenceLatentEngagementResult(const PlayerAtTick & playerAtTick) : playerAtTick(playerAtTick) { };

        void runQuery(const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates,
                      bool useHitEngagementDefinition);

    };
}

#endif //CSKNOW_INFERENCE_LATENT_ENGAGEMENT_H
