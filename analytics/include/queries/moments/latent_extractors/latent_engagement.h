//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_LATENT_ENGAGEMENT_H
#define CSKNOW_LATENT_ENGAGEMENT_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
#include "queries/moments/engagement.h"
#include "queries/moments/behavior_tree_latent_states.h"


namespace csknow::latent_engagement {
    class LatentEngagementResult : public EngagementResult {
    public:
        LatentEngagementResult() = default;

        void runQuery(const Rounds & rounds, const Ticks & ticks, const Hurt & hurt,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates);
    };
}

#endif //CSKNOW_LATENT_ENGAGEMENT_H
