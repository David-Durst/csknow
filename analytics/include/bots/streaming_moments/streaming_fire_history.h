//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_FIRE_HISTORY_H
#define CSKNOW_STREAMING_FIRE_HISTORY_H

#include "queries/moments/fire_history.h"
#include "queries/training_moments/training_engagement_aim.h"
#include "bots/streaming_bot_database.h"
#include "streaming_client_history.h"

namespace csknow::fire_history {
    struct FireClientData {
        CSGOId playerId;
        bool holdingAttackButton;
        int64_t ticksSinceLastFire;
        int64_t ticksSinceLastHoldingAttack;
        bool hitEnemy;
        set<int64_t> victims;
    };

    class StreamingFireHistory {
    public:
        StreamingClientHistory<FireClientData> fireClientHistory;

        void addTickData(const StreamingBotDatabase & db);
    };
}

#endif //CSKNOW_STREAMING_FIRE_HISTORY_H
