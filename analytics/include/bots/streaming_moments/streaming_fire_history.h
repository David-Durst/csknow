//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_FIRE_HISTORY_H
#define CSKNOW_STREAMING_FIRE_HISTORY_H

#include "queries/moments/fire_history.h"
#include "queries/training_moments/training_engagement_aim.h"
#include "bots/streaming_bot_database.h"

namespace csknow::fire_history {
    class StreamingFireHistory {
    public:
        struct FirePlayerData {
            CSGOId playerId;
            bool holdingAttackButton;
            int64_t ticksSinceLastFire;
            int64_t ticksSinceLastHoldingAttack;
            bool hitEnemy;
            set<int64_t> victims;
        };

        unordered_map<CSGOId, CircularBuffer<FirePlayerData>> firePlayerHistory;

        bool addPlayer(CSGOId csgoId) {
            if (firePlayerHistory.find(csgoId) == firePlayerHistory.end()) {
                firePlayerHistory.insert({csgoId, CircularBuffer<FirePlayerData>(PAST_AIM_TICKS)});
                return true;
            }
            return false;
        }

        void addTickData(const StreamingBotDatabase & db);
    };
}

#endif //CSKNOW_STREAMING_FIRE_HISTORY_H
