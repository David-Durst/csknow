//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_PRIORITY_DATA_H
#define CSKNOW_PRIORITY_DATA_H

#include <queries/query.h>
#include "load_save_bot_data.h"

struct TargetPlayer {
    CSGOId playerId = INVALID_ID;
    int64_t round;
    int32_t firstTargetFrame;
};

struct Priority {
    uint32_t targetAreaId;
    Vec3 targetPos;
    TargetPlayer targetPlayer;
    bool stuck = false;
    int stuckTicks = 0;

    string print(const ServerState & state) const {
        stringstream result;

        result << "target pos: (" << targetPos.toString() << "), target player id:"
            << state.getPlayerString(targetPlayer.playerId) << ", target player round: " << targetPlayer.round
            << ", target player first frame: " << targetPlayer.firstTargetFrame;

        result << ", stuck: " << boolToString(stuck) << ", stuck ticks: " << stuckTicks;

        return result.str();
    }
};

#endif //CSKNOW_PRIORITY_DATA_H
