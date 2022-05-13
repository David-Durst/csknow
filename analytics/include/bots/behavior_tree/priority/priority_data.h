//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_PRIORITY_DATA_H
#define CSKNOW_PRIORITY_DATA_H

#include <queries/query.h>
#include "load_save_bot_data.h"

struct PriorityMovementOptions {
    bool move;
    bool walk;
    bool crouch;
};

enum class PriorityShootOptions {
    DontShoot,
    Tap,
    Burst,
    Spray,
    NUM_PRIORITY_SHOOT_OPTIONS
};

struct TargetPlayer {
    CSGOId playerId = INVALID_ID;
    int64_t round;
    int32_t firstTargetFrame;
};

struct Priority {
    uint32_t targetAreaId;
    Vec3 targetPos;
    TargetPlayer targetPlayer;
    PriorityMovementOptions movementOptions;
    PriorityShootOptions shootOptions;

    string print(const ServerState & state) const {
        stringstream result;

        string shootOptionStr;
        switch (shootOptions) {
            case PriorityShootOptions::DontShoot:
                shootOptionStr = "DontShoot";
                break;
            case PriorityShootOptions::Tap:
                shootOptionStr = "Tap";
                break;
            case PriorityShootOptions::Burst:
                shootOptionStr = "Burst";
                break;
            case PriorityShootOptions::Spray:
                shootOptionStr = "Spray";
                break;
            default:
                shootOptionStr = "Invalid";
        }

        result << "target pos: (" << targetPos.toString() << "), target player id:"
            << state.getPlayerString(targetPlayer.playerId) << ", target player round: " << targetPlayer.round
            << ", target player first frame: " << targetPlayer.firstTargetFrame
            << ", move: " << boolToString(movementOptions.move) << ", walk: " << boolToString(movementOptions.walk)
            << ", crouch: " << boolToString(movementOptions.crouch)
            << ", shoot option: " << shootOptionStr;

        return result.str();
    }
};

#endif //CSKNOW_PRIORITY_DATA_H
