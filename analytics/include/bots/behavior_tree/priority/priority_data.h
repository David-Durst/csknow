//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_PRIORITY_DATA_H
#define CSKNOW_PRIORITY_DATA_H

#include <queries/query.h>
#include "load_save_bot_data.h"

struct MoveOptions {
    bool move;
    bool walk;
    bool crouch;
};

enum class ShootOptions {
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
    Vec3 footPos;
    Vec3 eyePos;
    bool visible;
};

enum class PriorityType {
    Order,
    Engagement,
    NUM_PRIORITY_TYPES
};

struct Priority {
    PriorityType priorityType;
    uint32_t targetAreaId;
    Vec3 targetPos;
    TargetPlayer targetPlayer;
    MoveOptions moveOptions;
    ShootOptions shootOptions;
    bool stuck = false;
    int stuckTicks = 0;

    string print(const ServerState & state) const {
        stringstream result;

        result << "priority type: ";
        if (priorityType == PriorityType::Order) {
            result << "Order, ";
        }
        else {
            result << "Engagement, ";
        }

        result << "target pos: (" << targetPos.toString() << "), target player id:"
            << state.getPlayerString(targetPlayer.playerId) << ", target player round: " << targetPlayer.round
            << ", target player first frame: " << targetPlayer.firstTargetFrame;

        string shootOptionStr;
        switch (shootOptions) {
            case ShootOptions::DontShoot:
                shootOptionStr = "DontShoot";
                break;
            case ShootOptions::Tap:
                shootOptionStr = "Tap";
                break;
            case ShootOptions::Burst:
                shootOptionStr = "Burst";
                break;
            case ShootOptions::Spray:
                shootOptionStr = "Spray";
                break;
            default:
                shootOptionStr = "Invalid";
        }

        result << ", move: " << boolToString(moveOptions.move) << ", walk: " << boolToString(moveOptions.walk)
               << ", crouch: " << boolToString(moveOptions.crouch) << ", shoot option: " << shootOptionStr;

        result << ", stuck: " << boolToString(stuck) << ", stuck ticks: " << stuckTicks;

        return result.str();
    }
};

#endif //CSKNOW_PRIORITY_DATA_H
