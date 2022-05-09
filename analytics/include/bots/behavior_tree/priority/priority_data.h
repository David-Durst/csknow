//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_PRIORITY_DATA_H
#define CSKNOW_PRIORITY_DATA_H
#include "load_save_bot_data.h"

enum class PriorityType {
    NavArea,
    Player,
    C4,
    NUM_PRIORITY_TYPES
};

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
};

#endif //CSKNOW_PRIORITY_DATA_H
