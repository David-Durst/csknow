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

enum class PriorityMovement {
    Wait,
    Move,
    Shoot,
    ShootAndMove
};

struct Priority {
    PriorityType priorityType;
    int64_t areaId;
    CSGOId playerId;
    PriorityMovement priorityMovement;
};

#endif //CSKNOW_PRIORITY_DATA_H
