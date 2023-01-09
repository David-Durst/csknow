//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_SCRIPT_DATA_H
#define CSKNOW_SCRIPT_DATA_H

#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/blackboard.h"

enum class ObserveType {
    FirstPerson,
    ThirdPerson,
    Absolute,
    None,
    NUM_OBSERVE_TYPE
};

struct ObserveSettings {
    ObserveType observeType;
    CSKnowId neededBotIndex;
    Vec3 cameraOrigin = {0., 0., 0.};
    Vec2 cameraAngle = {0., 0.};
};

struct NeededBot {
    CSGOId id;
    int team;
    AggressiveType type = AggressiveType::Push;
    bool human = false;
};

#endif //CSKNOW_SCRIPT_DATA_H
