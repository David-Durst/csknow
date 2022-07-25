//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_SCRIPT_DATA_H
#define CSKNOW_SCRIPT_DATA_H

#include "bots/load_save_bot_data.h"

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
    Vec3 cameraOrigin;
    Vec2 cameraAngle;
};

struct NeededBot {
    CSGOId id;
    int team;
};

#endif //CSKNOW_SCRIPT_DATA_H
