//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_ACTION_DATA_H
#define CSKNOW_ACTION_DATA_H

#include "load_save_bot_data.h"

struct Action {
    // keyboard/mouse inputs sent to game engine
    int32_t buttons;

    void setButton(ServerState::Client & curClient, int32_t button, bool setTrue) {
        if (setTrue) {
            curClient.buttons |= button;
        }
        else {
            curClient.buttons &= ~button;
        }
    }

    bool getButton(ServerState::Client & curClient, int32_t button) {
        return curClient.buttons & button > 0;
    }
    // these range from -1 to 1
    float inputAngleDeltaPctX;
    float inputAngleDeltaPctY;
};

#endif //CSKNOW_ACTION_DATA_H
