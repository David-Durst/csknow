//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_ACTION_DATA_H
#define CSKNOW_ACTION_DATA_H

#include "load_save_bot_data.h"

struct Action {
    // keyboard/mouse inputs sent to game engine
    int32_t buttons;
    int32_t shotsInBurst;

    void setButton(int32_t button, bool setTrue) {
        if (setTrue) {
            buttons |= button;
        }
        else {
            buttons &= ~button;
        }
    }

    bool getButton(int32_t button) {
        return (buttons & button) > 0;
    }
    // these range from -1 to 1
    float inputAngleDeltaPctX;
    float inputAngleDeltaPctY;

    string print() {
        return "buttons: " + std::to_string(buttons) + ", shots in burst: " + std::to_string(shotsInBurst)
            + ", mouse delta x: " + std::to_string(inputAngleDeltaPctX) + ", mouse delta y: " + std::to_string(inputAngleDeltaPctY);
    }
};

#endif //CSKNOW_ACTION_DATA_H
