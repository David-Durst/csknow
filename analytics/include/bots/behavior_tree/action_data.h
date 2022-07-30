//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_ACTION_DATA_H
#define CSKNOW_ACTION_DATA_H

#include "bots/load_save_bot_data.h"
#include "bots/input_bits.h"
#include "circular_buffer.h"
#define PID_HISTORY_LENGTH 10

struct PIDState {
    CircularBuffer<double> errorHistory{PID_HISTORY_LENGTH};
};

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

    bool getButton(int32_t button) const {
        return (buttons & button) > 0;
    }
    bool movingForward() const {
        return getButton(IN_FORWARD) && !getButton(IN_BACK);
    }
    bool movingBackward() const {
        return !getButton(IN_FORWARD) && getButton(IN_BACK);
    }
    bool movingLeft() const {
        return getButton(IN_MOVELEFT) && !getButton(IN_MOVERIGHT);
    }
    bool movingRight() const {
        return !getButton(IN_MOVELEFT) && getButton(IN_MOVERIGHT);
    }
    bool moving() const {
        return movingForward() || movingBackward() || movingLeft() || movingRight();
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
