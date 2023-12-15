//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_ACTION_DATA_H
#define CSKNOW_ACTION_DATA_H

#include "bots/load_save_bot_data.h"
#include "bots/input_bits.h"
#include "circular_buffer.h"
#define PID_HISTORY_LENGTH 10
#define MIN_JUMP_RESET_SECONDS 0.5
#define MIN_SCOPE_RESET_SECONDS 0.1

struct PIDState {
    CircularBuffer<double> errorHistory{PID_HISTORY_LENGTH};
};

enum class AimTargetType {
    Player,
    C4,
    HoldNonDangerArea,
    DangerArea,
    PathNonDangerArea,
    MovementDirection,
    Waypoint
};

struct Action {
    // keyboard/mouse inputs sent to game engine
    CSKnowTime lastJumpTime = defaultTime, lastScopeTime = defaultTime, lastActionTime = defaultTime;
    double rollingAvgMouseVelocity;
    bool enableSecondOrder;
    // default initialize this one since it isn't read from file
    int32_t lastTeleportConfirmationId = 0;
    int32_t buttons;
    bool intendedToFire;
    int32_t shotsInBurst;
    bool keepCrouching = false;
    AimTargetType aimTargetType;
    CSGOId targetPlayerId;
    Vec3 aimTarget;
    Vec2 targetViewAngle;
    string aimTargetTypeToString(AimTargetType type) {
        switch (type) {
            case AimTargetType::Player:
                return "Player";
            case AimTargetType::C4:
                return "C4";
            case AimTargetType::HoldNonDangerArea:
                return "HoldNonDangerArea";
            case AimTargetType::DangerArea:
                return "DangerArea";
            case AimTargetType::PathNonDangerArea:
                return "PathNonDangerArea";
            case AimTargetType::Waypoint:
                return "Waypoint";
            default:
                return "invalid";
        }
    }

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

    float inputAngleX;
    float inputAngleY;
    bool inputAngleAbsolute;
    // set these here as part of blackboard, so easily changeable in testing infra
    // unlike client server state which is const in tree node args
    // force human players inputs
    bool forceInput;
    // absolute position setting
    bool enableAbsPos;
    Vec3 absPos;
    Vec2 absView;

    string print() {
        return "buttons: " + std::to_string(buttons) + ", shots in burst: " + std::to_string(shotsInBurst)
               + ", mouse x: " + std::to_string(inputAngleX) +
               ", mouse y: " + std::to_string(inputAngleY) +
               ", enable second order: " + boolToString(enableSecondOrder) +
            ", aim target type: " + aimTargetTypeToString(aimTargetType);
    }
};

#endif //CSKNOW_ACTION_DATA_H
