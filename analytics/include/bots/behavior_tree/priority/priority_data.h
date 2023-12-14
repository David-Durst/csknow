//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_PRIORITY_DATA_H
#define CSKNOW_PRIORITY_DATA_H

#include <queries/query.h>
#include "geometryNavConversions.h"
#include "bots/load_save_bot_data.h"
#include "bots/analysis/weapon_speed.h"

struct PriorityPlaceAssignment {
    PlaceIndex nextPlace;
    bool valid;
};

struct PriorityAreaAssignment {
    Vec3 targetPos;
    AreaId targetAreaId;
    bool valid;
};

struct PriorityDeltaPosAssignment {
    std::optional<csknow::weapon_speed::MovementStatus> learnedMovementStatus;
    Vec3 targetPos;
    AreaId targetAreaId;
    int64_t radialVelIndex;
    bool walk;
    bool crouch;
    bool valid;
};

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
    NUM_PRIORITY_SHOOT_OPTIONS [[maybe_unused]]
};

struct TargetPlayer {
    CSGOId playerId = INVALID_ID;
    int64_t round = INVALID_ID;
    int32_t firstTargetFrame = INVALID_ID;
    Vec3 footPos = {};
    Vec3 eyePos = {};
    bool visible = false;
};

enum class PriorityType {
    Order,
    Engagement,
    NUM_PRIORITY_TYPES [[maybe_unused]]
};

enum class NonDangerAimAreaType {
    Hold,
    Path,
    Learned,
    NUM_PRIORITY_TYPES [[maybe_unused]]
};

struct Priority {
    PriorityType priorityType;
    uint32_t targetAreaId;
    Vec3 targetPos;
    bool learnedTargetPos;
    std::optional<csknow::weapon_speed::MovementStatus> learnedMovementStatus;
    std::optional<bool> directPathToLearnedTargetPos;
    int numConsecutiveLearnedPathOverrides;
    TargetPlayer targetPlayer;
    NonDangerAimAreaType nonDangerAimAreaType;
    std::optional<AreaId> nonDangerAimArea;
    MoveOptions moveOptions;
    ShootOptions shootOptions;

    [[nodiscard]]
    string print(const ServerState & state) const {
        stringstream result;

        result << "priority type: ";
        if (priorityType == PriorityType::Order) {
            result << "Order, ";
        }
        else {
            result << "Engagement, ";
        }

        result << "target pos: (" << targetPos.toString() << "),";
        if (targetPlayer.playerId != INVALID_ID) {
            result << " target player id:" << state.getPlayerString(targetPlayer.playerId);
        }
        else {
            result << " target area id:" << targetAreaId;
        }
        if (nonDangerAimArea) {
            result << " non danger aim area id:" << nonDangerAimArea.value();
        }
        result << ", target player round: " << targetPlayer.round
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

        return result.str();
    }
};

#endif //CSKNOW_PRIORITY_DATA_H
