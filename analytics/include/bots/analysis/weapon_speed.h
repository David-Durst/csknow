//
// Created by durst on 7/14/23.
//

#ifndef CSKNOW_WEAPON_SPEED_H
#define CSKNOW_WEAPON_SPEED_H

#include "bots/analysis/weapon_id_converter.h"
#include "bots/load_save_bot_data.h"

constexpr double walkingModifier = 0.52;
constexpr double crouchingModifier = 0.34;
constexpr double airwalkSpeed = 30.;
constexpr int numDirections = 8;
constexpr double directionAngleRange = 360. / numDirections;

enum class StatureOptions {
    Standing,
    Walking,
    Crouching,
};

double engineWeaponIdToMaxSpeed(EngineWeaponId engineWeaponId, StatureOptions statureOption, bool scoped);

enum class MovementTypes {
    Crouching = 0,
    Walking = 1,
    Running = 2,
    NUM_MOVEMENT_TYPES
};
struct MovementStatus {
    MovementTypes movementType;
    bool moving;
};
MovementStatus getMovementType(EngineWeaponId engineWeaponId, Vec3 vel, StatureOptions statureOption, bool scoped, bool airborne);

int velocityToDir(Vec3 vel);
Vec3 movementTypeAndDirToVel(MovementStatus movementStatus, int dir, EngineWeaponId engineWeaponId, bool scoped);

#endif //CSKNOW_WEAPON_SPEED_H
