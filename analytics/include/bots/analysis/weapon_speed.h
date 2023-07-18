//
// Created by durst on 7/14/23.
//

#ifndef CSKNOW_WEAPON_SPEED_H
#define CSKNOW_WEAPON_SPEED_H

#include "bots/analysis/weapon_id_converter.h"
#include "bots/load_save_bot_data.h"
#include "enum_helpers.h"

namespace csknow::weapon_speed {
    enum class StatureOptions {
        Standing = 0,
        Walking,
        Ducking,
        NUM_STATURE_OPTIONS
    };

    constexpr double walking_modifier = 0.52;
    constexpr double ducking_modifier = 0.34;
    constexpr double airwalk_speed = 30.;
    constexpr int num_directions = 16;
    constexpr double direction_angle_range = 360. / num_directions;
    constexpr int num_z_axis_layers = 3;
    constexpr int num_radial_bins = num_z_axis_layers * num_directions * enumAsInt(StatureOptions::NUM_STATURE_OPTIONS);
    constexpr double speed_threshold = 0.9;

    double engineWeaponIdToMaxSpeed(EngineWeaponId engineWeaponId, StatureOptions statureOption, bool scoped);

    // plus 1 for standing still
    constexpr int radial_movement_options = num_directions * enumAsInt(StatureOptions::NUM_STATURE_OPTIONS) + 1;
    struct MovementStatus {
        Vec3 vel;
        StatureOptions statureOption;
        bool moving, jumping, falling;
        int dir;

        MovementStatus(EngineWeaponId engineWeaponId, Vec3 curVel, Vec3 nextVel, StatureOptions statureOption,
                       bool scoped, bool airborne, bool jumping, bool falling);
        MovementStatus(EngineWeaponId engineWeaponId, bool scoped, int radialMovementBin);

        int velocityToDir(Vec3 vel);
        int toRadialMovementBin();
        Vec3 movementTypeAndDirToVel();
    };


}

#endif //CSKNOW_WEAPON_SPEED_H
