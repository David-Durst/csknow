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
    constexpr int num_z_axis_layers = 2;
    // add 1 for not moving bin
    constexpr int num_radial_bins_per_z_axis = num_directions * enumAsInt(StatureOptions::NUM_STATURE_OPTIONS);
    constexpr int num_radial_bins = 1 + num_z_axis_layers * num_radial_bins_per_z_axis;
    constexpr double speed_threshold = 0.98;

    double engineWeaponIdToMaxSpeed(EngineWeaponId engineWeaponId, StatureOptions statureOption, bool scoped);

    // plus 1 for standing still
    struct MovementStatus {
        Vec3 vel;
        StatureOptions statureOption;
        bool moving, jumping;
        int dir, zBin;

        MovementStatus(EngineWeaponId engineWeaponId, Vec3 curVel, Vec3 nextVel, StatureOptions statureOption,
                       bool scoped, bool airborne, bool jumping);
        MovementStatus(EngineWeaponId engineWeaponId, bool scoped, int radialMovementBin);

        int toRadialMovementBin() const;
    };
    int velocityToDir(Vec3 vel);


}

#endif //CSKNOW_WEAPON_SPEED_H
