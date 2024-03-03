//
// Created by durst on 7/14/23.
//

#include <glm/trigonometric.hpp>
#include "bots/analysis/weapon_speed.h"

namespace csknow::weapon_speed {
// https://old.reddit.com/r/GlobalOffensive/comments/a28h8r/movement_speed_chart/
    double engineWeaponIdToMaxSpeed(EngineWeaponId engineWeaponId, StatureOptions statureOption, bool scoped) {
        double maxWeaponSpeed;
        switch (engineWeaponId) {
            case (EngineWeaponId::Deagle):
                maxWeaponSpeed = 230.;
                break;
            case (EngineWeaponId::Dualies):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::FiveSeven):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::Glock):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::AK):
                maxWeaponSpeed = 215.;
                break;
            case (EngineWeaponId::AUG):
                if (scoped) {
                    maxWeaponSpeed = 150.;
                } else {
                    maxWeaponSpeed = 220.;
                }
                break;
            case (EngineWeaponId::AWP):
                if (scoped) {
                    maxWeaponSpeed = 100.;
                } else {
                    maxWeaponSpeed = 200.;
                }
                break;
            case (EngineWeaponId::FAMAS):
                maxWeaponSpeed = 220.;
                break;
            case (EngineWeaponId::G3):
                if (scoped) {
                    maxWeaponSpeed = 120.;
                } else {
                    maxWeaponSpeed = 215.;
                }
                break;
            case (EngineWeaponId::Galil):
                maxWeaponSpeed = 215.;
                break;
            case (EngineWeaponId::M249):
                maxWeaponSpeed = 195.;
                break;
            case (EngineWeaponId::M4A4):
                maxWeaponSpeed = 225.;
                break;
            case (EngineWeaponId::Mac10):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::P90):
                maxWeaponSpeed = 230.;
                break;
            case (EngineWeaponId::MP5):
                maxWeaponSpeed = 235.;
                break;
            case (EngineWeaponId::UMP):
                maxWeaponSpeed = 230.;
                break;
            case (EngineWeaponId::XM1014):
                maxWeaponSpeed = 215.;
                break;
            case (EngineWeaponId::Bizon):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::MAG7):
                maxWeaponSpeed = 225.;
                break;
            case (EngineWeaponId::Negev):
                maxWeaponSpeed = 150.;
                break;
            case (EngineWeaponId::SawedOff):
                maxWeaponSpeed = 210.;
                break;
            case (EngineWeaponId::Tec9):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::Zeus):
                maxWeaponSpeed = 220.;
                break;
            case (EngineWeaponId::P2000):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::MP7):
                maxWeaponSpeed = 220.;
                break;
            case (EngineWeaponId::MP9):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::Nova):
                maxWeaponSpeed = 220.;
                break;
            case (EngineWeaponId::P250):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::Scar):
                if (scoped) {
                    maxWeaponSpeed = 120.;
                } else {
                    maxWeaponSpeed = 215.;
                }
                break;
            case (EngineWeaponId::SG553):
                if (scoped) {
                    maxWeaponSpeed = 150.;
                } else {
                    maxWeaponSpeed = 210.;
                }
                break;
            case (EngineWeaponId::SSG):
                maxWeaponSpeed = 230.;
                break;
            case (EngineWeaponId::Flashbang):
                maxWeaponSpeed = 245.;
                break;
            case (EngineWeaponId::HEGrenade):
                maxWeaponSpeed = 245.;
                break;
            case (EngineWeaponId::Smoke):
                maxWeaponSpeed = 245.;
                break;
            case (EngineWeaponId::Molotov):
                maxWeaponSpeed = 245.;
                break;
            case (EngineWeaponId::Decoy):
                maxWeaponSpeed = 245.;
                break;
            case (EngineWeaponId::Incendiary):
                maxWeaponSpeed = 245.;
                break;
            case (EngineWeaponId::C4):
                maxWeaponSpeed = 250.;
                break;
            case (EngineWeaponId::M4A1S):
                maxWeaponSpeed = 225.;
                break;
            case (EngineWeaponId::USPS):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::CZ):
                maxWeaponSpeed = 240.;
                break;
            case (EngineWeaponId::R8):
                maxWeaponSpeed = 220.;
                break;
            default:
                // this catches all knifes
                maxWeaponSpeed = 250.;
                break;
        }

        double actualMaxSpeed = maxWeaponSpeed;
        if (statureOption == StatureOptions::Walking) {
            actualMaxSpeed *= walking_modifier;
        } else if (statureOption == StatureOptions::Ducking) {
            actualMaxSpeed *= ducking_modifier;
        }

        return actualMaxSpeed;
    }


    MovementStatus::MovementStatus(EngineWeaponId engineWeaponId, Vec3 curVel, Vec3 nextVel, StatureOptions statureOption,
                                   bool scoped, bool airborne, bool jumping) : vel(curVel),
                                   statureOption(statureOption), jumping(jumping) {
        (void) nextVel;
        double weaponMaxSpeed = engineWeaponIdToMaxSpeed(engineWeaponId, statureOption, scoped);
        // check if within threshold of moving or not moving. otherwise look ad delta in vel
        double movingSpeedThreshold = weaponMaxSpeed * stopped_threshold;
        // airborne check is for the 30 unit speed that you can accel from stopped while in air
        if (airborne) {
            movingSpeedThreshold = airwalk_speed * stopped_threshold;
        }

        Vec2 curVel2D{vel.x, vel.y};
        double speed2D = computeMagnitude(curVel2D);
        /*
        Vec2 nextVel2D{nextVel.x, nextVel.y};
        // if moving in same direciton and magnitude decreases, then stopping
        // otherwise just changing direciton
        bool increasingVel = (computeMagnitude(nextVel2D) - computeMagnitude(curVel2D)) > 0;
        // must check for 0, don't want a divide by 0
        bool curVelZero = computeMagnitude(curVel2D) == 0;
        bool nextVelZero = computeMagnitude(nextVel2D) == 0;
        // no direction for not moving
        bool sameDir = (curVelZero != nextVelZero) || angleBetween(curVel2D, nextVel2D) < 10.;

        // either at max speed currently
        // or changing to a non-zero direction
        // or increasing speed
        if (speed2D >= movingSpeedThreshold || (!nextVelZero && !sameDir) || increasingVel) {
         */
        if (speed2D >= movingSpeedThreshold || (jumping && speed2D > 0.)) {
            moving = true;
            dir = velocityToDir(vel);
        }
        else {
            moving = false;
            dir = 0;
        }
        if (!moving) {
            zBin = 0;
        }
        else if (jumping) {
            zBin = 1;
        }
        else {
            zBin = 0;
        }
    }

    MovementStatus::MovementStatus(EngineWeaponId engineWeaponId, bool scoped, int radialMovementBin) {
        if (radialMovementBin == 0) {
            vel = {0., 0., 0.};
            zBin = 0;
            statureOption = StatureOptions::Standing;
            moving = false;
            dir = 0;
            jumping = false;
        }
        else {
            int movementBin3D = radialMovementBin - 1;
            zBin = movementBin3D / num_radial_bins_per_z_axis;
            moving = true;
            jumping = zBin == 1;
            int dirStatureRadialIndex = movementBin3D % num_radial_bins_per_z_axis;
            dir = dirStatureRadialIndex / enumAsInt(StatureOptions::NUM_STATURE_OPTIONS);
            statureOption =
                    static_cast<StatureOptions>(dirStatureRadialIndex % enumAsInt(StatureOptions::NUM_STATURE_OPTIONS));
            double maxSpeed = engineWeaponIdToMaxSpeed(engineWeaponId, statureOption, scoped);
            double velAngle = dir * direction_angle_range;
            vel = {
                    std::cos(glm::radians(velAngle)) * maxSpeed,
                    std::sin(glm::radians(velAngle)) * maxSpeed,
                    0.
            };
        }
    }

    int MovementStatus::toRadialMovementBin() const {
        if (!moving) {
            return 0;
        }
        else {
            int zAxisIndex = jumping ? 1 : 0;
            return 1 + zAxisIndex * num_directions * enumAsInt(StatureOptions::NUM_STATURE_OPTIONS) +
                dir * enumAsInt(StatureOptions::NUM_STATURE_OPTIONS) + enumAsInt(statureOption);
        }
    }

    int velocityToDir(Vec3 vel) {
        double velAngle = glm::degrees(std::atan2(vel.y, vel.x));
        // make velAngle always positive
        if (velAngle < 0.) {
            velAngle = 360. + velAngle;
        }
        // adjust by half angle range so that first angle range is centered around 0.
        return static_cast<int>((velAngle + direction_angle_range / 2.) / direction_angle_range) % num_directions;
    }
}