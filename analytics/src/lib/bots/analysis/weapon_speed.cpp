//
// Created by durst on 7/14/23.
//

#include <glm/trigonometric.hpp>
#include "bots/analysis/weapon_speed.h"

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
            }
            else {
                maxWeaponSpeed = 220.;
            }
            break;
        case (EngineWeaponId::AWP):
            if (scoped) {
                maxWeaponSpeed = 100.;
            }
            else {
                maxWeaponSpeed = 200.;
            }
            break;
        case (EngineWeaponId::FAMAS):
            maxWeaponSpeed = 220.;
            break;
        case (EngineWeaponId::G3):
            if (scoped) {
                maxWeaponSpeed = 120.;
            }
            else {
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
            }
            else {
                maxWeaponSpeed = 215.;
            }
            break;
        case (EngineWeaponId::SG553):
            if (scoped) {
                maxWeaponSpeed = 150.;
            }
            else {
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
        actualMaxSpeed *= walkingModifier;
    }
    else if (statureOption == StatureOptions::Crouching) {
        actualMaxSpeed *= crouchingModifier;
    }

    return actualMaxSpeed;
}

MovementStatus getMovementType(EngineWeaponId engineWeaponId, Vec3 vel, StatureOptions statureOption, bool scoped, bool airborne) {
    double weaponMaxSpeed = engineWeaponIdToMaxSpeed(engineWeaponId, statureOption, scoped);
    Vec2 vel2D{vel.x, vel.y};
    double speed2D = computeMagnitude(vel2D);
    // airborne check is for the 30 unit speed that you can accel from stopped while in air
    bool moving = speed2D >= weaponMaxSpeed / 2. || (airborne && speed2D >= airwalkSpeed / 2.);
    if (statureOption == StatureOptions::Standing) {
        return {MovementTypes::Running, moving};
    }
    else if (statureOption == StatureOptions::Walking) {
        return {MovementTypes::Walking, moving};
    }
    else if (statureOption == StatureOptions::Crouching) {
        return {MovementTypes::Crouching, moving};
    }
}

int velocityToDir(Vec3 vel) {
    double velAngle = glm::degrees(std::atan2(vel.y, vel.x));
    // adjust by half angle range so that first angle range is centered around 0.
    return static_cast<int>((velAngle + directionAngleRange / 2.) / directionAngleRange) % numDirections;
}

Vec3 movementTypeAndDirToVel(MovementStatus movementStatus, int dir, EngineWeaponId engineWeaponId, bool scoped) {
    StatureOptions statureOption;
    if (!movementStatus.moving) {
        return {0., 0., 0.};
    }
    else if (movementStatus.movementType == MovementTypes::Crouching) {
        statureOption = StatureOptions::Crouching;
    }
    else if (movementStatus.movementType == MovementTypes::Walking) {
        statureOption = StatureOptions::Walking;
    }
    else {
        statureOption = StatureOptions::Standing;
    }
    double speed = engineWeaponIdToMaxSpeed(engineWeaponId, statureOption, scoped);

    double velAngle = (static_cast<double>(dir) - 0.5) * directionAngleRange;
    return {
        std::sin(glm::radians(velAngle)) * speed,
        std::cos(glm::radians(velAngle)) * speed,
        0.
    };
}
