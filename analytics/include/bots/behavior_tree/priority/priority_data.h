//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_PRIORITY_DATA_H
#define CSKNOW_PRIORITY_DATA_H
#include "load_save_bot_data.h"

enum class PriorityType {
   NavArea,
   Player,
   C4,
   NUM_PRIORITY_TYPES
};

struct PriorityMovementOptions {
    bool move;
    bool walk;
    bool crouch;
};

enum class PriorityShootOptions {
    DontShoot,
    Tap,
    Burst,
    Spray,
    NUM_PRIORITY_SHOOT_OPTIONS
};

struct Priority {
    PriorityType priorityType;
    uint32_t areaId;
    CSGOId playerId;
    PriorityMovementOptions movementOptions;
    PriorityShootOptions shootOptions;


    Vec3 getTargetPos(const ServerState & state, const nav_mesh::nav_file & navFile) {
        if (priorityType == PriorityType::NavArea) {
            return vec3tConv(navFile.get_area_by_id(areaId).get_center());
        }
        else if (priorityType == PriorityType::Player) {
            return state.clients[state.csgoIdToCSKnowId[playerId]].getFootPosForPlayer();
        }
        else {
            return state.getC4Pos();
        }
    }

    const nav_mesh::nav_area & getTargetNavArea(const ServerState & state, const nav_mesh::nav_file & navFile) {
        if (priorityType == PriorityType::NavArea) {
            return navFile.get_area_by_id_fast(areaId);
        }
        else if (priorityType == PriorityType::Player) {
            return navFile.get_nearest_area_by_position(
                    vec3Conv(state.clients[state.csgoIdToCSKnowId[playerId]].getFootPosForPlayer()));
        }
        else {
            return navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos()));
        }
    }
};

#endif //CSKNOW_PRIORITY_DATA_H
