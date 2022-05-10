//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_IMPLEMENTATION_DATA_H
#define CSKNOW_IMPLEMENTATION_DATA_H
#include "load_save_bot_data.h"
#include "bots/behavior_tree/priority/priority_data.h"

struct Path {
    bool pathCallSucceeded;
    vector<Vec3> waypoints;
    uint32_t pathEndAreaId;
    size_t curWaypoint;

    string print(const ServerState & state, const nav_mesh::nav_file & navFile) const {
        stringstream result;

        result << boolToString(pathCallSucceeded) << ", ";
        if (curWaypoint > 0 && curWaypoint < waypoints.size()) {
            result << waypoints[curWaypoint].toString() << ", ";
        }
        else {
            result << "invalid waypoint, ";
        }
        result << pathEndAreaId << ", " << vec3tConv(navFile.get_area_by_id_fast(pathEndAreaId).get_center()).toString();

        return result.str();
    }
};

#endif //CSKNOW_IMPLEMENTATION_DATA_H
