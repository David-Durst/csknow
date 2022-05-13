//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_IMPLEMENTATION_DATA_H
#define CSKNOW_IMPLEMENTATION_DATA_H
#include "load_save_bot_data.h"
#include "bots/behavior_tree/priority/priority_data.h"

struct PathNode {
    bool edgeMidpoint;
    uint32_t area1;
    uint32_t area2;
    Vec3 pos;

    PathNode(nav_mesh::PathNode pathNode) {
        edgeMidpoint = pathNode.edgeMidpoint;
        area1 = pathNode.area1;
        area2 = pathNode.area2;
        pos = vec3tConv(pathNode.pos);
    }
};

struct Path {
    bool pathCallSucceeded;
    vector<PathNode> waypoints;
    uint32_t pathEndAreaId;
    size_t curWaypoint;

    string print(const ServerState & state, const nav_mesh::nav_file & navFile) const {
        stringstream result;

        result << "path call succeeded: " + boolToString(pathCallSucceeded);
        if (pathCallSucceeded) {
            result << ", ";
            if (curWaypoint >= 0 && curWaypoint < waypoints.size()) {
                result << "cur waypoint: " << curWaypoint << " " << waypoints[curWaypoint].pos.toString() << ", ";
            }
            else {
                result << "invalid waypoint, ";
            }
            result << "end waypoint: " << waypoints.size() - 1 << " " << vec3tConv(navFile.get_area_by_id_fast(pathEndAreaId).get_center()).toString();

        }

        return result.str();
    }
};

#endif //CSKNOW_IMPLEMENTATION_DATA_H
