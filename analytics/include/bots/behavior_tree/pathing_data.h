//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_PATHING_DATA_H
#define CSKNOW_PATHING_DATA_H
#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/priority/priority_data.h"

struct PathNode {
    bool edgeMidpoint;
    uint32_t area1;
    uint32_t area2;
    Vec3 pos;

    explicit PathNode(nav_mesh::PathNode pathNode) :
        edgeMidpoint(pathNode.edgeMidpoint),
        area1(pathNode.area1),
        area2(pathNode.area2),
        pos(vec3tConv(pathNode.pos)) { }

    [[nodiscard]]
    string toString() const {
        string result = boolToString(edgeMidpoint) + " " + std::to_string(area1);
        if (edgeMidpoint) {
            result += " -> " + std::to_string(area2);
        }
        result += " " + pos.toString();
        return result;
    }
};

struct Path {
    bool pathCallSucceeded;
    vector<PathNode> waypoints;
    set<uint32_t> areas;
    uint32_t pathEndAreaId;
    size_t curWaypoint;

    [[nodiscard]]
    string print() const {
        stringstream result;

        result << "path call succeeded: " + boolToString(pathCallSucceeded);
        if (pathCallSucceeded) {
            result << ", ";
            if (curWaypoint < waypoints.size()) {
                result << "cur waypoint: " << curWaypoint << " " << waypoints[curWaypoint].toString() << ", ";
            }
            else {
                result << "invalid waypoint, ";
            }
            result << "end waypoint: " << waypoints.size() - 1 << " " << waypoints.back().toString();
        }

        return result.str();
    }
};

#endif //CSKNOW_PATHING_DATA_H
