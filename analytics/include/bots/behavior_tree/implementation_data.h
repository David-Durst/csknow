//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_IMPLEMENTATION_DATA_H
#define CSKNOW_IMPLEMENTATION_DATA_H
#include "load_save_bot_data.h"
#include "bots/behavior_tree/priority/priority_data.h"

struct PathMovementOptions {
    bool move;
    bool walk;
    bool crouch;
};

enum class PathShootOptions {
    DontShoot,
    Tap,
    Burst,
    Spray,
    NUM_PRIORITY_SHOOT_OPTIONS
};

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
    PathMovementOptions movementOptions;
    PathShootOptions shootOptions;

    string print(const ServerState & state, const nav_mesh::nav_file & navFile) const {
        stringstream result;

        string shootOptionStr;
        switch (shootOptions) {
            case PathShootOptions::DontShoot:
                shootOptionStr = "DontShoot";
                break;
            case PathShootOptions::Tap:
                shootOptionStr = "Tap";
                break;
            case PathShootOptions::Burst:
                shootOptionStr = "Burst";
                break;
            case PathShootOptions::Spray:
                shootOptionStr = "Spray";
                break;
            default:
                shootOptionStr = "Invalid";
        }

        result << "path call succeeded: " + boolToString(pathCallSucceeded);
        if (pathCallSucceeded) {
            result << ", ";
            if (curWaypoint >= 0 && curWaypoint < waypoints.size()) {
                result << "cur waypoint: " << curWaypoint << " " << waypoints[curWaypoint].toString() << ", ";
            }
            else {
                result << "invalid waypoint, ";
            }
            result << "end waypoint: " << waypoints.size() - 1 << " " << vec3tConv(navFile.get_area_by_id_fast(pathEndAreaId).get_center()).toString();
        }

        result << ", move: " << boolToString(movementOptions.move) << ", walk: " << boolToString(movementOptions.walk)
            << ", crouch: " << boolToString(movementOptions.crouch) << ", shoot option: " << shootOptionStr;

        return result.str();
    }
};

#endif //CSKNOW_IMPLEMENTATION_DATA_H
