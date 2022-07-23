//
// Created by durst on 5/2/22.
//

#ifndef CSKNOW_ORDER_DATA_H
#define CSKNOW_ORDER_DATA_H
#include "load_save_bot_data.h"
#include "navmesh/nav_file.h"
#include <sstream>
using std::stringstream;
using std::map;

enum class WaypointType {
    NavPlace,
    Player,
    C4,
    NUM_WAYPOINTS
};

struct Waypoint {
    WaypointType waypointType;
    string placeName;
    CSGOId playerId;
};

struct Order {
    vector<Waypoint> waypoints;
    vector<CSGOId> followers;

    vector<string> print(const map<CSGOId, int64_t> & playerToCurWaypoint, const map<CSGOId, int32_t> & playerToPushOrder, const ServerState & state, size_t orderIndex) const {
        vector<string> result;
        stringstream waypointsStream;
        waypointsStream << orderIndex << " waypoints: [";
        for (const auto & waypoint : waypoints) {
            string typeString;
            switch (waypoint.waypointType) {
                case WaypointType::NavPlace:
                    typeString += "NavPlace";
                    break;
                case WaypointType::Player:
                    typeString += "Player";
                    break;
                case WaypointType::C4:
                    typeString += "C4";
                    break;
                default:
                    typeString += "INVALID_TYPE";
            }
            waypointsStream << "(" << typeString << "," << waypoint.placeName
                            << "," << state.getPlayerString(waypoint.playerId) << ")";
        }
        waypointsStream << "]";
        result.push_back(waypointsStream.str());

        stringstream followersStream;
        followersStream << orderIndex << " followers: <follower, waypoint index, push index> : [";
        for (const auto & follower : followers) {
            followersStream << "<" << state.getPlayerString(follower) << ", "
                << playerToCurWaypoint.find(follower)->second << ", "
                << playerToPushOrder.find(follower)->second << ">; ";
        }
        followersStream << "]";
        result.push_back(followersStream.str());

        return result;
    }
};

#endif //CSKNOW_ORDER_DATA_H
