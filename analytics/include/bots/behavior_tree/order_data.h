//
// Created by durst on 5/2/22.
//

#ifndef CSKNOW_ORDER_DATA_H
#define CSKNOW_ORDER_DATA_H
#include "load_save_bot_data.h"
#include "navmesh/nav_file.h"
using std::map;

struct GrenadeThrow {
    CSGOId thrower;
    Vec3 origin;
    Vec2 angle;
    Vec3 target;
    bool thrown;
    // for now assume always smoke
};

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
    vector<GrenadeThrow> grenadeThrows;
    map<CSGOId, vector<int64_t>> playerToGrenades;
    int16_t numTeammates;
};

#endif //CSKNOW_ORDER_DATA_H
