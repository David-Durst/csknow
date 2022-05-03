//
// Created by durst on 5/2/22.
//

#include "bots/behavior_tree/order_node.h"
#include "geometryNavConversions.h"
#include <algorithm>

/**
 * D2 assigns players to one of a couple known paths
 */
NodeState D2OrderTaskNode::exec(const ServerState &state, const TreeThinker &treeThinker) {
    if (state.mapName != "de_dust2") {
        this->nodeState = NodeState::Failure;
        return NodeState::Failure;
    }


    if (this->nodeState == NodeState::Uninitialized) {
        bool plantedA = blackboard.navFile.m_places[
                                blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos())).m_place] == "BombsiteA";

        vector<vector<string>> tPathPlaces;
        if (plantedA) {
            tPathPlaces = {
                { "LongDoors", "LongA", "ARamp", "BombsiteA" },
                { "CTSpawn", "UnderA", "ARamp", "BombsiteA" },
                { "Catwalk", "ShortStairs", "ExtendedA", "BombsiteA" },
            };
        }
        else {
            tPathPlaces = {
                    { "BDoors", "BombsiteB" },
                    { "LowerTunnel", "UpperTunnel", "BombsiteB" },
                    { "OutsideTunnel", "UpperTunnel", "BombsiteB" },
            };
        }
        set<string> tStartPlaces;
        for (const auto & pathPlaces : tPathPlaces) {
            tStartPlaces.insert(pathPlaces[0]);
        }

        vector<vector<string>> ctPathPlaces;
        for (const auto pathPlaces : tPathPlaces) {
            vector<string> reversedPlaces(pathPlaces.rbegin(), pathPlaces.rend());
            ctPathPlaces.push_back(reversedPlaces);
        }
        set<string> ctStartPlaces;
        for (const auto & pathPlaces : ctPathPlaces) {
            ctStartPlaces.insert(pathPlaces[0]);
        }

        vector<vector<string>> pathPlaces = tPathPlaces;
        pathPlaces.insert(pathPlaces.end(), ctPathPlaces.begin(), ctPathPlaces.end());

        map<string, size_t> startPlaceToOrderIndex;

        for (const auto & pathPlace : pathPlaces) {
            vector<Waypoint> waypoints;
            for (const auto & p : pathPlace) {
                waypoints.push_back({WaypointType::NavPlace, p, INVALID_ID});
            }
            startPlaceToOrderIndex[pathPlace[0]] = this->blackboard.orders.size();
            this->blackboard.orders.push_back({waypoints, {}, {}, 0});
        }

        for (const auto & client : state.clients) {
            if (client.isAlive && client.isBot) {
                set<string> & startPlaces = client.team == T_TEAM ? tStartPlaces : ctStartPlaces;

                // get the nearest path start for the current team
                double minDistance = std::numeric_limits<double>::max();
                string closestPlace;
                for (const auto & area : treeThinker.navFile.m_areas) {
                    double newDistance = computeDistance({client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ},
                                                         vec3tConv(area.get_center()));
                    string newPlace = blackboard.navFile.m_places[area.m_place];
                    if (startPlaces.find(newPlace) != startPlaces.end() && newDistance < minDistance) {
                        minDistance = newDistance;
                        closestPlace = newPlace;
                    }
                }

                size_t orderIndex = startPlaceToOrderIndex[closestPlace];
                this->blackboard.orders[orderIndex].numTeammates++;
                this->blackboard.playerToOrder[client.csgoId] = orderIndex;
            }
        }

        this->nodeState == NodeState::Success;
    }

    return NodeState::Success;
}

/**
 * General order assigns to kill
 */
NodeState GeneralOrderTaskNode::exec(const ServerState &state, const TreeThinker &treeThinker) {

    map<CSGOId, Vec3> tIdsToPositions, ctIdsToPositions;

    for (const auto & client : state.clients) {
        if (client.isAlive) {
            if (client.team == T_TEAM) {
                tIdsToPositions[client.csgoId] = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
            }
            else if (client.team == CT_TEAM) {
                ctIdsToPositions[client.csgoId] = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
            }
        }
    }

    map<CSGOId, size_t> targetToOrderId;
    for (const auto & client : state.clients) {
        if (client.isAlive) {
            Vec3 curPos = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
            map<CSGOId, Vec3> & idsToPositions = client.team == T_TEAM ? ctIdsToPositions : tIdsToPositions;
            double minDistance = std::numeric_limits<double>::max();
            CSGOId minCSGOId = INVALID_ID;
            for (const auto [enemyId, enemyPos] : idsToPositions) {
                double newDistance = computeDistance(curPos, enemyPos);
                if (newDistance < minDistance) {
                    minDistance = newDistance;
                    minCSGOId = enemyId;
                }
            }

            // if a teammate already has him as a target, group them by the same order
            if (targetToOrderId.find(minCSGOId) != targetToOrderId.end()) {
                size_t orderId = targetToOrderId[minCSGOId];
                blackboard.playerToOrder[client.csgoId] = orderId;
                blackboard.orders[orderId].numTeammates++;
            }
            else {
                targetToOrderId[minCSGOId] = blackboard.orders.size();
                blackboard.playerToOrder[client.csgoId] = blackboard.orders.size();
                blackboard.orders.push_back({{{WaypointType::Player,"", minCSGOId}}, {}, {}, 1});
            }
        }
    }

    this->nodeState == NodeState::Success;
    return NodeState::Success;
}
