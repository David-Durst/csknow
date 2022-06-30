//
// Created by durst on 5/2/22.
//

#include "bots/behavior_tree/order_node.h"
#include "geometryNavConversions.h"
#include <algorithm>

namespace order {
    void resetTreeThinkers(Blackboard & blackboard) {
        for (auto & [_, treeThinker] : blackboard.playerToTreeThinkers) {
            treeThinker.orderWaypointIndex = 0;
            treeThinker.orderGrenadeIndex = 0;
        }
    }

    /**
     * D2 assigns players to one of a couple known paths
     */
    NodeState D2OrderNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (state.mapName != "de_dust2") {
            playerNodeState[INVALID_ID] = NodeState::Failure;
            return NodeState::Failure;
        }


        if (playerNodeState.find(INVALID_ID) == playerNodeState.end() ||
            playerNodeState[INVALID_ID] != NodeState::Running || state.roundNumber != blackboard.planRoundNumber) {
            blackboard.planRoundNumber = state.roundNumber;

            // first setup orders to go A or B
            bool plantedA = blackboard.navFile.get_place(
                                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos())).m_place) == "BombsiteA";

            vector<vector<string>> ctPathPlaces;
            if (plantedA) {
                ctPathPlaces = {longToAPathPlaces, spawnToAPathPlaces, catToAPathPlace};
            }
            else {
                ctPathPlaces = {bDoorsToBPathPlaces, lowerTunsToBPathPlaces, outsideTunsToBPathPlaces};
            }
            map<string, size_t> ctPlacesToPath;
            for (size_t i = 0; i < ctPathPlaces.size(); i++) {
                for (const auto & pathPlace : ctPathPlaces[i]) {
                    ctPlacesToPath[pathPlace] = i;
                }
            }

            vector<vector<string>> tPathPlaces;
            for (const auto ctOnePathPlaces : ctPathPlaces) {
                vector<string> reversedPlaces(ctOnePathPlaces.rbegin(), ctOnePathPlaces.rend());
                tPathPlaces.push_back(reversedPlaces);
            }
            // going to combine ct and t paths, so increase the t indices for final merged list
            size_t tIndexOffset = ctPathPlaces.size();
            map<string, size_t> tPlacesToPath;
            for (size_t i = 0; i < tPathPlaces.size(); i++) {
                for (const auto & pathPlace : tPathPlaces[i]) {
                    tPlacesToPath[pathPlace] = i + tIndexOffset;
                }
            }

            vector<vector<string>> pathPlaces = ctPathPlaces;
            pathPlaces.insert(pathPlaces.end(), tPathPlaces.begin(), tPathPlaces.end());


            // clear orders before setting new ones
            blackboard.orders.clear();
            blackboard.playerToOrder.clear();
            blackboard.playerToPath.clear();
            blackboard.playerToPriority.clear();
            for (const auto & pathPlace : pathPlaces) {
                vector<Waypoint> waypoints;
                for (const auto & p : pathPlace) {
                    waypoints.push_back({WaypointType::NavPlace, p, INVALID_ID});
                }
                blackboard.orders.push_back({waypoints, {}, {}, {}});
            }

            // next assign clients to orders
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    map<string, size_t> & placesToPath = client.team == ENGINE_TEAM_T ? tPlacesToPath : ctPlacesToPath;

                    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
                            {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});

                    // get the nearest path start for the current team
                    double minDistance = std::numeric_limits<double>::max();
                    size_t orderIndex = INVALID_ID;

                    for (const auto & area : blackboard.navFile.m_areas) {
                        double newDistance = blackboard.getDistance(curArea.get_id(), area.get_id());
                        string newPlace = blackboard.navFile.get_place(area.m_place);
                        if (placesToPath.find(newPlace) != placesToPath.end() && newDistance < minDistance) {
                            minDistance = newDistance;
                            orderIndex = placesToPath[newPlace];
                        }
                    }

                    blackboard.orders[orderIndex].followers.push_back(client.csgoId);
                    blackboard.playerToOrder[client.csgoId] = orderIndex;
                }
            }

            // finally some house keeping
            resetTreeThinkers(blackboard);
            playerNodeState[INVALID_ID] = NodeState::Running;
        }
        return playerNodeState[INVALID_ID];
    }

    /**
     * General order assigns to kill
     */
    NodeState GeneralOrderNode::exec(const ServerState &state, TreeThinker &treeThinker) {

        if (playerNodeState.find(INVALID_ID) == playerNodeState.end() ||
            playerNodeState[INVALID_ID] != NodeState::Running) {
            map<CSGOId, Vec3> tIdsToPositions, ctIdsToPositions;

            for (const auto &client : state.clients) {
                if (client.isAlive) {
                    if (client.team == ENGINE_TEAM_T) {
                        tIdsToPositions[client.csgoId] = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
                    } else if (client.team == ENGINE_TEAM_CT) {
                        ctIdsToPositions[client.csgoId] = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
                    }
                }
            }

            // assign every player an enemy to attack, grouping together players with the same enemy under same order
            map<CSGOId, size_t> targetToOrderId;
            for (const auto &client : state.clients) {
                if (client.isAlive) {
                    const nav_mesh::nav_area &curArea = blackboard.navFile.get_nearest_area_by_position(
                            {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});
                    map<CSGOId, Vec3> &idsToPositions = client.team == T_TEAM ? ctIdsToPositions : tIdsToPositions;
                    double minDistance = std::numeric_limits<double>::max();
                    CSGOId minCSGOId = INVALID_ID;
                    for (const auto[enemyId, enemyPos] : idsToPositions) {
                        double newDistance = blackboard.getDistance(curArea.get_id(),
                                                                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(enemyPos)).get_id());
                        if (newDistance < minDistance) {
                            minDistance = newDistance;
                            minCSGOId = enemyId;
                        }
                    }

                    // if a teammate already has him as a target, group them by the same order
                    if (targetToOrderId.find(minCSGOId) != targetToOrderId.end()) {
                        size_t orderId = targetToOrderId[minCSGOId];
                        blackboard.playerToOrder[client.csgoId] = orderId;
                        blackboard.orders[orderId].followers.push_back(client.csgoId);
                    } else {
                        targetToOrderId[minCSGOId] = blackboard.orders.size();
                        blackboard.playerToOrder[client.csgoId] = blackboard.orders.size();
                        blackboard.orders.push_back({{{WaypointType::Player, "", minCSGOId}}, {}, {}, {client.csgoId}});
                    }
                }
            }

            resetTreeThinkers(blackboard);
            playerNodeState[INVALID_ID] = NodeState::Running;
        }
        else if (playerNodeState[INVALID_ID] == NodeState::Running) {
            // finish if anyone died (as need to repick everyone's target) or if round restarted
            if (state.clients.size() != blackboard.lastFrameState.clients.size()) {
                playerNodeState[INVALID_ID] = NodeState::Success;
            }
            else {
                for (size_t i = 0; i < state.clients.size(); i++) {
                    if (state.clients[i].isAlive != blackboard.lastFrameState.clients[i].isAlive) {
                        playerNodeState[INVALID_ID] = NodeState::Success;
                    }
                }
            }
        }

        return playerNodeState[INVALID_ID];
    }

}
