//
// Created by durst on 5/2/22.
//

#include "bots/behavior_tree/order_node.h"
#include "geometryNavConversions.h"
#include <algorithm>

namespace order {
    void resetTreeThinkers(Blackboard & blackboard) {
        for (auto & [_, treeThinker] : blackboard.treeThinkers) {
            treeThinker.orderWaypointIndex = 0;
            treeThinker.orderGrenadeIndex = 0;
        }
    }

    /**
     * D2 assigns players to one of a couple known paths
     */
    NodeState D2TaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (state.mapName != "de_dust2") {
            this->nodeState = NodeState::Failure;
            return NodeState::Failure;
        }


        if (this->nodeState != NodeState::Running) {
            // first setup orders to go A or B
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
            map<string, size_t> tPlacesToPath;
            for (size_t i = 0; i < tPathPlaces.size(); i++) {
                for (const auto & pathPlace : tPathPlaces[i]) {
                    tPlacesToPath[pathPlace] = i;
                }
            }

            vector<vector<string>> ctPathPlaces;
            for (const auto pathPlaces : tPathPlaces) {
                vector<string> reversedPlaces(pathPlaces.rbegin(), pathPlaces.rend());
                ctPathPlaces.push_back(reversedPlaces);
            }
            map<string, size_t> ctPlacesToPath;
            for (size_t i = 0; i < ctPathPlaces.size(); i++) {
                for (const auto & pathPlace : ctPathPlaces[i]) {
                    ctPlacesToPath[pathPlace] = i;
                }
            }

            vector<vector<string>> pathPlaces = tPathPlaces;
            size_t ctOrderOffset = pathPlaces.size();
            pathPlaces.insert(pathPlaces.end(), ctPathPlaces.begin(), ctPathPlaces.end());


            for (const auto & pathPlace : pathPlaces) {
                vector<Waypoint> waypoints;
                for (const auto & p : pathPlace) {
                    waypoints.push_back({WaypointType::NavPlace, p, INVALID_ID});
                }
                this->blackboard.orders.push_back({waypoints, {}, {}, 0});
            }

            // next assign clients to orders
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    map<string, size_t> & placesToPath = client.team == T_TEAM ? tPlacesToPath : ctPlacesToPath;

                    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
                            {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});

                    // get the nearest path start for the current team
                    double minDistance = std::numeric_limits<double>::max();
                    size_t orderIndex = INVALID_ID;

                    for (const auto & area : blackboard.navFile.m_areas) {
                        double newDistance = blackboard.reachability.getDistance(curArea.get_id(), area.get_id());
                        string newPlace = blackboard.navFile.m_places[area.m_place];
                        if (placesToPath.find(newPlace) != placesToPath.end() && newDistance < minDistance) {
                            minDistance = newDistance;
                            orderIndex = placesToPath[newPlace];
                        }
                    }

                    this->blackboard.orders[orderIndex].numTeammates++;
                    this->blackboard.playerToOrder[client.csgoId] = orderIndex;
                }
            }

            // finally some house keeping
            resetTreeThinkers(blackboard);
            this->nodeState = NodeState::Running;
        }
        else if (this->nodeState == NodeState::Running && state.roundNumber != blackboard.lastFrameState.roundNumber) {
            this->nodeState = NodeState::Success;
        }
        return this->nodeState;
    }

    /**
     * General order assigns to kill
     */
    NodeState GeneralTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {

        if (this->nodeState != NodeState::Running) {
            map<CSGOId, Vec3> tIdsToPositions, ctIdsToPositions;

            for (const auto &client : state.clients) {
                if (client.isAlive) {
                    if (client.team == T_TEAM) {
                        tIdsToPositions[client.csgoId] = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
                    } else if (client.team == CT_TEAM) {
                        ctIdsToPositions[client.csgoId] = {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
                    }
                }
            }

            map<CSGOId, size_t> targetToOrderId;
            for (const auto &client : state.clients) {
                if (client.isAlive) {
                    const nav_mesh::nav_area &curArea = blackboard.navFile.get_nearest_area_by_position(
                            {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});
                    map<CSGOId, Vec3> &idsToPositions = client.team == T_TEAM ? ctIdsToPositions : tIdsToPositions;
                    double minDistance = std::numeric_limits<double>::max();
                    CSGOId minCSGOId = INVALID_ID;
                    for (const auto[enemyId, enemyPos] : idsToPositions) {
                        double newDistance = blackboard.reachability.getDistance(curArea.get_id(),
                                                                                 blackboard.navFile.get_nearest_area_by_position(
                                                                                         vec3Conv(enemyPos)).get_id());
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
                    } else {
                        targetToOrderId[minCSGOId] = blackboard.orders.size();
                        blackboard.playerToOrder[client.csgoId] = blackboard.orders.size();
                        blackboard.orders.push_back({{{WaypointType::Player, "", minCSGOId}}, {}, {}, 1});
                    }
                }
            }

            resetTreeThinkers(blackboard);
            this->nodeState = NodeState::Running;
        }
        else if (this->nodeState == NodeState::Running) {
            // finish if anyone died (as need to repick everyone's target) or if round restarted
            if (state.clients.size() != blackboard.lastFrameState.clients.size()) {
                this->nodeState = NodeState::Success;
            }
            else {
                for (size_t i = 0; i < state.clients.size(); i++) {
                    if (state.clients[i].isAlive != blackboard.lastFrameState.clients[i].isAlive) {
                        this->nodeState = NodeState::Success;
                    }
                }
            }
        }

        return this->nodeState;
    }

}
