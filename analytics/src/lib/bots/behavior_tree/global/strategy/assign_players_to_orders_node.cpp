//
// Created by durst on 7/24/22.
//
#include "bots/behavior_tree/global/strategy_node.h"
#include "bots/behavior_tree/pathing_node.h"
namespace strategy {
    struct OrderPlaceDistance {
        OrderId orderId;
        string place;
        double distance;
        bool assignedPlayer = false;
    };

    NodeState AssignPlayersToOrders::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            map<string, OrderId> ctOptions;
            vector<OrderPlaceDistance> tOptions;
            // build the Order Place options, will compute distance for each client
            for (const auto & orderId : blackboard.strategy.getOrderIds(false, true)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::NavPlace) {
                        ctOptions[waypoint.placeName] = orderId;
                    }
                    else if (waypoint.type == WaypointType::NavAreas) {
                        for (const auto & areaId : waypoint.areaIds) {
                            string placeName = blackboard.navFile.get_place(
                                    blackboard.navFile.get_area_by_id_fast(areaId).m_place);
                            if (placeName != "") {
                                ctOptions[placeName] = orderId;
                            }
                        }
                    }
                }
            }
            if (ctOptions.empty()) {
                throw std::runtime_error("no ct orders with a nav place or nav area");
            }

            for (const auto & orderId : blackboard.strategy.getOrderIds(true, false)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::HoldPlace) {
                        tOptions.push_back({orderId, waypoint.placeName});
                    }
                    else if (waypoint.type == WaypointType::HoldAreas) {
                        for (const auto & areaId : waypoint.areaIds) {
                            string placeName = blackboard.navFile.get_place(
                                    blackboard.navFile.get_area_by_id_fast(areaId).m_place);
                            if (placeName != "") {
                                // ok to push same placename multiple times, sort will just take first
                                tOptions.push_back({orderId, placeName});
                            }
                        }
                    }
                }
            }
            if (tOptions.empty()) {
                throw std::runtime_error("no t orders with a hold place or hold area");
            }

            // for CT, compute distance to each waypoint in each order and assign player to order with closest waypoint
            // for T, same thing but also considering covering every order - nearest unassigned order, or nearest unassigned if all assigned
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    if (client.team == ENGINE_TEAM_CT) {
                        if (blackboard.inTest && client.name == "Adam") {
                            int x = 1;
                        }
                        Path pathToC4 = movement::computePath(state, blackboard, vec3Conv(state.getC4Pos()), client);
                        for (PathNode pathNode : pathToC4.waypoints) {
                            string placeName = blackboard.navFile.get_place(
                                    blackboard.navFile.get_area_by_id_fast(pathNode.area1).m_place);
                            if (ctOptions.find(placeName) != ctOptions.end()) {
                                blackboard.strategy.assignPlayerToOrder(client.csgoId, ctOptions[placeName]);
                                break;
                            }
                        }
                    }

                    else if (client.team == ENGINE_TEAM_T) {
                        const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
                                {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});
                        for (auto & option : tOptions) {
                            option.distance = blackboard.distanceToPlaces.getClosestDistance(curArea.get_id(), option.place,
                                                                                             blackboard.navFile);
                        }
                        std::sort(tOptions.begin(), tOptions.end(),
                                  [](const OrderPlaceDistance & a, const OrderPlaceDistance & b){
                            return (!a.assignedPlayer && b.assignedPlayer) || (a.assignedPlayer == b.assignedPlayer && a.distance < b.distance);
                        });
                        blackboard.strategy.assignPlayerToOrder(client.csgoId, tOptions[0].orderId);
                        tOptions[0].assignedPlayer = true;
                    }
                }
            }
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    blackboard.strategy.getOrderIdForPlayer(client.csgoId);
                }
            }


        }
        else {
            blackboard.newOrderThisFrame = false;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
