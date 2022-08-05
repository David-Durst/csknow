//
// Created by durst on 7/24/22.
//
#include "bots/behavior_tree/global/strategy_node.h"
namespace strategy {
    struct OrderPlaceDistance {
        OrderId orderId;
        string place;
        double distance;
        bool assignedPlayer = false;
    };

    NodeState AssignPlayersToOrders::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            vector<OrderPlaceDistance> tOptions, ctOptions;
            // build the Order Place options, will compute distance for each client
            for (const auto & orderId : blackboard.strategy.getOrderIds(false, true)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::NavPlace || waypoint.type == WaypointType::NavAreas || waypoint.type == WaypointType::C4) {
                        ctOptions.push_back({orderId, waypoint.placeName});
                    }
                }
            }
            if (ctOptions.empty()) {
                throw std::runtime_error("no ct orders with a nav place, nav area, or c4");
            }

            for (const auto & orderId : blackboard.strategy.getOrderIds(true, false)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::HoldPlace || waypoint.type == WaypointType::HoldAreas || waypoint.type == WaypointType::C4) {
                        tOptions.push_back({orderId, waypoint.placeName});
                    }
                }
            }
            if (tOptions.empty()) {
                throw std::runtime_error("no t orders with a hold place, hold area, or c4");
            }

            // for CT, compute distance to each waypoint in each order and assign player to order with closest waypoint
            // for T, same thing but also considering covering every order - nearest unassigned order, or nearest unassigned if all assigned
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
                            {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});

                    vector<OrderPlaceDistance> & options = client.team == ENGINE_TEAM_T ? tOptions : ctOptions;

                    for (auto & option : options) {
                        option.distance = blackboard.distanceToPlaces.getClosestDistance(curArea.get_id(), option.place,
                                                                                  blackboard.navFile);
                    }

                    if (client.team == ENGINE_TEAM_CT) {
                        std::sort(options.begin(), options.end(),
                                  [](const OrderPlaceDistance & a, const OrderPlaceDistance & b){ return a.distance < b.distance; });
                    }
                    else if (client.team == ENGINE_TEAM_T) {
                        std::sort(options.begin(), options.end(),
                                  [](const OrderPlaceDistance & a, const OrderPlaceDistance & b){
                            return (!a.assignedPlayer && b.assignedPlayer) || (a.assignedPlayer == b.assignedPlayer && a.distance < b.distance);
                        });
                    }

                    blackboard.strategy.assignPlayerToOrder(client.csgoId, options[0].orderId);
                    options[0].assignedPlayer = true;
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
