//
// Created by durst on 7/24/22.
//
#include "bots/behavior_tree/global/strategy_node.h"
namespace strategy {
    struct OrderPlaceDistance {
        OrderId orderId;
        string place;
        double distance;
    };

    NodeState AssignPlayersToOrders::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            vector<OrderPlaceDistance> tOptions, ctOptions;
            // build the Order Place options, will compute distance for each client
            for (const auto & orderId : blackboard.strategy.getOrderIds(true, false)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::NavPlace || waypoint.type == WaypointType::C4) {
                        tOptions.push_back({orderId, waypoint.placeName});
                    }
                }
            }
            if (tOptions.empty()) {
                throw std::runtime_error("no ct orders with a nav place or c4");
            }

            for (const auto & orderId : blackboard.strategy.getOrderIds(false, true)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::NavPlace || waypoint.type == WaypointType::C4) {
                        ctOptions.push_back({orderId, waypoint.placeName});
                    }
                }
            }
            if (ctOptions.empty()) {
                throw std::runtime_error("no ct orders with a nav place or c4");
            }

            // next compute distance to each waypoint in each order and assign player to order with closest waypoint
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
                            {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ});

                    vector<OrderPlaceDistance> & options = client.team == ENGINE_TEAM_T ? tOptions : ctOptions;

                    for (auto & option : options) {
                        option.distance = blackboard.distanceToPlaces.getDistance(curArea.get_id(), option.place,
                                                                                  blackboard.navFile);
                    }

                    std::sort(options.begin(), options.end(),
                              [](const OrderPlaceDistance & a, const OrderPlaceDistance & b){ return a.distance < b.distance; });

                    blackboard.strategy.assignPlayerToOrder(client.csgoId, options[0].orderId);
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
