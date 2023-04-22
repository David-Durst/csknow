//
// Created by durst on 7/24/22.
//
#include "bots/behavior_tree/global/strategy_node.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/analysis/learned_models.h"

namespace strategy {
    struct OrderPlaceDistance {
        OrderId orderId;
        string place;
        double distance = -1.;
        bool assignedPlayer = false;
    };

    NodeState AssignPlayersToOrders::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            map<string, OrderId> ctOptions;
            OrderId defaultCTOption{INVALID_ID, INVALID_ID};
            vector<OrderPlaceDistance> tOptions;
            // build the Order Place options, will compute distance for each client
            for (const auto & orderId : blackboard.strategy.getOrderIds(false, true)) {
                defaultCTOption = orderId;
                const Order & order = blackboard.strategy.getOrder(orderId);
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::NavPlace) {
                        ctOptions[waypoint.placeName] = orderId;
                    }
                    else if (waypoint.type == WaypointType::NavAreas) {
                        for (const auto & areaId : waypoint.areaIds) {
                            string placeName = blackboard.navFile.get_place(
                                    blackboard.navFile.get_area_by_id_fast(areaId).m_place);
                            if (!placeName.empty()) {
                                ctOptions[placeName] = orderId;
                            }
                        }
                    }
                }
            }
            if (ctOptions.empty()) {
                throw std::runtime_error("no ct orders with a nav place or nav area");
            }

            map<OrderId, size_t> tPlayersPerOrder;
            for (const auto & orderId : blackboard.strategy.getOrderIds(true, false)) {
                const Order & order = blackboard.strategy.getOrder(orderId);
                tPlayersPerOrder[orderId] = 0;
                for (const auto & waypoint : order.waypoints) {
                    if (waypoint.type == WaypointType::HoldPlace) {
                        tOptions.push_back({orderId, waypoint.placeName});
                    }
                    else if (waypoint.type == WaypointType::HoldAreas) {
                        for (const auto & areaId : waypoint.areaIds) {
                            string placeName = blackboard.navFile.get_place(
                                    blackboard.navFile.get_area_by_id_fast(areaId).m_place);
                            if (!placeName.empty()) {
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

            bool plantedA = blackboard.navFile.get_place(
                blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos())).m_place) == "BombsiteA";

            /*
            for (const auto & tOrderId  : blackboard.strategy.getOrderIds(true, false)) {
                std::cout << "t order id " << tOrderId.team << ", " << tOrderId.index << std::endl;
            }
            for (const auto & ctOrderId  : blackboard.strategy.getOrderIds(false, true)) {
                std::cout << "ct order id " << ctOrderId.team << ", " << ctOrderId.index << std::endl;
            }
             */

            // for CT, compute distance to each waypoint in each order and assign player to order with closest waypoint
            // for T, same thing but also considering covering every order - nearest unassigned order, or nearest unassigned if all assigned
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    if (!blackboard.inTest && !blackboard.inAnalysis && useOrderModelProbabilities &&
                        assignPlayerToOrderProbabilistic(client, plantedA)) {
                        continue;
                    }
                    if (client.team == ENGINE_TEAM_CT) {
                        Path pathToC4 = movement::computePath(blackboard, vec3Conv(state.getC4Pos()), client);
                        bool foundPath = false;
                        for (PathNode pathNode : pathToC4.waypoints) {
                            string placeName = blackboard.navFile.get_place(
                                    blackboard.navFile.get_area_by_id_fast(pathNode.area1).m_place);
                            if (ctOptions.find(placeName) != ctOptions.end()) {
                                blackboard.strategy.assignPlayerToOrder(client.csgoId, ctOptions[placeName]);
                                foundPath = true;
                                break;
                            }
                        }
                        // in brief situations (like a restart), can have mismatch between c4 position and orders
                        // so just pick any order in this case
                        if (!foundPath) {
                            blackboard.strategy.assignPlayerToOrder(client.csgoId, defaultCTOption);
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
                        /*
                        if (tOptions[0].assignedPlayer) {
                            int x =1;
                        }
                         */
                        tPlayersPerOrder[tOptions[0].orderId]++;
                        for (size_t i = 0; i < tOptions.size(); i++) {
                            tOptions[i].assignedPlayer = tPlayersPerOrder[tOptions[i].orderId] > 0;
                        }
                    }
                }
            }
            /*
            for (const auto & client : state.clients) {
                if (client.isAlive && client.isBot) {
                    blackboard.strategy.getOrderIdForPlayer(client.csgoId);
                }
            }
             */


        }
        else {
            blackboard.newOrderThisFrame = false;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    bool AssignPlayersToOrders::assignPlayerToOrderProbabilistic(const ServerState::Client & client, bool plantedA) {
        if (blackboard.inferenceManager.playerToInferenceData.find(client.csgoId) ==
            blackboard.inferenceManager.playerToInferenceData.end()) {
            return false;
        }
        blackboard.ticksSinceLastProbOrderAssignment = 0;
        vector<float> probabilities;
        const csknow::inference_latent_order::InferenceOrderPlayerAtTickProbabilities & orderProbabilities =
            blackboard.inferenceManager.playerToInferenceData.at(client.csgoId).orderProbabilities;
        if (plantedA) {
            probabilities = {
                orderProbabilities.orderProbabilities[aHeuristicToModelOrderIndices.at(0)],
                orderProbabilities.orderProbabilities[aHeuristicToModelOrderIndices.at(1)],
                orderProbabilities.orderProbabilities[aHeuristicToModelOrderIndices.at(2)]
            };
        }
        else {
            probabilities = {
                orderProbabilities.orderProbabilities[bHeuristicToModelOrderIndices.at(0)],
                orderProbabilities.orderProbabilities[bHeuristicToModelOrderIndices.at(1)],
                orderProbabilities.orderProbabilities[bHeuristicToModelOrderIndices.at(2)]
            };
        }

        // re-weight just for one site
        double reweightFactor = 0.;
        for (size_t i = 0; i < probabilities.size(); i++) {
            reweightFactor += probabilities[i];
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= 1/reweightFactor;
        }
        double probSample = blackboard.aggressionDis(blackboard.gen);
        double weightSoFar = 0.;
        for (size_t i = 0; i < probabilities.size(); i++) {
            weightSoFar += probabilities[i];
            if (probSample < weightSoFar) {
                //std::cout << "assigning to " << client.team << ", " << i << std::endl;
                blackboard.strategy.assignPlayerToOrder(client.csgoId,
                                                        {client.team, static_cast<int64_t>(i)});
                return true;
            }
        }
        // default if probs don't sum perfectly is take last one as this will result from a
        // slight numerical instability mismatch
        //std::cout << "bad assigning to " << client.team << ", " << 2 << std::endl;
        blackboard.strategy.assignPlayerToOrder(client.csgoId, {client.team, 2});
        return true;
    }
}
