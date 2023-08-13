//
// Created by durst on 7/24/22.
//

#include "bots/behavior_tree/global/strategy_node.h"
#include "bots/analysis/learned_models.h"
#include "bots/behavior_tree/order_model_heuristic_mapping.h"

namespace strategy {
    NodeState CreateOrdersNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        // 2348721_120274_apeks-vs-sangal-m1-dust2_c3fd422a-b73c-11eb-a514-0a58a9feac02.dem has capitalized map name
        if (state.mapName.find("de_dust2") == std::string::npos && state.mapName.find("DE_DUST2") == std::string::npos) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return NodeState::Failure;
        }

        bool botNeedsAnOrder = false;
        for (const auto & client : state.clients) {
            if (client.isAlive && client.isBot && !blackboard.strategy.haveOrderIdForPlayer(client.csgoId)) {
                botNeedsAnOrder = true;
                break;
            }
        }

        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            state.roundNumber != planRoundNumber || state.numPlayersAlive() != playersAliveLastPlan ||
            state.getPlayersOnTeam(ENGINE_TEAM_CT) != ctPlayers || state.getPlayersOnTeam(ENGINE_TEAM_T) != tPlayers ||
            botNeedsAnOrder || blackboard.recomputeOrders) {
            planRoundNumber = state.roundNumber;
            playersAliveLastPlan = state.numPlayersAlive();
            ctPlayers = state.getPlayersOnTeam(ENGINE_TEAM_CT);
            tPlayers = state.getPlayersOnTeam(ENGINE_TEAM_T);
            blackboard.newOrderThisFrame = true;
            blackboard.recomputeOrders = false;

            blackboard.strategy.clear();
            blackboard.playerToPath.clear();
            blackboard.playerToPriority.clear();
            blackboard.playerToModelNavData.clear();

            // first setup orders to go A or B
            bool plantedA = blackboard.navFile.get_place(
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos())).m_place) == "BombsiteA";

            if (plantedA) {
                blackboard.placesVisibleFromDestination = blackboard.ordersResult.aPlacesVisibleFromDestination;
            }
            else {
                blackboard.placesVisibleFromDestination = blackboard.ordersResult.bPlacesVisibleFromDestination;
            }

            for (const auto & order : plantedA ? aOffenseOrders : bOffenseOrders) {
                blackboard.strategy.addOrder(ENGINE_TEAM_CT, order, blackboard.navFile, blackboard.reachability,
                                             blackboard.visPoints, blackboard.distanceToPlaces);
            }
            for (const auto & order : plantedA ? aDefenseOrders : bDefenseOrders) {
                blackboard.strategy.addOrder(ENGINE_TEAM_T, order, blackboard.navFile, blackboard.reachability,
                                             blackboard.visPoints, blackboard.distanceToPlaces);
            }
        }
        else {
            blackboard.newOrderThisFrame = false;
        }

        // save player team state in feature store
        blackboard.featureStorePreCommitBuffer.updateCurTeamData(state, blackboard.navFile);

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    void CreateOrdersNode::createModelOrders() {
        if (!blackboard.strategy.aModelCTOrders.empty()) {
            return;
        }
        vector<Order> ctOrders, tOrders;
        // create orders in bot format from query format
        for (const auto & queryOrder : blackboard.ordersResult.orders) {
            Waypoints ctWaypoints;
            for (const auto & queryOrderPlace : queryOrder.places) {
                ctWaypoints.push_back({WaypointType::NavPlace,
                                       blackboard.distanceToPlaces.places[queryOrderPlace]});
            }
            // add c4 for CT
            Waypoint lastCTWaypoint = ctWaypoints.back();
            lastCTWaypoint.type = WaypointType::C4;
            ctWaypoints.push_back(lastCTWaypoint);
            Waypoints tWaypoints = ctWaypoints;
            std::reverse(tWaypoints.begin(), tWaypoints.end());

            ctOrders.push_back({ctWaypoints});
            tOrders.push_back({tWaypoints});
        }

        // insert in order matching hueristics
        for (size_t heuristicIndex = 0; heuristicIndex < csknow::feature_store::num_orders_per_site; heuristicIndex++) {
            blackboard.strategy.aModelCTOrders.push_back(ctOrders[aHeuristicToModelOrderIndices.at(heuristicIndex)]);
            blackboard.strategy.aModelTOrders.push_back(tOrders[aHeuristicToModelOrderIndices.at(heuristicIndex)]);
            blackboard.strategy.bModelCTOrders.push_back(ctOrders[bHeuristicToModelOrderIndices.at(heuristicIndex)]);
            blackboard.strategy.bModelTOrders.push_back(tOrders[bHeuristicToModelOrderIndices.at(heuristicIndex)]);
        }
    }
}
