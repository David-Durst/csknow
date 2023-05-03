//
// Created by durst on 7/24/22.
//

#include "bots/behavior_tree/global/strategy_node.h"
#include "bots/analysis/learned_models.h"
#include "bots/behavior_tree/order_model_heuristic_mapping.h"

namespace strategy {
    NodeState CreateOrdersNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (state.mapName != "de_dust2") {
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

        blackboard.ticksSinceLastProbOrderAssignment++;
        bool ctPlayersAlive = false, tPlayersAlive = false;
        for (const auto & client : state.clients) {
            if (client.isAlive && client.team == ENGINE_TEAM_CT) {
                ctPlayersAlive = true;
            }
            if (client.isAlive && client.team == ENGINE_TEAM_T) {
                tPlayersAlive = true;
            }
        }
        bool useOrderModelProbabilitiesEitherTeam = useOrderModelProbabilitiesT || useOrderModelProbabilitiesCT;
        bool probOrderChange = false && useOrderModelProbabilitiesEitherTeam &&
            !blackboard.inTest && !blackboard.inAnalysis &&
            blackboard.ticksSinceLastProbOrderAssignment >= newOrderTicks && ctPlayersAlive && tPlayersAlive &&
            blackboard.defuserId == INVALID_ID; // if have a defuser, don't interrup them
        // as soon as have valid inference data, need to get model orders, as rest of tree will assume switching to models
        bool switchToModelOrders = useOrderModelProbabilitiesEitherTeam && !blackboard.inTest && !blackboard.inAnalysis &&
            !blackboard.modelOrdersT && !blackboard.modelOrdersCT && blackboard.inferenceManager.haveValidData();
        // if start with model orders and switch off, then need to remove model orders immediately
        bool switchFromModelOrders = (blackboard.modelOrdersT || blackboard.modelOrdersCT) && (blackboard.inTest || blackboard.inAnalysis);
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            state.roundNumber != planRoundNumber || /*state.numPlayersAlive() != playersAliveLastPlan ||*/
            state.getPlayersOnTeam(ENGINE_TEAM_CT) != ctPlayers || state.getPlayersOnTeam(ENGINE_TEAM_T) != tPlayers ||
            botNeedsAnOrder ||
            blackboard.recomputeOrders || probOrderChange || switchToModelOrders || switchFromModelOrders) {
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

            bool eitherTeamModelRequirements = !blackboard.inTest && !blackboard.inAnalysis &&
                blackboard.inferenceManager.haveValidData();
            if (eitherTeamModelRequirements && getPlaceAreaModelProbabilities(ENGINE_TEAM_CT)) {
                blackboard.modelOrdersCT = true;
                createModelOrders();
                //std::cout << (plantedA ? "A " : "B ") << "CT orders: " << std::endl;
                for (const auto & order : plantedA ? blackboard.strategy.aModelCTOrders : blackboard.strategy.bModelCTOrders) {
                    blackboard.strategy.addOrder(ENGINE_TEAM_CT, order, blackboard.navFile, blackboard.reachability,
                                                 blackboard.visPoints, blackboard.distanceToPlaces);
                    /*
                    for (const auto & waypoint : order.waypoints) {
                        std::cout << waypoint.placeName << ", ";
                    }
                    std::cout << std::endl;
                     */
                }
            }
            else {
                blackboard.modelOrdersCT = false;
                for (const auto & order : plantedA ? aOffenseOrders : bOffenseOrders) {
                    blackboard.strategy.addOrder(ENGINE_TEAM_CT, order, blackboard.navFile, blackboard.reachability,
                                                 blackboard.visPoints, blackboard.distanceToPlaces);
                }
            }
            if (eitherTeamModelRequirements && getPlaceAreaModelProbabilities(ENGINE_TEAM_T)) {
                blackboard.modelOrdersT = true;
                createModelOrders();
                // std::cout << (plantedA ? "A " : "B ") << "T orders: ";
                for (const auto & order : plantedA ? blackboard.strategy.aModelTOrders : blackboard.strategy.bModelTOrders) {
                    blackboard.strategy.addOrder(ENGINE_TEAM_T, order, blackboard.navFile, blackboard.reachability,
                                                 blackboard.visPoints, blackboard.distanceToPlaces);
                    /*
                    for (const auto & waypoint : order.waypoints) {
                        std::cout << waypoint.placeName << ", ";
                    }
                    std::cout << std::endl;
                     */
                }
            }
            else {
                blackboard.modelOrdersT = false;
                for (const auto & order : plantedA ? aDefenseOrders : bDefenseOrders) {
                    blackboard.strategy.addOrder(ENGINE_TEAM_T, order, blackboard.navFile, blackboard.reachability,
                                                 blackboard.visPoints, blackboard.distanceToPlaces);
                }
            }
        }
        else {
            blackboard.newOrderThisFrame = false;
        }

        // save player team state in feature store
        vector<csknow::feature_store::BTTeamPlayerData> & btTeamPlayerData =
            blackboard.featureStorePreCommitBuffer.btTeamPlayerData;
        btTeamPlayerData.clear();
        for (const auto & client : state.clients) {
            if (!client.isAlive) {
                continue;
            }
            AreaId curAreaId = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer()))
                .get_id();
            int64_t curAreaIndex = blackboard.navFile.m_area_ids_to_indices.at(curAreaId);
            btTeamPlayerData.push_back({client.csgoId, client.team, curAreaId, curAreaIndex,
                                        client.getFootPosForPlayer(), client.getVelocity()});
        }
        blackboard.featureStorePreCommitBuffer.appendPlayerHistory();
        AreaId c4AreaId = blackboard.navFile
            .get_nearest_area_by_position(vec3Conv(state.getC4Pos()))
            .get_id();
        int64_t c4AreaIndex = blackboard.navFile.m_area_ids_to_indices.at(c4AreaId);
        blackboard.featureStorePreCommitBuffer.c4MapData = {
            state.getC4Pos(),
            state.c4IsPlanted,
            state.ticksSinceLastPlant,
            c4AreaId,
            c4AreaIndex
        };

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
