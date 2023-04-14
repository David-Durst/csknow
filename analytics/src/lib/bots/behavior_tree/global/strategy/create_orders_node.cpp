//
// Created by durst on 7/24/22.
//

#include "bots/behavior_tree/global/strategy_node.h"

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
        bool probOrderChange =
            !blackboard.inTest && blackboard.ticksSinceLastProbOrderAssignment >= newOrderTicks &&
            ctPlayersAlive && tPlayersAlive;
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            state.roundNumber != planRoundNumber || state.numPlayersAlive() != playersAliveLastPlan ||
            state.getPlayersOnTeam(ENGINE_TEAM_CT) != ctPlayers || state.getPlayersOnTeam(ENGINE_TEAM_T) != tPlayers ||
            botNeedsAnOrder ||
            blackboard.recomputeOrders || probOrderChange) {
            planRoundNumber = state.roundNumber;
            playersAliveLastPlan = state.numPlayersAlive();
            ctPlayers = state.getPlayersOnTeam(ENGINE_TEAM_CT);
            tPlayers = state.getPlayersOnTeam(ENGINE_TEAM_T);
            blackboard.newOrderThisFrame = true;
            blackboard.recomputeOrders = false;

            blackboard.strategy.clear();
            blackboard.playerToPath.clear();
            blackboard.playerToPriority.clear();

            // first setup orders to go A or B
            bool plantedA = blackboard.navFile.get_place(
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos())).m_place) == "BombsiteA";

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
                                        client.getFootPosForPlayer()});
        }
        AreaId c4AreaId = blackboard.navFile
            .get_nearest_area_by_position(vec3Conv(state.getC4Pos()))
            .get_id();
        int64_t c4AreaIndex = blackboard.navFile.m_area_ids_to_indices.at(c4AreaId);
        blackboard.featureStorePreCommitBuffer.c4MapData = {
            state.c4IsPlanted,
            c4AreaId,
            c4AreaIndex
        };

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
