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


        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            state.roundNumber != planRoundNumber || state.numPlayersAlive() != playersAliveLastPlan) {
            planRoundNumber = state.roundNumber;
            playersAliveLastPlan = state.numPlayersAlive();
            blackboard.newOrderThisFrame = true;

            blackboard.strategy.clear();
            blackboard.playerToPath.clear();
            blackboard.playerToPriority.clear();

            // first setup orders to go A or B
            bool plantedA = blackboard.navFile.get_place(
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos())).m_place) == "BombsiteA";

            for (const auto & order : plantedA ? aOffenseOrders : bOffenseOrders) {
                blackboard.strategy.addOrder(ENGINE_TEAM_CT, order);
            }
            for (const auto & order : plantedA ? aDefenseOrders : bDefenseOrders) {
                blackboard.strategy.addOrder(ENGINE_TEAM_T, order);
            }
        }
        else {
            blackboard.newOrderThisFrame = false;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
