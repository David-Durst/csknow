//
// Created by durst on 8/20/23.
//

#include "bots/testing/scripts/trace/replay_node.h"

namespace csknow::tests::trace {
    NodeState ReplayNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        int64_t ctBotIndex = 0, tBotIndex = 0;
        int64_t tickInFeatureStore = tracesData.startIndices[roundIndex] + tickInRound;
        if (tickInRound == 0) {
            roundStartTime = state.loadTime;
        }

        // stop when one team not alive
        bool ctAlive = false, tAlive = false;
        for (size_t i = 0; i < state.clients.size(); i++) {
            const ServerState::Client & client = state.clients[i];
            if (client.isAlive) {
                if (client.team == ENGINE_TEAM_CT) {
                    ctAlive = true;
                }
                else if (client.team == ENGINE_TEAM_T) {
                    tAlive = true;
                }
            }
        }
        if (!ctAlive || !tAlive || state.getSecondsBetweenTimes(roundStartTime, state.loadTime) > max_time_per_replay) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return playerNodeState[treeThinker.csgoId];
        }

        for (size_t i = 0; i < state.clients.size(); i++) {
            const ServerState::Client & client = state.clients[i];
            // assuming that script already configurated players to be alive
            if (client.isAlive) {
                int64_t columnIndex = client.team == ENGINE_TEAM_CT ?
                        tracesData.ctBotIndexToFeatureStoreIndex[roundIndex][ctBotIndex] :
                        tracesData.tBotIndexToFeatureStoreIndex[roundIndex][tBotIndex];
                const array<feature_store::TeamFeatureStoreResult::ColumnPlayerData, feature_store::max_enemies> &
                        columnData = client.team == ENGINE_TEAM_CT ?
                                     tracesData.teamFeatureStoreResult.columnCTData :
                                     tracesData.teamFeatureStoreResult.columnTData;

                // if alive in game but not trace, just let bot controller handle it
                if (columnData[columnIndex].alive[roundIndex]) {
                    blackboard.playerToAction[client.csgoId].enableAbsPos = true;
                    blackboard.playerToAction[client.csgoId].absPos =
                            columnData[columnIndex].footPos[tickInFeatureStore];
                    blackboard.playerToAction[client.csgoId].absView =
                            columnData[columnIndex].viewAngle[tickInFeatureStore];
                }

            }

            // track indices even when dead to keep consistent (don't want to teleport to replace a player whne they die)
            if (client.team == ENGINE_TEAM_CT) {
                ctBotIndex++;
            }
            else if (client.team == ENGINE_TEAM_T) {
                tBotIndex++;
            }
        }

        tickInRound++;
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }
}