//
// Created by durst on 8/20/23.
//

#include "bots/testing/scripts/trace/replay_node.h"

namespace csknow::tests::trace {
    NodeState ReplayNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        int64_t ctBotIndex = 0, tBotIndex = 0;
        if (curRoundTick == 0) {
            roundStartTime = state.loadTime;
            startFrame = static_cast<int64_t>(state.getLastFrame());
        }
        curRoundTick++;
        int64_t curFrame = static_cast<int64_t>(state.getLastFrame());
        bool firstFrame = curFrame == startFrame;
        if (curFrame < startFrame) {
            // start frame restarts at 0, so add 1 for that offset and continue
            curFrame = startFrame + curFrame + 1;
        }
        int64_t framesAfterStartFrame = curFrame - startFrame;
        int64_t tickInFeatureStore = tracesData.startIndices[roundIndex] +
                (framesAfterStartFrame / feature_store::every_nth_row);

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

            // limit to right team/bot as requested by config
            // ignore on first tick so everyone in right place
            bool botTeamForTest = oneTeam && !firstFrame &&
                    ((client.team == ENGINE_TEAM_CT && tracesData.ctBot[roundIndex]) ||
                    (client.team == ENGINE_TEAM_T && !tracesData.ctBot[roundIndex]));
            bool botPlayerForTest = oneBot && botTeamForTest && !firstFrame &&
                    ((client.team == ENGINE_TEAM_CT && tracesData.ctBotIndexToFeatureStoreIndex[roundIndex][ctBotIndex]
                        == tracesData.oneBotFeatureStoreIndex[roundIndex]) ||
                    (client.team == ENGINE_TEAM_T && tracesData.tBotIndexToFeatureStoreIndex[roundIndex][tBotIndex]
                        == tracesData.oneBotFeatureStoreIndex[roundIndex]));

            // assuming that script already configured players to be alive
            if (client.isAlive && !botTeamForTest && !botPlayerForTest) {
                int64_t columnIndex = client.team == ENGINE_TEAM_CT ?
                        tracesData.ctBotIndexToFeatureStoreIndex[roundIndex][ctBotIndex] :
                        tracesData.tBotIndexToFeatureStoreIndex[roundIndex][tBotIndex];
                const array<feature_store::TeamFeatureStoreResult::ColumnPlayerData, feature_store::max_enemies> &
                        columnData = client.team == ENGINE_TEAM_CT ?
                                     tracesData.teamFeatureStoreResult.columnCTData :
                                     tracesData.teamFeatureStoreResult.columnTData;

                // if alive in game but not trace, just let bot controller handle it
                // stop 1 tick early as averaging between ticks
                if (columnData[columnIndex].alive[tickInFeatureStore + 1] &&
                    tickInFeatureStore < tracesData.startIndices[roundIndex] + tracesData.lengths[roundIndex] - 1) {
                    blackboard.playerToAction[client.csgoId].enableAbsPos = true;
                    double curTickWeight = 1. -
                            (static_cast<double>(framesAfterStartFrame % feature_store::every_nth_row) /
                             static_cast<double>(feature_store::every_nth_row));
                    blackboard.playerToAction[client.csgoId].absPos =
                            avg(columnData[columnIndex].footPos[tickInFeatureStore],
                                columnData[columnIndex].footPos[tickInFeatureStore + 1], curTickWeight);
                    blackboard.playerToAction[client.csgoId].absView =
                            avg(columnData[columnIndex].viewAngle[tickInFeatureStore],
                                columnData[columnIndex].viewAngle[tickInFeatureStore + 1], curTickWeight);
                    blackboard.playerToAction[client.csgoId]
                        .setButton(IN_WALK, columnData[columnIndex].walking[tickInFeatureStore]);
                    blackboard.playerToAction[client.csgoId]
                        .setButton(IN_DUCK, columnData[columnIndex].ducking[tickInFeatureStore]);
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

        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }
}