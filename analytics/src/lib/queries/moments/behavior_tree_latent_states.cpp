//
// Created by durst on 2/21/23.
//

#include "queries/moments/behavior_tree_latent_states.h"
#include "indices/build_indexes.h"
#include "bots/behavior_tree/tree.h"
#include "file_helpers.h"

namespace csknow::behavior_tree_latent_states {

    NodeState ClearPriorityNode::exec(const ServerState &, TreeThinker &treeThinker) {
        // default values are set to invalid where necessary, so this is fine
        Priority &curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.nonDangerAimArea = {};
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    struct TemporaryStateData {
        int64_t startTickId = INVALID_ID;
        StatePayload payload;
    };

    void finishEvent(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                     vector<vector<int64_t>> & tmpLength, vector<vector<LatentStateType>> & tmpLatentStateType,
                     vector<vector<StatePayload>> & tmpStatePayload,
                     const LatentStateType & latentStateType,
                     const EngagementStatePayload & engagementStatePayload,
                     int threadNum, int64_t curTickIndex, const TemporaryStateData & temporaryStateData) {
        tmpStartTickId[threadNum].push_back(temporaryStateData.startTickId);
        // subtract 1 as ended 1 tick before start of next event (when this is written)
        tmpEndTickId[threadNum].push_back(curTickIndex - 1);
        tmpLength[threadNum].push_back(curTickIndex - 1 - temporaryStateData.startTickId + 1);
        tmpLatentStateType[threadNum].push_back(latentStateType);
        tmpStatePayload[threadNum].push_back(engagementStatePayload);
    }

    void BehaviorTreeLatentStates::runQuery(const string & navPath, VisPoints visPoints,
                                            const MapMeshResult & mapMeshResult, const ReachableResult & reachability,
                                            const DistanceToPlacesResult & distanceToPlaces,
                                            const nearest_nav_cell::NearestNavCell & nearestNavCell,
                                            const csknow::orders::OrdersResult & ordersResult,
                                            const Players & players, const Games & games, const Rounds & rounds,
                                            const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                            const WeaponFire & weaponFire, const Hurt & hurt,
                                            const Plants & plants, const Defusals & defusals,
                                            const EngagementResult & acausalEngagementResult) {

        int numThreads = omp_get_max_threads();
        std::atomic<int64_t> roundsProcessed = 0;
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpStartTickId(numThreads);
        vector<vector<int64_t>> tmpEndTickId(numThreads);
        vector<vector<int64_t>> tmpLength(numThreads);
        vector<vector<LatentStateType>> tmpLatentStateType(numThreads);
        vector<vector<StatePayload>> tmpStatePayload(numThreads);
        vector<feature_store::FeatureStorePreCommitBuffer> tmpPreCommitBuffer(numThreads);
        TreeThinker defaultThinker{INVALID_ID, AggressiveType::Push};

#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));
            Blackboard blackboard(navPath, visPoints, nearestNavCell, mapMeshResult, reachability, distanceToPlaces,
                                  ordersResult, tmpPreCommitBuffer[threadNum]);
            GlobalQueryNode globalQueryNode(blackboard);
            PlayerQueryNode playerQueryNode(blackboard);
            RoundPlantDefusal roundPlantDefusal = processRoundPlantDefusals(rounds, ticks, plants, defusals, roundIndex);

            TickRates tickRates = computeTickRates(games, rounds, roundIndex);

            TemporaryStateData activeOrderState;
            map<CSGOId, TemporaryStateData> activeEngagementState;

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                blackboard.streamingManager.update(games, roundPlantDefusal, rounds, players, ticks, weaponFire, hurt,
                                                   playerAtTick, tickIndex, nearestNavCell, visPoints, tickRates);
                const ServerState & curState = blackboard.streamingManager.db.batchData.fromNewest();
                addTreeThinkersToBlackboard(curState, &blackboard);
                tmpPreCommitBuffer[threadNum].updateFeatureStoreBufferPlayers(curState);
                globalQueryNode.exec(curState, defaultThinker);

                featureStoreResult.teamFeatureStoreResult.commitTeamRow(tmpPreCommitBuffer[threadNum],
                                                                        blackboard.distanceToPlaces, blackboard.navFile,
                                                                        roundIndex, tickIndex);

                /*
                if (tickIndex == 8080) {
                    for (int64_t i = 0; i < 3; i++) {
                        const ServerState::Client & curClient = curState.getClient(3);
                        const ServerState::Client & victimClient = curState.getClient(i);
                        csknow::feature_store::EngagementPossibleEnemy possibleEnemy;
                        Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(victimClient.getEyePosForPlayer(),
                                                                               victimClient.getCurrentViewAngles(),
                                                                               victimClient.duckAmount);
                        Vec2 deltaViewAngle = deltaViewFromOriginToDest(curClient.getEyePosForPlayer(), victimHeadPos,
                                                                        curClient.getCurrentViewAngles());
                        std::cout << "tick index " << tickIndex << " cur pos " << curClient.getEyePosForPlayer().toCSV() <<
                            " enemy pos " << victimClient.getEyePosForPlayer().toCSV() << " cur client view angle "
                            << curClient.getCurrentViewAngles().toCSV() << std::endl;
                        std::cout << " delta view angle " << computeMagnitude(deltaViewAngle)
                                  << " world distance " << computeDistance(curClient.getEyePosForPlayer(), victimClient.getEyePosForPlayer()) << std::endl;
                    }
                }
                 */

                std::set<int64_t> firingPlayers;
                for (const auto & [_0, _1, fireIndex] :
                    ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    firingPlayers.insert(weaponFire.shooter[fireIndex]);
                }

                map<int64_t, int64_t> playerToACausalTarget;
                for (const auto & [_0, _1, engagementIndex] :
                    acausalEngagementResult.engagementsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    playerToACausalTarget[acausalEngagementResult.playerId[engagementIndex][0]] =
                        acausalEngagementResult.playerId[engagementIndex][1];
                }

                // order state transition whenever new orders
                // since my bots don't need to handle plants (plant start of round for retakes mode), I force a transition
                // in real data
                if (curState.c4IsPlanted) {
                    // this forces state transition on first plant frame (as INVALID_ID until plant)
                    if (blackboard.newOrderThisFrame || activeOrderState.startTickId == INVALID_ID) {
                        if (activeOrderState.startTickId != INVALID_ID) {
                            finishEvent(tmpStartTickId, tmpEndTickId, tmpLength,
                                        tmpLatentStateType, tmpStatePayload,
                                        LatentStateType::Order, {}, threadNum,
                                        tickIndex, activeOrderState);
                        }
                        activeOrderState.startTickId = tickIndex;
                        activeOrderState.payload = {};
                    }
                }

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                    patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    CSGOId curPlayerId = playerAtTick.playerId[patIndex];
                    if (playerAtTick.isAlive[patIndex]) {
                        playerQueryNode.exec(curState, blackboard.playerToTreeThinkers[curPlayerId]);

                        if (firingPlayers.find(curPlayerId) != firingPlayers.end()) {
                            featureStoreResult.fireCurTick[patIndex] = true;
                        }

                        bool prevActiveEngagement =
                            activeEngagementState.find(curPlayerId) != activeEngagementState.end();
                        int64_t curTarget = blackboard.playerToPriority[curPlayerId].targetPlayer.playerId;
                        CSGOId prevTarget = INVALID_ID;
                        if (prevActiveEngagement) {
                            prevTarget =
                                std::get<EngagementStatePayload>(activeEngagementState[curPlayerId].payload).targetId;
                        }

                        // state transition whenever a target change
                        if (prevTarget != curTarget) {
                            // if state change where previously had a target, write the state
                            if (prevActiveEngagement) {
                                finishEvent(tmpStartTickId, tmpEndTickId, tmpLength,
                                            tmpLatentStateType, tmpStatePayload,
                                            LatentStateType::Engagement,
                                            EngagementStatePayload{curPlayerId, prevTarget}, threadNum,
                                            tickIndex, activeEngagementState.at(curPlayerId));
                                activeEngagementState.erase(curPlayerId);
                            }
                            if (curTarget != INVALID_ID) {
                                activeEngagementState[curPlayerId] = {
                                    tickIndex,
                                    EngagementStatePayload{curPlayerId, curTarget}
                                };
                            }
                        }

                        bool visibleEngagement = curTarget != INVALID_ID;
                        bool hitEngagement = playerToACausalTarget.find(curPlayerId) != playerToACausalTarget.end();
                        tmpPreCommitBuffer[threadNum].addEngagementLabel(hitEngagement, visibleEngagement);
                        if (visibleEngagement && hitEngagement) {
                            int64_t acausalTarget = playerToACausalTarget[curPlayerId];
                            if (curTarget == acausalTarget) {
                                tmpPreCommitBuffer[threadNum].addTargetPossibleEnemyLabel({curTarget, true, true});
                            }
                            else {
                                tmpPreCommitBuffer[threadNum].addTargetPossibleEnemyLabel({curTarget, true, false});
                                tmpPreCommitBuffer[threadNum].addTargetPossibleEnemyLabel({acausalTarget, false, true});
                            }
                        }
                        else if (visibleEngagement) {
                            tmpPreCommitBuffer[threadNum].addTargetPossibleEnemyLabel({curTarget, true, false});
                        }
                        else if (hitEngagement) {
                            tmpPreCommitBuffer[threadNum].addTargetPossibleEnemyLabel({
                                playerToACausalTarget[curPlayerId], false, true
                            });
                        }
                    }
                    else {
                        bool prevActiveEngagement =
                            activeEngagementState.find(curPlayerId) != activeEngagementState.end();
                        if (prevActiveEngagement) {
                            CSGOId prevTarget =
                                std::get<EngagementStatePayload>(activeEngagementState[curPlayerId].payload).targetId;
                            finishEvent(tmpStartTickId, tmpEndTickId, tmpLength,
                                        tmpLatentStateType, tmpStatePayload,
                                        LatentStateType::Engagement,
                                        EngagementStatePayload{curPlayerId, prevTarget}, threadNum,
                                        tickIndex, activeEngagementState.at(curPlayerId));
                            activeEngagementState.erase(curPlayerId);
                        }
                    }
                    featureStoreResult.commitPlayerRow(tmpPreCommitBuffer[threadNum], patIndex,
                                                       roundIndex, tickIndex, curPlayerId);
                }
            }

            // finish all active events
            if (activeOrderState.startTickId != INVALID_ID) {
                finishEvent(tmpStartTickId, tmpEndTickId, tmpLength,
                            tmpLatentStateType, tmpStatePayload,
                            LatentStateType::Order, {}, threadNum,
                            rounds.ticksPerRound[roundIndex].maxId, activeOrderState);
            }
            for (const auto & [playerId, temporaryStateData] : activeEngagementState) {
                finishEvent(tmpStartTickId, tmpEndTickId, tmpLength,
                            tmpLatentStateType, tmpStatePayload,
                            LatentStateType::Engagement,
                            EngagementStatePayload{playerId,
                            std::get<EngagementStatePayload>(temporaryStateData.payload).targetId}, threadNum,
                            rounds.ticksPerRound[roundIndex].maxId,
                            activeEngagementState.at(playerId));

            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }

        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           startTickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                               endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                               tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                               latentStateType.push_back(tmpLatentStateType[minThreadId][tmpRowId]);
                               statePayload.push_back(tmpStatePayload[minThreadId][tmpRowId]);
                           });
        vector<const int64_t *> foreignKeyCols{startTickId.data(), endTickId.data()};
        eventsPerTick = buildIntervalIndex(foreignKeyCols, size);
    }

}
