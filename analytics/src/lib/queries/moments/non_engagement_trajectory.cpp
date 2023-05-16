//
// Created by durst on 9/15/22.
//
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include "indices/build_indexes.h"
#include <omp.h>
#include <atomic>

struct NETData {
    int64_t startTickId;
};

void finishEngagement(vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                      vector<vector<int64_t>> & tmpLength, vector<vector<int64_t>> & tmpPlayerId,
                      int threadNum, int64_t tickIndex, int64_t playerId,
                      map<int64_t, NETData> & playerToCurTrajectory, bool remove = true) {
    tmpStartTickId[threadNum].push_back(playerToCurTrajectory[playerId].startTickId);
    tmpEndTickId[threadNum].push_back(tickIndex);
    tmpLength[threadNum].push_back(tmpEndTickId[threadNum].back() - tmpStartTickId[threadNum].back() + 1);
    tmpPlayerId[threadNum].push_back(playerId);
    if (remove) {
        playerToCurTrajectory.erase(playerId);
    }
}

NonEngagementTrajectoryResult queryNonEngagementTrajectory(const Rounds & rounds, const Ticks & ticks,
                                                           const PlayerAtTick & playerAtTick,
                                                           const EngagementResult & engagementResult) {
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpStartTickId(numThreads);
    vector<vector<int64_t>> tmpEndTickId(numThreads);
    vector<vector<int64_t>> tmpLength(numThreads);
    vector<vector<int64_t>> tmpPlayerId(numThreads);

    // for each round
    // for each tick
    // record all players in engagement
    // if a player is in engagement, then terminate their trajectory (if one was in progress) without recording
    // otherwise, terminate trajectory on end of round
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

        //TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<int64_t, NETData> playerToCurTrajectory;
        RollingWindow rollingWindow(rounds, ticks, playerAtTick);

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            // round 17 (after halftime), k0nfig disappears for round start time. Remove a player without writing
            // trajectory if this happens
            map<int64_t, int64_t> curPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex);
            vector<int64_t> disappearingPlayers;
            for (const auto & [playerId, _] : playerToCurTrajectory) {
                if (curPlayerToPAT.find(playerId) == curPlayerToPAT.end()) {
                    disappearingPlayers.push_back(playerId);
                }
            }
            for (const auto disappearingPlayer : disappearingPlayers) {
                playerToCurTrajectory.erase(disappearingPlayer);
            }

            std::set<int64_t> inEngagement;
            for (const auto & [_0, _1, engagementIndex] :
                    engagementResult.engagementsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                for (const auto playerId : engagementResult.playerId[engagementIndex]) {
                    if (playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                        finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, tickIndex,
                                         playerId, playerToCurTrajectory);
                    }
                    inEngagement.insert(playerId);
                }
            }

            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                int64_t playerId = playerAtTick.playerId[patIndex];
                if (playerAtTick.isAlive[patIndex] &&
                    playerToCurTrajectory.find(playerId) == playerToCurTrajectory.end() &&
                    inEngagement.find(playerId) == inEngagement.end()) {
                    playerToCurTrajectory[playerId] = {tickIndex};
                }
                // if player somehow dies without being in engagement (like from nade), end trajectory on prior tick
                // when they were last alive
                if (!playerAtTick.isAlive[patIndex] &&
                    playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                    finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, tickIndex - 1,
                                     playerId, playerToCurTrajectory);
                }
            }
        }

        for (const auto [playerId, tData] : playerToCurTrajectory) {
            finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, rounds.ticksPerRound[roundIndex].maxId,
                             playerId, playerToCurTrajectory, false);

        }
        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        //roundsProcessed++;
        //printProgress((roundsProcessed * 1.0) / rounds.size);
    }

    NonEngagementTrajectoryResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.startTickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                           result.endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                       });
    vector<std::reference_wrapper<const vector<int64_t>>> foreignKeyCols{result.startTickId, result.endTickId};
    result.trajectoriesPerTick = buildIntervalIndex(foreignKeyCols, result.size);
    return result;
}
