//
// Created by durst on 9/15/22.
//
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include "indices/build_indexes.h"
#include <omp.h>
#include <atomic>

struct SegmentData {
    int64_t startTickId;
};

void finishEngagement(vector<int64_t> tmpStartTickId[], vector<int64_t> tmpEndTickId[],
                      vector<int64_t> tmpLength[], vector<int64_t> tmpPlayerId[],
                      int threadNum, int64_t tickIndex, int64_t playerId, const SegmentData &tData,
                      map<int64_t, SegmentData> & playerToCurTrajectory, bool remove = true) {
    tmpStartTickId[threadNum].push_back(playerToCurTrajectory[playerId].startTickId);
    tmpEndTickId[threadNum].push_back(tickIndex);
    tmpLength[threadNum].push_back(tmpEndTickId[threadNum].back() - tmpStartTickId[threadNum].back() + 1);
    tmpPlayerId[threadNum].push_back(playerId);
    if (remove) {
        playerToCurTrajectory.erase(playerId);
    }
}

NonEngagementTrajectoryResult queryNonEngagementTrajectory(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                           const PlayerAtTick & playerAtTick,
                                                           const EngagementResult & engagementResult) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpStartTickId[numThreads];
    vector<int64_t> tmpEndTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<int64_t> tmpPlayerId[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;
    double maxSpeed = -1;

    // for each round
    // for each tick
    // record all players in engagement
    // if a player is in engagement, then terminate their trajectory (if one was in progress) without recording
    // otherwise, terminate trajectory on end of round
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpStartTickId[threadNum].size());

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<int64_t, SegmentData> playerToCurTrajectory;


        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            std::set<int64_t> inEngagement;
            for (const auto & [_0, _1, engagementIndex] :
                    engagementResult.engagementsPerTick.findOverlapping(tickIndex, tickIndex)) {
                for (const auto playerId : engagementResult.playerId[engagementIndex]) {
                    if (playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                        finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, tickIndex,
                                         playerId, playerToCurTrajectory.find(playerId)->second, playerToCurTrajectory);
                    }
                    inEngagement.insert(playerId);
                }
            }

            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                int64_t playerId = playerAtTick.playerId[patIndex];
                if (playerAtTick.isAlive[playerId] && playerToCurTrajectory.find(playerId) == playerToCurTrajectory.end() &&
                    inEngagement.find(playerId) == inEngagement.end()) {
                    playerToCurTrajectory[playerId] = {tickIndex};
                }
                // if player somehow dies without being in engagement (like from nade), end trajectory
                if (!playerAtTick.isAlive[playerId] && playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                    finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, tickIndex,
                                     playerId, playerToCurTrajectory.find(playerId)->second, playerToCurTrajectory);
                }
            }
        }

        for (const auto [playerId, tData] : playerToCurTrajectory) {
            finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, rounds.ticksPerRound[roundIndex].maxId,
                             playerId, playerToCurTrajectory.find(playerId)->second, playerToCurTrajectory, false);

        }
        tmpRoundSizes[threadNum].push_back(tmpStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
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
    vector<const int64_t *> foreignKeyCols{result.startTickId.data(), result.endTickId.data()};
    result.trajectoriesPerTick = buildIntervalIndex(foreignKeyCols, result.size);
    return result;
}
