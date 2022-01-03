//
// Created by durst on 12/31/21.
//

#include "queries/team_looking_at_cover_edge_cluster.h"
#include "queries/lookback.h"
#include "geometry.h"
#include "file_helpers.h"
#include <omp.h>
#include <set>
#include <map>
#include "cmath"

TeamLookingAtCoverEdgeCluster
queryTeamLookingAtCoverEdgeCluster(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                   const PlayerAtTick & playerAtTick, const CoverOrigins & coverOrigins,
                                   const CoverEdges & coverEdges, const NearestOriginResult & nearestOriginResult) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpCurPlayerAtTickIds[numThreads];
    vector<int64_t> tmpCurPlayerIds[numThreads];
    vector<int64_t> tmpLookerPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookerPlayerIds[numThreads];
    vector<int64_t> tmpNearestOriginIds[numThreads];
    vector<int64_t> tmpCoverEdgeClusterIds[numThreads];
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;

#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpTickId[threadNum].size());
        TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        int64_t lookbackGameTicks = tickRates.gameTickRate;

        for (int64_t nearestOriginIndex = nearestOriginResult.nearestOriginPerRound[roundIndex].minId;
             nearestOriginIndex <= nearestOriginResult.nearestOriginPerRound[roundIndex].maxId; nearestOriginIndex++) {
            int64_t originIndex = nearestOriginResult.originId[nearestOriginIndex];
            int64_t tickIndex = nearestOriginResult.tickId[nearestOriginIndex];
            int64_t startLookedAtTickIndex =
                    tickIndex - getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, lookbackGameTicks);
            std::set<int64_t> coveredClusters;

            for (int64_t lookedAtPatIndex = ticks.patPerTick[startLookedAtTickIndex].minId;
                 lookedAtPatIndex != -1 && lookedAtPatIndex <= ticks.patPerTick[tickIndex].maxId; lookedAtPatIndex++) {
                if (!playerAtTick.isAlive[lookedAtPatIndex] ||
                    playerAtTick.team[lookedAtPatIndex] != playerAtTick.team[nearestOriginResult.playerAtTickId[originIndex]]) {
                    continue;
                }
                Ray playerEyeCoord = getEyeCoordinatesForPlayer(
                        {playerAtTick.posX[lookedAtPatIndex], playerAtTick.posY[lookedAtPatIndex],
                         playerAtTick.posZ[lookedAtPatIndex]},
                        {playerAtTick.viewX[lookedAtPatIndex], playerAtTick.viewY[lookedAtPatIndex]});

                double t0, t1;
                if (!intersectP(coverOrigins.coverEdgeBoundsPerOrigin[originIndex], playerEyeCoord, t0, t1)) {
                    continue;
                }
                for (int64_t coverEdgeIndex = coverOrigins.coverEdgesPerOrigin[originIndex].minId;
                     coverEdgeIndex != -1 && coverEdgeIndex <= coverOrigins.coverEdgesPerOrigin[originIndex].maxId;
                     coverEdgeIndex++) {
                    if (coveredClusters.count(coverEdges.clusterId[coverEdgeIndex]) == 1) {
                        continue;
                    }
                    if (intersectP(coverEdges.aabbs[coverEdgeIndex], playerEyeCoord, t0, t1)) {
                        tmpTickId[threadNum].push_back(tickIndex);
                        tmpCurPlayerAtTickIds[threadNum].push_back(lookedAtPatIndex);
                        tmpCurPlayerIds[threadNum].push_back(playerAtTick.playerId[lookedAtPatIndex]);
                        tmpLookerPlayerAtTickIds[threadNum].push_back(nearestOriginResult.playerAtTickId[originIndex]);
                        tmpLookerPlayerIds[threadNum].push_back(nearestOriginResult.playerId[originIndex]);
                        tmpNearestOriginIds[threadNum].push_back(nearestOriginIndex);
                        tmpCoverEdgeClusterIds[threadNum].push_back(coverEdges.clusterId[coverEdgeIndex]);
                        coveredClusters.insert(coverEdges.clusterId[coverEdgeIndex]);
                    }

                }
            }

        }
        tmpRoundSizes[threadNum].push_back(tmpTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
        roundsProcessed++;
        printProgress((roundsProcessed * 1.0) / rounds.size);
    }

    TeamLookingAtCoverEdgeCluster result;
    vector<int64_t> roundsProcessedPerThread(numThreads, 0);
    while (true) {
        bool roundToProcess = false;
        int64_t minThreadId = -1;
        int64_t minRoundId = -1;
        for (int64_t threadId = 0; threadId < numThreads; threadId++) {
            if (roundsProcessedPerThread[threadId] < tmpRoundIds[threadId].size()) {
                roundToProcess = true;
                if (minThreadId == -1 || tmpRoundIds[threadId][roundsProcessedPerThread[threadId]] < minRoundId) {
                    minThreadId = threadId;
                    minRoundId = tmpRoundIds[minThreadId][roundsProcessedPerThread[minThreadId]];
                }

            }
        }
        if (!roundToProcess) {
            break;
        }
        result.teamLookingAtCoverEdgeClusterPerRound.push_back({});
        result.teamLookingAtCoverEdgeClusterPerRound[minRoundId].minId = result.tickId.size();
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
            result.curPlayerAtTickId.push_back(tmpCurPlayerAtTickIds[minThreadId][tmpRowId]);
            result.curPlayerId.push_back(tmpCurPlayerIds[minThreadId][tmpRowId]);
            result.lookerPlayerAtTickId.push_back(tmpLookerPlayerAtTickIds[minThreadId][tmpRowId]);
            result.lookerPlayerId.push_back(tmpLookerPlayerIds[minThreadId][tmpRowId]);
            result.nearestOriginId.push_back(tmpNearestOriginIds[minThreadId][tmpRowId]);
            result.coverEdgeClusterId.push_back(tmpCoverEdgeClusterIds[minThreadId][tmpRowId]);
        }
        result.teamLookingAtCoverEdgeClusterPerRound[minRoundId].maxId = result.tickId.size();
        roundsProcessedPerThread[minThreadId]++;
    }
    result.size = result.tickId.size();
    return result;
}
