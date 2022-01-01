//
// Created by durst on 12/30/21.
//

#include "queries/player_in_cover_edge.h"
#include "geometry.h"
#include "file_helpers.h"
#include <omp.h>
#include <set>
#include <map>
#include "cmath"
#include <atomic>

PlayerInCoverEdgeResult queryPlayerInCoverEdge(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                               const CoverOrigins & coverOrigins, const CoverEdges & coverEdges,
                                               const NearestOriginResult & nearestOriginResult) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpLookerPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookerPlayerIds[numThreads];
    vector<int64_t> tmpLookedAtPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookedAtPlayerIds[numThreads];
    vector<int64_t> tmpNearestOriginIds[numThreads];
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;

#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpTickId[threadNum].size());

        for (int64_t nearestOriginIndex = nearestOriginResult.nearestOriginPerRound[roundIndex].minId;
             nearestOriginIndex <= nearestOriginResult.nearestOriginPerRound[roundIndex].maxId; nearestOriginIndex++) {

            int64_t originIndex = nearestOriginResult.originId[nearestOriginIndex];
            int64_t tickIndex = nearestOriginResult.tickId[nearestOriginIndex];
            for (int64_t lookedAtPatIndex = ticks.patPerTick[tickIndex].minId;
                 lookedAtPatIndex != -1 && lookedAtPatIndex <= ticks.patPerTick[tickIndex].maxId; lookedAtPatIndex++) {
                if (!playerAtTick.isAlive[lookedAtPatIndex]) {
                    continue;
                }
                AABB playerAABB = getAABBForPlayer({playerAtTick.posX[lookedAtPatIndex],
                                                    playerAtTick.posY[lookedAtPatIndex],
                                                    playerAtTick.posZ[lookedAtPatIndex]});
                if (!aabbOverlap(coverOrigins.coverEdgeBoundsPerOrigin[originIndex], playerAABB)) {
                    continue;
                }
                for (int64_t coverEdgeIndex = coverOrigins.coverEdgesPerOrigin[originIndex].minId;
                     coverEdgeIndex != -1 && coverEdgeIndex <= coverOrigins.coverEdgesPerOrigin[originIndex].maxId;
                     coverEdgeIndex++) {
                    if (aabbOverlap(coverEdges.aabbs[coverEdgeIndex], playerAABB)) {
                        tmpTickId[threadNum].push_back(tickIndex);
                        tmpLookerPlayerAtTickIds[threadNum].push_back(nearestOriginResult.playerAtTickId[originIndex]);
                        tmpLookerPlayerIds[threadNum].push_back(nearestOriginResult.playerId[originIndex]);
                        tmpLookedAtPlayerAtTickIds[threadNum].push_back(lookedAtPatIndex);
                        tmpLookedAtPlayerIds[threadNum].push_back(playerAtTick.playerId[lookedAtPatIndex]);
                        tmpNearestOriginIds[threadNum].push_back(nearestOriginIndex);
                    }

                }
            }

        }
        tmpRoundSizes[threadNum].push_back(tmpTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
        roundsProcessed++;
        printProgress((roundsProcessed * 1.0) / rounds.size);
    }

    PlayerInCoverEdgeResult result;
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
        result.playerInCoverEdgePerRound.push_back({});
        result.playerInCoverEdgePerRound[minRoundId].minId = result.tickId.size();
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
            result.lookerPlayerAtTickId.push_back(tmpLookerPlayerAtTickIds[minThreadId][tmpRowId]);
            result.lookerPlayerId.push_back(tmpLookerPlayerIds[minThreadId][tmpRowId]);
            result.lookedAtPlayerAtTickId.push_back(tmpLookedAtPlayerAtTickIds[minThreadId][tmpRowId]);
            result.lookedAtPlayerId.push_back(tmpLookedAtPlayerIds[minThreadId][tmpRowId]);
            result.nearestOriginId.push_back(tmpNearestOriginIds[minThreadId][tmpRowId]);
        }
        result.playerInCoverEdgePerRound[minRoundId].maxId = result.tickId.size();
        roundsProcessedPerThread[minThreadId]++;
    }
    result.size = result.tickId.size();
    return result;
}
