//
// Created by durst on 12/30/21.
//
#include "queries/nearest_origin.h"
#include "geometry.h"
#include "file_helpers.h"
#include <omp.h>
#include "cmath"

[[maybe_unused]]
NearestOriginResult queryNearestOrigin(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                       const CoverOrigins & coverOrigins) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpPlayerAtTickIds[numThreads];
    vector<int64_t> tmpPlayerIds[numThreads];
    vector<int64_t> tmpOriginIds[numThreads];
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;

#pragma omp parallel
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                if (!playerAtTick.isAlive[patIndex]) {
                    continue;
                }
                tmpTickId[threadNum].push_back(tickIndex);
                tmpPlayerAtTickIds[threadNum].push_back(patIndex);
                tmpPlayerIds[threadNum].push_back(playerAtTick.playerId[patIndex]);
                Vec3 nearestOrigin;
                int64_t nearestIndex = coverOrigins.originsGrid.getNearest(coverOrigins.origins,
                                                                        {playerAtTick.posX[patIndex],
                                                                         playerAtTick.posY[patIndex],
                                                                         playerAtTick.posZ[patIndex] + EYE_HEIGHT},
                                                                         nearestOrigin);
                tmpOriginIds[threadNum].push_back(nearestIndex);
            }

        }
        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        roundsProcessed++;
        printProgress(roundsProcessed, rounds.size);
    }

    NearestOriginResult result;
    vector<int64_t> roundsProcessedPerThread(numThreads, 0);
    while (true) {
        bool roundToProcess = false;
        int64_t minThreadId = -1;
        int64_t minRoundId = -1;
        for (int64_t threadId = 0; threadId < numThreads; threadId++) {
            if (roundsProcessedPerThread[threadId] < static_cast<int64_t>(tmpRoundIds[threadId].size())) {
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
        result.nearestOriginPerRound.push_back({});
        result.nearestOriginPerRound[minRoundId].minId = static_cast<int64_t>(result.tickId.size());
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int64_t tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
            result.playerAtTickId.push_back(tmpPlayerAtTickIds[minThreadId][tmpRowId]);
            result.playerId.push_back(tmpPlayerIds[minThreadId][tmpRowId]);
            result.originId.push_back(tmpOriginIds[minThreadId][tmpRowId]);
        }
        result.nearestOriginPerRound[minRoundId].maxId = static_cast<int64_t>(result.tickId.size());
        roundsProcessedPerThread[minThreadId]++;
    }
    result.size = static_cast<int64_t>(result.tickId.size());
    return result;
}
