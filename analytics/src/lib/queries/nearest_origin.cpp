//
// Created by durst on 12/30/21.
//
#include "queries/nearest_origin.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
#include "cmath"

NearestOriginResult queryNearestOrigin(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                       const CoverOrigins & coverOrigins) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpPlayerAtTickIds[numThreads];
    vector<int64_t> tmpPlayerIds[numThreads];
    vector<int64_t> tmpOriginIds[numThreads];

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();

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
                                                                         playerAtTick.posZ[patIndex]}, nearestOrigin);
                tmpOriginIds[threadNum].push_back(nearestIndex);
            }

        }
    }

    NearestOriginResult result;
    for (int i = 0; i < numThreads; i++) {
        result.nearestOriginPerRound.push_back({});
        result.nearestOriginPerRound[i].minId = result.tickId.size();
        for (int j = 0; j < tmpTickId[i].size(); j++) {
            result.tickId.push_back(tmpTickId[i][j]);
            result.playerAtTickId.push_back(tmpPlayerAtTickIds[i][j]);
            result.playerId.push_back(tmpPlayerIds[i][j]);
            result.originId.push_back(tmpOriginIds[i][j]);
        }
        result.nearestOriginPerRound[i].maxId = result.tickId.size();
    }
    result.size = result.tickId.size();
    return result;
}
