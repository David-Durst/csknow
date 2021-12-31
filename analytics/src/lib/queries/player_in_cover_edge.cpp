//
// Created by durst on 12/30/21.
//

#include "queries/player_in_cover_edge.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
#include "cmath"

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

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();

        for (int64_t nearestOriginIndex = nearestOriginResult.nearestOriginPerRound[roundIndex].minId;
             nearestOriginIndex <= nearestOriginResult.nearestOriginPerRound[roundIndex].maxId; nearestOriginIndex++) {

            int64_t originIndex = nearestOriginResult.originId[nearestOriginIndex];
            int64_t tickIndex = nearestOriginResult.tickId[nearestOriginIndex];
            for (int64_t lookedAtPatIndex = ticks.patPerTick[tickIndex].minId;
                 lookedAtPatIndex != -1 && lookedAtPatIndex <= ticks.patPerTick[tickIndex].maxId; lookedAtPatIndex++) {
                if (!playerAtTick.isAlive[lookedAtPatIndex]) {
                    continue;
                }
                for (int64_t coverEdgeIndex = coverOrigins.coverEdgesPerOrigin[originIndex].minId;
                     coverEdgeIndex != -1 && coverEdgeIndex <= coverOrigins.coverEdgesPerOrigin[originIndex].maxId;
                     coverEdgeIndex++) {
                    if (pointInRegion(coverEdges.aabbs[coverEdgeIndex], {playerAtTick.posX[lookedAtPatIndex],
                                                                         playerAtTick.posY[lookedAtPatIndex],
                                                                         playerAtTick.posZ[lookedAtPatIndex]})) {
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
    }

    PlayerInCoverEdgeResult result;
    for (int i = 0; i < numThreads; i++) {
        result.playerInCoverEdgePerRound.push_back({});
        result.playerInCoverEdgePerRound[i].minId = result.tickId.size();
        for (int j = 0; j < tmpTickId[i].size(); j++) {
            result.tickId.push_back(tmpTickId[i][j]);
            result.lookerPlayerAtTickId.push_back(tmpLookerPlayerAtTickIds[i][j]);
            result.lookerPlayerId.push_back(tmpLookerPlayerIds[i][j]);
            result.lookedAtPlayerAtTickId.push_back(tmpLookedAtPlayerAtTickIds[i][j]);
            result.lookedAtPlayerId.push_back(tmpLookedAtPlayerIds[i][j]);
            result.nearestOriginId.push_back(tmpNearestOriginIds[i][j]);
        }
        result.playerInCoverEdgePerRound[i].maxId = result.tickId.size();
    }
    result.size = result.tickId.size();
    return result;
}
