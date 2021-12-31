//
// Created by durst on 12/31/21.
//

#include "queries/player_looking_at_cover_edge.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
#include "cmath"

PlayerLookingAtCoverEdgeResult
queryPlayerLookingAtCoverEdge(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                              const CoverOrigins & coverOrigins, const CoverEdges & coverEdges,
                              const NearestOriginResult & nearestOriginResult) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpCurPlayerAtTickIds[numThreads];
    vector<int64_t> tmpCurPlayerIds[numThreads];
    vector<int64_t> tmpLookerPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookerPlayerIds[numThreads];
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
                    Ray playerEyeCoord = getEyeCoordinatesForPlayer(
                            {playerAtTick.posX[lookedAtPatIndex], playerAtTick.posY[lookedAtPatIndex],
                             playerAtTick.posZ[lookedAtPatIndex]},
                            {playerAtTick.viewX[lookedAtPatIndex], playerAtTick.viewY[lookedAtPatIndex]});

                    double t0, t1;
                    if (intersectP(coverEdges.aabbs[coverEdgeIndex], playerEyeCoord, t0, t1)) {
                        tmpTickId[threadNum].push_back(tickIndex);
                        tmpCurPlayerAtTickIds[threadNum].push_back(lookedAtPatIndex);
                        tmpCurPlayerIds[threadNum].push_back(playerAtTick.playerId[lookedAtPatIndex]);
                        tmpLookerPlayerAtTickIds[threadNum].push_back(nearestOriginResult.playerAtTickId[originIndex]);
                        tmpLookerPlayerIds[threadNum].push_back(nearestOriginResult.playerId[originIndex]);
                        tmpNearestOriginIds[threadNum].push_back(nearestOriginIndex);
                    }

                }
            }

        }
    }

    PlayerLookingAtCoverEdgeResult result;
    for (int i = 0; i < numThreads; i++) {
        result.playerLookingAtCoverEdgePerRound.push_back({});
        result.playerLookingAtCoverEdgePerRound[i].minId = result.tickId.size();
        for (int j = 0; j < tmpTickId[i].size(); j++) {
            result.tickId.push_back(tmpTickId[i][j]);
            result.curPlayerAtTickId.push_back(tmpCurPlayerAtTickIds[i][j]);
            result.curPlayerId.push_back(tmpCurPlayerIds[i][j]);
            result.lookerPlayerAtTickId.push_back(tmpLookerPlayerAtTickIds[i][j]);
            result.lookerPlayerId.push_back(tmpLookerPlayerIds[i][j]);
            result.nearestOriginId.push_back(tmpNearestOriginIds[i][j]);
        }
        result.playerLookingAtCoverEdgePerRound[i].maxId = result.tickId.size();
    }
    result.size = result.tickId.size();
    return result;
}
