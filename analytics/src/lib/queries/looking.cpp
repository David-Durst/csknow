#include "queries/looking.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>

using std::set;
using std::map;

LookingResult queryLookers(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpLookerPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookerPlayerIds[numThreads];
    vector<int64_t> tmpLookedAtPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookedAtPlayerIds[numThreads];

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
            tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            vector<Ray> playerEyes;
            vector<AABB> playerAABBs;
            vector<int64_t> playerIds;
            vector<int64_t> patIds;
            vector<bool> alive;

            // since spotted tracks names for spotted player, need to map that to the player index
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId; patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                playerEyes.push_back(getEyeCoordinatesForPlayer(
                        {playerAtTick.posX[patIndex],
                         playerAtTick.posY[patIndex],
                         playerAtTick.posZ[patIndex]},
                        {playerAtTick.viewX[patIndex],
                         playerAtTick.viewY[patIndex]}
                ));
                playerAABBs.push_back(getAABBForPlayer({playerAtTick.posX[patIndex],
                                                        playerAtTick.posY[patIndex],
                                                        playerAtTick.posZ[patIndex]}
                ));
                playerIds.push_back(playerAtTick.playerId[patIndex]);
                patIds.push_back(patIndex);
                alive.push_back(playerAtTick.isAlive[patIndex]);
            }

            for (int lookerId = 0; lookerId < playerEyes.size(); lookerId++) {
                if (!alive[lookerId]) {
                    continue;
                }
                for (int lookedAtId = 0; lookedAtId < playerEyes.size(); lookedAtId++) {
                    if (!alive[lookerId] || lookerId == lookedAtId) {
                        continue;
                    }

                    double t0, t1;
                    if (intersectP(playerAABBs[lookedAtId], playerEyes[lookerId], t0, t1)) {
                        tmpTickId[threadNum].push_back(tickIndex);
                        tmpLookerPlayerAtTickIds[threadNum].push_back(patIds[lookerId]);
                        tmpLookerPlayerIds[threadNum].push_back(playerIds[lookerId]);
                        tmpLookedAtPlayerAtTickIds[threadNum].push_back(patIds[lookedAtId]);
                        tmpLookedAtPlayerIds[threadNum].push_back(playerIds[lookedAtId]);
                    }
                }
            }
        }
    }

    LookingResult result;
    for (int i = 0; i < numThreads; i++) {
        result.lookersPerRound.push_back({});
        result.lookersPerRound[i].minId = result.tickId.size();
        for (int j = 0; j < tmpTickId[i].size(); j++) {
            result.tickId.push_back(tmpTickId[i][j]);
            result.lookerPlayerAtTickId.push_back(tmpLookerPlayerAtTickIds[i][j]);
            result.lookerPlayerId.push_back(tmpLookerPlayerIds[i][j]);
            result.lookedAtPlayerAtTickId.push_back(tmpLookedAtPlayerAtTickIds[i][j]);
            result.lookedAtPlayerId.push_back(tmpLookedAtPlayerIds[i][j]);
        }
        result.lookersPerRound[i].maxId = result.tickId.size();
    }
    result.size = result.lookerPlayerId.size();
    return result;
}
