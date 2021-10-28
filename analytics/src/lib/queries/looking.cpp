#include "queries/looking.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
#include "cmath"

using std::set;
using std::map;
// max time you can look back in ms
// this assumes tick in demo for every tick in game
const int maxLookBackTime = 300;
const int clInterp = 31;

LookingResult queryLookers(const Games & games, const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpLookerPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookerPlayerIds[numThreads];
    vector<int64_t> tmpLookedAtPlayerAtTickIds[numThreads];
    vector<int64_t> tmpLookedAtPlayerIds[numThreads];

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        int demoTickRate = static_cast<int>(games.demoTickRate[rounds.gameId[roundIndex]]);
        int maxLookBackDemoTicks = ceil(demoTickRate * maxLookBackTime / 1000.0);
        int gameTickRate = static_cast<int>(games.gameTickRate[rounds.gameId[roundIndex]]);

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId + maxLookBackDemoTicks;
            tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            // since spotted tracks names for spotted player, need to map that to the player index
            for (int64_t lookerPatIndex = ticks.patPerTick[tickIndex].minId;
                    lookerPatIndex != -1 && lookerPatIndex <= ticks.patPerTick[tickIndex].maxId; lookerPatIndex++) {
                if (!playerAtTick.isAlive[lookerPatIndex]) {
                    continue;
                }
                Ray playerEyes = getEyeCoordinatesForPlayer(
                        {playerAtTick.posX[lookerPatIndex],
                         playerAtTick.posY[lookerPatIndex],
                         playerAtTick.posZ[lookerPatIndex]},
                        {playerAtTick.viewX[lookerPatIndex],
                         playerAtTick.viewY[lookerPatIndex]}
                );

                // find the right tick within lookback window for other players' positions
                // only looking on ticks, so ignore interpolating between ticks for now, just take one of two old ticks
                // used for interpolation
                double lookbackGameTicks = std::max(1.0, gameTickRate * (clInterp + playerAtTick.ping[lookerPatIndex]) / 1000.0);
                int lookbackDemoTicks = 1;
                for (;ticks.gameTickNumber[tickIndex - lookbackDemoTicks] > ticks.gameTickNumber[tickIndex] - lookbackGameTicks &&
                        lookbackDemoTicks < maxLookBackDemoTicks;
                        lookbackDemoTicks++);
                int64_t lookedAtTickId = tickIndex - lookbackDemoTicks;

                for (int64_t lookedAtPatIndex = ticks.patPerTick[lookedAtTickId].minId;
                     lookedAtPatIndex != -1 && lookedAtPatIndex <= ticks.patPerTick[lookedAtTickId].maxId;
                     lookedAtPatIndex++) {
                    if (!playerAtTick.isAlive[lookedAtPatIndex] ||
                        (playerAtTick.playerId[lookerPatIndex] == playerAtTick.playerId[lookedAtPatIndex])) {
                        continue;
                    }

                    AABB lookedAtAABB = getAABBForPlayer({
                        playerAtTick.posX[lookedAtPatIndex],
                        playerAtTick.posY[lookedAtPatIndex],
                        playerAtTick.posZ[lookedAtPatIndex]
                    });

                    double t0, t1;
                    if (intersectP(lookedAtAABB, playerEyes, t0, t1)) {
                        tmpTickId[threadNum].push_back(tickIndex);
                        tmpLookerPlayerAtTickIds[threadNum].push_back(lookerPatIndex);
                        tmpLookerPlayerIds[threadNum].push_back(playerAtTick.playerId[lookerPatIndex]);
                        tmpLookedAtPlayerAtTickIds[threadNum].push_back(lookedAtPatIndex);
                        tmpLookedAtPlayerIds[threadNum].push_back(playerAtTick.playerId[lookedAtPatIndex]);
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
