#include "queries/looking.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>

using std::set;
using std::map;
/*
LookingResult queryLookers(const Position & position) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int> tmpLookers[numThreads];
    vector<int> tmpLookedAt[numThreads];
    vector<int64_t> tmpGameIndex[numThreads];
    vector<int64_t> tmpGameStarts[numThreads];

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        tmpGameIndex[threadNum].push_back(gameIndex);
        tmpGameStarts[threadNum].push_back(tmpLookers[threadNum].size());

        // since spotted tracks names for spotted player, need to map that to the player index
        for (int64_t positionIndex = position.firstRowAfterWarmup[gameIndex];
             positionIndex < position.gameStarts[gameIndex+1];
             positionIndex++) {

            for (int possibleLooker = 0; possibleLooker < NUM_PLAYERS; possibleLooker++) {
                for (int possibleLookedAt = 0; possibleLookedAt < NUM_PLAYERS; possibleLookedAt++) {
                    if (possibleLookedAt == possibleLooker) {
                        continue;
                    }
                    if (!position.players[possibleLooker].isAlive[positionIndex] ||
                        !position.players[possibleLookedAt].isAlive[positionIndex]) {
                        continue;
                    }
                    AABB lookeeBox = getAABBForPlayer({position.players[possibleLookedAt].xPosition[positionIndex],
                                                       position.players[possibleLookedAt].yPosition[positionIndex],
                                                       position.players[possibleLookedAt].zPosition[positionIndex]});
                    Ray lookerEyes = getEyeCoordinatesForPlayer(
                            {position.players[possibleLooker].xPosition[positionIndex],
                             position.players[possibleLooker].yPosition[positionIndex],
                             position.players[possibleLooker].zPosition[positionIndex]},
                            {position.players[possibleLooker].xViewDirection[positionIndex],
                             position.players[possibleLooker].yViewDirection[positionIndex]}
                    );
                    double t0, t1;
                    if (intersectP(lookeeBox, lookerEyes, t0, t1)) {
                        tmpIndices[threadNum].push_back(positionIndex);
                        tmpLookers[threadNum].push_back(possibleLooker);
                        tmpLookedAt[threadNum].push_back(possibleLookedAt);
                    }
                }
            }
        }
    }

    LookingResult result;
    result.gameStarts.resize(position.fileNames.size() + 1);
    result.fileNames = position.fileNames;
    for (int i = 0; i < numThreads; i++) {
        // for all games in thread, note position as position in thread plus start of thread results
        for (int j = 0; j < tmpGameStarts[i].size(); j++) {
            result.gameStarts[tmpGameIndex[i][j]] = tmpGameStarts[i][j] + result.positionIndex.size();
        }

        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.lookers.push_back(tmpLookers[i][j]);
            result.lookedAt.push_back({tmpLookedAt[i][j]});
        }
    }
    result.gameStarts[numGames] = result.positionIndex.size();
    return result;
}

*/