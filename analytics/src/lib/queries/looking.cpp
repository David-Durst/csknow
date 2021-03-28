#include "queries/looking.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>

using std::set;
using std::map;
#define BAIT_WINDOW_SIZE 64

LookingResult queryLookers(const Position & position) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int> tmpLookers[numThreads];
    vector<int> tmpLookedAt[numThreads];

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        // assuming first position is less than first kills
        int64_t positionGameStartIndex = position.gameStarts[gameIndex];

        // since spotted tracks names for spotted player, need to map that to the player index
        for (int64_t positionIndex = position.firstRowAfterWarmup[gameIndex];
             positionIndex < position.gameStarts[gameIndex+1];
             positionIndex++) {

            for (int possibleLooker = 0; possibleLooker < NUM_PLAYERS; possibleLooker++) {
                for (int possibleLookedAt = 0; possibleLookedAt < NUM_PLAYERS; possibleLookedAt++) {
                    if (possibleLookedAt == possibleLooker) {
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
    result.gameStarts.resize(position.fileNames.size());
    result.fileNames = position.fileNames;
    for (int i = 0; i < numThreads; i++) {
        result.gameStarts[position.demoFile[tmpIndices[i][0]]] = result.positionIndex.size();
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.lookers.push_back(tmpLookers[i][j]);
            result.lookedAt.push_back({tmpLookedAt[i][j]});
        }
    }
    return result;
}

