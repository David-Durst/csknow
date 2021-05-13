#include "queries/velocity.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>

using std::set;
using std::map;
/*
VelocityResult queryVelocity(const Position & position) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<double> tmpVelocities[numThreads][NUM_PLAYERS];
    vector<int64_t> tmpGameIndex[numThreads];
    vector<int64_t> tmpGameStarts[numThreads];

    VelocityResult result;
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        tmpGameIndex[threadNum].push_back(gameIndex);
        tmpGameStarts[threadNum].push_back(tmpVelocities[threadNum][0].size());

        // since spotted tracks names for spotted player, need to map that to the player index
        for (int64_t positionIndex = position.firstRowAfterWarmup[gameIndex];
             // since want to have velocity at every position (even if garbage), will check max for window overrun
             // later
             positionIndex < position.gameStarts[gameIndex+1];
             positionIndex++) {

            tmpIndices[threadNum].push_back(positionIndex);
            for (int playerID = 0; playerID < NUM_PLAYERS; playerID++) {
                double tickDistance;
                if (!position.players[playerID].isAlive[positionIndex]) {
                    tickDistance = 0;
                }
                else {
                    tickDistance = computeDistance(position, playerID, playerID, positionIndex,
                                                   std::min(positionIndex+1, position.gameStarts[gameIndex+1]-1));
                }
                tmpVelocities[threadNum][playerID].push_back(tickDistance * TICKS_PER_SECOND / ((double) result.ticksPerEvent));
            }
        }
    }

    result.gameStarts.resize(position.fileNames.size() + 1);
    result.fileNames = position.fileNames;
    for (int i = 0; i < numThreads; i++) {
        // for all games in thread, note position as position in thread plus start of thread results
        for (int j = 0; j < tmpGameStarts[i].size(); j++) {
            result.gameStarts[tmpGameIndex[i][j]] = tmpGameStarts[i][j] + result.positionIndex.size();
        }

        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            for (int playerID = 0; playerID < NUM_PLAYERS; playerID++) {
                result.resultsPerPlayer[playerID].push_back(tmpVelocities[i][playerID][j]);
            }
        }
    }
    result.gameStarts[numGames] = result.positionIndex.size();
    return result;
}

*/