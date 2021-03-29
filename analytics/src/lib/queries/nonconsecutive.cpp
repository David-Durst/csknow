#include "queries/nonconsecutive.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>

using std::set;
using std::map;

NonConsecutiveResult queryNonConsecutive(const Position & position) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int64_t> tmpNextTicks[numThreads];
    vector<int64_t> tmpGameIndex[numThreads];
    vector<int64_t> tmpGameStarts[numThreads];

    NonConsecutiveResult result;
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        tmpGameIndex[threadNum].push_back(gameIndex);
        tmpGameStarts[threadNum].push_back(tmpNextTicks[threadNum].size());

        // since spotted tracks names for spotted player, need to map that to the player index
        for (int64_t positionIndex = position.firstRowAfterWarmup[gameIndex];
             positionIndex < position.gameStarts[gameIndex+1] - result.ticksPerEvent;
             positionIndex++) {

            if (position.demoTickNumber[positionIndex] + 1 != position.demoTickNumber[positionIndex+1]) {
                tmpIndices[threadNum].push_back(positionIndex);
                tmpNextTicks[threadNum].push_back(position.demoTickNumber[positionIndex+1]);
            }
        }
    }

    result.gameStarts.resize(position.fileNames.size());
    result.fileNames = position.fileNames;
    for (int i = 0; i < numThreads; i++) {
        // for all games in thread, note position as position in thread plus start of thread results
        for (int j = 0; j < tmpGameStarts[i].size(); j++) {
            result.gameStarts[tmpGameIndex[i][j]] = tmpGameStarts[i][j] + result.positionIndex.size();
        }

        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.nextTicks.push_back(tmpNextTicks[i][j]);
        }
    }
    return result;
}

