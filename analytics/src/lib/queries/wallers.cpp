#include "queries/wallers.h"
#include <omp.h>
#define WALL_WINDOW_SIZE 32

PredicateResult queryWallers(const Position & position, const Spotted & spotted) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];

    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // x angle 0 is looking towards large x positions
    // x angle 180 is looking towards smaller x posotions
    // x angle -90 is looking towards smaller y positions
    // x angle 90 is looking towards smaller y positions
    // y angle 90 is looking towards smaller z positions
    // y angle -90 is looking towards larger z positions
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        for (int64_t windowStartIndex = position.gameStarts[gameIndex];
            windowStartIndex + WALL_WINDOW_SIZE < position.gameStarts[gameIndex+1];
            windowStartIndex++) {
            for (int64_t windowIndex = windowStartIndex; windowIndex < windowStartIndex + WALL_WINDOW_SIZE; windowIndex++) {

            }
        }
    }

    PredicateResult result;
    result.collectResults(tmpIndices, numThreads);
    return result;
}
