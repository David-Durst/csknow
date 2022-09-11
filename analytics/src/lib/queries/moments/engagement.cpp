//
// Created by durst on 9/11/22.
//

#include "queries/moments/engagement.h"

EngagementResult queryEngagementResult(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick, const Hurt & hurt) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpStartTickId[numThreads];
    vector<int64_t> tmpEndTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<vector<int64_t>> tmpPlayerId[numThreads];
    vector<vector<AggressionRole>> tmpRole[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;
// for each round
    // for each player - identify current path - if in any regions of a path, or next region they will be in
    // for each time step in path - first player to shoot/be shot by enemy is pusher. baiter is everyone on same path
    // lurker is someone on different path
#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < std::min(6L, rounds.size); roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpStartTickId[threadNum].size());

        int64_t ticksSinceLastEngagement = 10000; // just some large number that will enable first engagement in each round
        // assuming first position is less than first kills
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

        }

    }
}
