//
// Created by durst on 2/21/22.
//
//#include "queries/lookback.h"
#include "queries/train_dataset.h"

TrainDatasetResult queryTrainDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                     const Players & players, const PlayerAtTick & playerAtTick) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpSourcePlayerId[numThreads];
    vector<string> tmpSourcePlayerName[numThreads];
    vector<string> tmpDemoName[numThreads];
    vector<TrainDatasetResult::TimeStepState> tmpCurState[numThreads];
    vector<TrainDatasetResult::TimeStepState> tmpLastState[numThreads];
    vector<TrainDatasetResult::TimeStepState> tmpOldState[numThreads];

#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        //TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        //int lookbackGameTicks = tickRates.gameTickRate * LOOKBACK_SECONDS;
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            /*
            if (secondsBetweenTicks(ticks, tickRates, rounds.ticksPerRound[roundIndex].minId, tickIndex) < LOOKBACK_SECONDS) {
            }
             */

            /*
            // since spotted tracks names for spotted player, need to map that to the player index
            for (int64_t lookerPatIndex = ticks.patPerTick[tickIndex].minId;
                 lookerPatIndex != -1 && lookerPatIndex <= ticks.patPerTick[tickIndex].maxId; lookerPatIndex++) {

            }
             */
        }
    }
    return {};
}
