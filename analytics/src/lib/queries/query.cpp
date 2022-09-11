//
// Created by durst on 9/11/22.
//

#include "queries/query.h"

void mergeThreadResults(int numThreads, vector<RangeIndexEntry> &rowIndicesPerRound, const vector<int64_t> tmpRoundIds[],
                        const vector<int64_t> tmpRoundStarts[], const vector<int64_t> tmpRoundSizes[],
                        vector<int64_t> & resultStartTickId, int64_t & resultSize,
                        std::function<void(int64_t, int64_t)> appendToResult) {
    vector<int64_t> roundsProcessedPerThread(numThreads, 0);
    while (true) {
        bool roundToProcess = false;
        int64_t minThreadId = -1;
        int64_t minRoundId = -1;
        for (int64_t threadId = 0; threadId < numThreads; threadId++) {
            if (roundsProcessedPerThread[threadId] < tmpRoundIds[threadId].size()) {
                roundToProcess = true;
                if (minThreadId == -1 || tmpRoundIds[threadId][roundsProcessedPerThread[threadId]] < minRoundId) {
                    minThreadId = threadId;
                    minRoundId = tmpRoundIds[minThreadId][roundsProcessedPerThread[minThreadId]];
                }

            }
        }
        if (!roundToProcess) {
            break;
        }
        rowIndicesPerRound.push_back({});
        rowIndicesPerRound[minRoundId].minId = resultStartTickId.size();
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            appendToResult(minThreadId, tmpRowId);
        }
        rowIndicesPerRound[minRoundId].maxId = resultStartTickId.size() - 1;
        roundsProcessedPerThread[minThreadId]++;
    }
    resultSize = resultStartTickId.size();
}

