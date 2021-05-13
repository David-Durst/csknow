#include "indices/playerToGroup.h"
#include "omp.h"
/*
PlayerToGroupIndex(const Ticks & ticks, const PlayerAtTick & pat, const GroupingResult & groupingResult) {
    int64_t numGames = groupingResult.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpGroupsPerPlayerTick[numThreads][NUM_PLAYERS];

    for (int i = 0; i < NUM_PLAYERS; i++) {
        groupsPerPlayerTick[i].resize(position.size);
    }

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        // need position as outer loop since since demoTick is unique only for position, may have multiple groups
        // for one position tick
        int64_t groupingIndex = groupingResult.gameStarts[gameIndex];
        for (int64_t positionIndex = position.gameStarts[gameIndex];
             positionIndex < position.gameStarts[gameIndex + 1];
             positionIndex++) {
            while (groupingIndex < groupingResult.positionIndex.size() &&
                   groupingIndex < groupingResult.gameStarts[gameIndex + 1] &&
                   groupingResult.positionIndex[groupingIndex] <= position.demoTickNumber[positionIndex]) {
                if (groupingResult.positionIndex[groupingIndex] != position.demoTickNumber[positionIndex]) {
                    std::cerr << "bad groupingResult at grouping index " << groupingIndex << std::endl;
                }
                else {
                    for (int64_t positionIndexInGroup = groupingResult.positionIndex[groupingIndex];
                         positionIndexInGroup <= position.demoTickNumber[groupingResult.endTick[groupingIndex]];
                         positionIndexInGroup++
                    ) {
                        groupsPerPlayerTick[groupingResult.teammates[groupingIndex][0]][positionIndexInGroup]
                            .push_back(groupingIndex);
                        groupsPerPlayerTick[groupingResult.teammates[groupingIndex][1]][positionIndexInGroup]
                                .push_back(groupingIndex);
                        groupsPerPlayerTick[groupingResult.teammates[groupingIndex][2]][positionIndexInGroup]
                                .push_back(groupingIndex);
                    }
                }
            }
        }
    }
}
     */
