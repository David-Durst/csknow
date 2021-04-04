#include "indices/playerToGroup.h"
#include "omp.h"

PlayerToGroupIndex::PlayerToGroupIndex(const Position &position, const GroupingResult &groupingResult) {
    int64_t numGames = groupingResult.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpGroupsPerPlayerTick[numThreads][NUM_PLAYERS];
    vector<int64_t> tmpIndices[numThreads][NUM_PLAYERS];

    for (int i = 0; i < NUM_PLAYERS; i++) {
        groupsPerPlayerTick[i].resize(position.size);
    }

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        // need position as outer loop since since demoTick is unique only for position, may have multiple groups
        // for one position tick
        int64_t groupingIndex = groupingResult.gameStarts[gameIndex];
        for (int64_t positionIndex = position.gameStarts[gameIndex];
             positionIndex < position.gameStarts[gameIndex + 1];
             positionIndex++) {
            while (groupingIndex < groupingResult.positionIndex.size() &&
                   groupingResult.demoFile[groupingIndex] == position.fileNames[position.demoFile[positionIndex]] &&
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

    for (int thread = 0; thread < numThreads; thread++) {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int64_t t = 0; t < tmpGroupsPerPlayerTick[thread][i].size(); t++)
            groupsPerPlayerTick[i][tmpIndices[thread][i][t]] = tmpGroupsPerPlayerTick[thread][i][t];
        }
    }
}
