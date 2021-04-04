#include "queries/groupInSequenceOfRegions.h"
#include "queries/grouping.h"
#include "geometry.h"
#include <omp.h>
#include <queue>
#include <set>
#include <map>
using std::set;
using std::map;

GroupInSequenceOfRegionsResult queryGroupingInSequenceOfRegions(const Position & position,
                                                                const GroupingResult & grouping,
                                                                vector<CompoundAABB> sequenceOfRegions,
                                                                vector<bool> wantToReachRegions) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<vector<int>> tmpTeamates[numThreads];
    vector<int64_t> tmpEndTick[numThreads];
    vector<vector<string>> tmpMemberInRegion[numThreads];
    vector<vector<int64_t>> tmpTickInRegion[numThreads];
    vector<vector<double>> tmpX[numThreads];
    vector<vector<double>> tmpY[numThreads];
    vector<vector<double>> tmpZ[numThreads];
    vector<int64_t> tmpGameIndex[numThreads];
    vector<int64_t> tmpGameStarts[numThreads];

    // find any frame when at least 3 people from a team are together
    // this means i can track all groups of 3 people togther, but only record 1 and have good recall
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
    }

    GroupInSequenceOfRegionsResult result(sequenceOfRegions, wantToReachRegions);
    result.gameStarts.resize(position.fileNames.size());
    for (int i = 0; i < numThreads; i++) {
        // for all games in thread, note position as position in thread plus start of thread results
        for (int j = 0; j < tmpGameStarts[i].size(); j++) {
            result.gameStarts[tmpGameIndex[i][j]] = tmpGameStarts[i][j] + result.positionIndex.size();
        }

        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.teammates.push_back(tmpTeamates[i][j]);
            result.endTick.push_back((tmpEndTick[i][j]));
            result.memberInRegion.push_back(tmpMemberInRegion[i][j]);
            result.tickInRegion.push_back(tmpTickInRegion[i][j]);
            result.xInRegion.push_back({tmpX[i][j]});
            result.yInRegion.push_back({tmpY[i][j]});
            result.zInRegion.push_back({tmpZ[i][j]});
        }
    }
    return result;
}
