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
                                                                const GroupingResult & groupingResult,
                                                                vector<CompoundAABB> sequenceOfRegions,
                                                                vector<bool> wantToReachRegions,
                                                                vector<bool> stillGrouped,
                                                                set<int> teams) {
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
//#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        int64_t groupingIndex = groupingResult.gameStarts[gameIndex];
        for (int64_t positionIndex = position.firstRowAfterWarmup[gameIndex];
             positionIndex < position.gameStarts[gameIndex + 1];
             positionIndex++) {
            while (groupingIndex < groupingResult.positionIndex.size() &&
                    groupingIndex < groupingResult.gameStarts[gameIndex + 1] &&
                   groupingResult.positionIndex[groupingIndex] <= positionIndex) {
                if (groupingResult.positionIndex[groupingIndex] != positionIndex) {
                    std::cerr << "bad groupingResult at grouping index " << groupingIndex << std::endl;
                    groupingIndex++;
                    continue;
                }
                vector<string> memberCurGrouping;
                vector<int64_t> tickInRegionCurGrouping;
                vector<double> xCurGrouping;
                vector<double> yCurGrouping;
                vector<double> zCurGrouping;
                // filter by teams
                if (teams.find(position.players[groupingResult.teammates[groupingIndex][0]].team[positionIndex]) ==
                    teams.end()) {
                    groupingIndex++;
                    continue;
                }

                // for each region in sequence, find first time a player is in first region (if wantToReachRegion)
                // or verify not in region (if not wantToReachRegion)
                // also check if still in grouping (if required for that step in sequence)
                // if above hold true, continue to later regions
                bool foundInstanceInRegion = false;

                // need to track positionIndex across all events in sequence
                int64_t positionIndexSeq = groupingResult.positionIndex[groupingIndex];
                for (int seqIndex = 0; seqIndex < sequenceOfRegions.size(); seqIndex++) {
                    bool playerInRegions = false;
                    // track last value of positionIndexInGroup
                    int64_t lastCheckedPositionIndex = 0;
                    // stop checking if everyone dead
                    bool groupAlive = true;
                    for (int64_t positionIndexInGroup = groupingResult.positionIndex[groupingIndex];
                        // always stop at end of round
                        !position.roundEnd[positionIndexInGroup] &&
                        // stop once found someone
                        !playerInRegions &&
                        // stop once everyone in group dead
                        groupAlive &&
                        // if also require still grouped, check that still in group
                        (!stillGrouped[seqIndex] ||
                            position.demoTickNumber[positionIndexInGroup] <= groupingResult.endTick[groupingIndex]);
                         positionIndexInGroup++) {
                        lastCheckedPositionIndex = positionIndexInGroup;
                        for (const auto & member : groupingResult.teammates[groupingIndex]) {
                            Vec3 memberPosition = {position.players[member].xPosition[positionIndexInGroup],
                                                   position.players[member].yPosition[positionIndexInGroup],
                                                   position.players[member].zPosition[positionIndexInGroup]};
                            if (position.players[member].isAlive[positionIndexInGroup] &&
                                pointInCompoundRegion(sequenceOfRegions[seqIndex], memberPosition)) {
                                memberCurGrouping.push_back(position.players[member].name[positionIndexInGroup]);
                                tickInRegionCurGrouping.push_back(position.demoTickNumber[positionIndexInGroup]);
                                xCurGrouping.push_back(memberPosition.x);
                                yCurGrouping.push_back(memberPosition.y);
                                zCurGrouping.push_back(memberPosition.z);
                                playerInRegions = true;
                                break;
                            }
                        }
                    }
                    // stop if didn't find region and wanted to or found region and didn't want to
                    if ((playerInRegions && !wantToReachRegions[seqIndex]) ||
                        (!playerInRegions && wantToReachRegions[seqIndex])) {
                        break;
                    }
                    // other wise this iteration was successful. if this the last iteration and was successful,
                    // then add results vector
                    else if (seqIndex == sequenceOfRegions.size() - 1) {
                        tmpTeamates[threadNum].push_back(groupingResult.teammates[groupingIndex]);
                        tmpEndTick[threadNum].push_back(position.demoTickNumber[lastCheckedPositionIndex]);
                        tmpMemberInRegion[threadNum].push_back(memberCurGrouping);
                        tmpX[threadNum].push_back(xCurGrouping);
                        tmpY[threadNum].push_back(yCurGrouping);
                        tmpZ[threadNum].push_back(zCurGrouping);
                    }
                    // if this was a succesful iteration but not last and wanted to be in a region
                    // updated positionIndexSeq to start after this event
                    // don't update if don't want player to be in region as there was no positive result, so
                    // nothing to start after
                    else if (playerInRegions) {
                        positionIndexSeq = lastCheckedPositionIndex;
                    }
                    // disable group is everyone dead
                    groupAlive = false;
                    for (const auto & member : groupingResult.teammates[groupingIndex]) {
                        if (position.players[member].isAlive) {
                            groupAlive = true;
                            break;
                        }
                    }
                }
                groupingIndex++;
            }
        }
    }

    GroupInSequenceOfRegionsResult result(sequenceOfRegions, wantToReachRegions, stillGrouped);
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
