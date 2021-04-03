#include "queries/grouping.h"
#include "geometry.h"
#include <omp.h>
#include <queue>
#include <set>
#include <map>
using std::set;
using std::priority_queue;
using std::map;

static inline __attribute__((always_inline))
double quadMin(double base, double a, double b, double c) {
    return std::min(base, std::min(a, std::min(b, c)));
}

static inline __attribute__((always_inline))
double quadMax(double base, double a, double b, double c) {
    return std::max(base, std::max(a, std::max(b, c)));
}

static inline __attribute__((always_inline))
void adjustMinMaxRegion(const Position & position, int64_t index, AABB & region, int playerA, int playerB, int playerC) {
    region.min.x = quadMin(region.min.x, position.players[playerA].xPosition[index],
                           position.players[playerB].xPosition[index], position.players[playerC].xPosition[index]);
    region.min.y = quadMin(region.min.y, position.players[playerA].yPosition[index],
                           position.players[playerB].yPosition[index], position.players[playerC].yPosition[index]);
    region.min.z = quadMin(region.min.z, position.players[playerA].zPosition[index],
                           position.players[playerB].zPosition[index], position.players[playerC].zPosition[index]);
    region.max.x = quadMax(region.max.x, position.players[playerA].xPosition[index],
                           position.players[playerB].xPosition[index], position.players[playerC].xPosition[index]);
    region.max.y = quadMax(region.max.y, position.players[playerA].yPosition[index],
                           position.players[playerB].yPosition[index], position.players[playerC].yPosition[index]);
    region.max.z = quadMax(region.max.z, position.players[playerA].zPosition[index],
                           position.players[playerB].zPosition[index], position.players[playerC].zPosition[index]);
}

GroupingResult queryGrouping(const Position & position) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<vector<int>> tmpTeamates[numThreads];
    vector<int64_t> tmpEndTick[numThreads];
    vector<double> tmpMinX[numThreads];
    vector<double> tmpMinY[numThreads];
    vector<double> tmpMinZ[numThreads];
    vector<double> tmpMaxX[numThreads];
    vector<double> tmpMaxY[numThreads];
    vector<double> tmpMaxZ[numThreads];

    // find any frame when at least 3 people from a team are together
    // this means i can track all groups of 3 people togther, but only record 1 and have good recall
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        // don't repeat cheating events within window, decrease duplicate events
        // so track last ending time for each group
        int64_t lastEndTimeForGroup[NUM_PLAYERS][NUM_PLAYERS][NUM_PLAYERS];
        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex = position.getPlayerNameToIndex(gameIndex);

        // iterating over each possible window
        for (int64_t windowStartIndex = position.firstRowAfterWarmup[gameIndex];
             windowStartIndex + GROUPING_WINDOW_SIZE < position.gameStarts[gameIndex+1];
             windowStartIndex++) {

            // track who I currently need
            // double buffer so no need to remove untracked, just won't add them after each frame
            set<vector<int>> possibleGroups[2];
            set<vector<int>> confirmedGroups[2];
            map<vector<int>, AABB> groupRegions;
            int curReader = 0, curWriter = 1;
            // start tracking all groups at current tick who weren't grouped together in last window
            for (int playerAIndex = 0; playerAIndex < NUM_PLAYERS; playerAIndex++) {
                if (!position.players[playerAIndex].isAlive) {
                    continue;
                }
                for (int playerBIndex = playerAIndex + 1; playerBIndex < NUM_PLAYERS; playerBIndex++) {
                    if (!position.players[playerBIndex].isAlive) {
                        continue;
                    }
                    for (int playerCIndex = playerBIndex + 1; playerCIndex < NUM_PLAYERS; playerCIndex++) {
                        if (!position.players[playerCIndex].isAlive) {
                            continue;
                        }
                        AABB region{{1E6, 1E6, 1E6}, {-1E6, -1E6, -1E6}};
                        adjustMinMaxRegion(position, windowStartIndex, region, playerAIndex, playerBIndex, playerCIndex);
                        if (position.players[playerAIndex].team[windowStartIndex] == position.players[playerBIndex].team[windowStartIndex] &&
                            position.players[playerAIndex].team[windowStartIndex] == position.players[playerCIndex].team[windowStartIndex] &&
                            computeAABBSize(region) < GROUPING_DISTANCE &&
                            lastEndTimeForGroup[playerAIndex][playerBIndex][playerCIndex] < windowStartIndex) {
                            possibleGroups[curReader].insert({playerAIndex, playerBIndex, playerCIndex});
                            groupRegions.insert({{playerAIndex, playerBIndex, playerCIndex}, region});
                        }
                    }
                }
            }

            int64_t lastTimeInWindow;
            // for as long as any groups possibly left
            for (int64_t windowIndex = windowStartIndex; !position.roundEnd[windowIndex] && !possibleGroups[curReader].empty();
                 windowIndex++) {
                lastTimeInWindow = windowIndex;
                // only track possible groups for this window
                for (const auto & possibleGroup : possibleGroups[curReader]) {
                    AABB regionCopy = groupRegions[possibleGroup];
                    adjustMinMaxRegion(position, windowIndex, regionCopy, possibleGroup[0],
                                       possibleGroup[1], possibleGroup[2]);
                    if (computeAABBSize(groupRegions[possibleGroup]) < GROUPING_DISTANCE) {
                        possibleGroups[curWriter].insert(possibleGroup);
                        groupRegions[possibleGroup] = regionCopy;
                    }
                    else if (windowIndex >= windowStartIndex + GROUPING_WINDOW_SIZE){
                        confirmedGroups[curWriter].insert(possibleGroup);
                        lastEndTimeForGroup[possibleGroup[0]][possibleGroup[1]][possibleGroup[2]] = windowIndex;
                    }
                }
                possibleGroups[curReader].clear();
                confirmedGroups[curReader].clear();
                curReader = (curReader + 1) % 2;
                curWriter = (curWriter + 1) % 2;
            }

            // for all possible groups not culled (i.e. lasted until round end or last tick with a group) add them to confirmedGroups
            for (const auto & group : possibleGroups[curReader]) {
                confirmedGroups[curWriter].insert(group);
                lastEndTimeForGroup[group[0]][group[1]][group[2]] = lastTimeInWindow;
            }

            for (const auto & group : confirmedGroups[curReader]) {
                tmpIndices[threadNum].push_back(windowStartIndex);
                tmpTeamates[threadNum].push_back(group);
                tmpEndTick[threadNum].push_back(lastEndTimeForGroup[group[0]][group[1]][group[2]]);
                tmpMinX[threadNum].push_back(groupRegions[group].min.x);
                tmpMinY[threadNum].push_back(groupRegions[group].min.y);
                tmpMinZ[threadNum].push_back(groupRegions[group].min.z);
                tmpMaxX[threadNum].push_back(groupRegions[group].max.x);
                tmpMaxY[threadNum].push_back(groupRegions[group].max.y);
                tmpMaxZ[threadNum].push_back(groupRegions[group].max.z);
            }
        }
    }

    GroupingResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.teammates.push_back(tmpTeamates[i][j]);
            result.endTick.push_back((tmpEndTick[i][j]));
            result.minX.push_back({tmpMinX[i][j]});
            result.minY.push_back({tmpMinY[i][j]});
            result.minZ.push_back({tmpMinZ[i][j]});
            result.maxX.push_back({tmpMaxX[i][j]});
            result.maxY.push_back({tmpMaxY[i][j]});
            result.maxZ.push_back({tmpMaxZ[i][j]});
        }
    }
    return result;
}
