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
    vector<double> tmpMinX[numThreads];
    vector<double> tmpMinY[numThreads];
    vector<double> tmpMinZ[numThreads];
    vector<double> tmpMaxX[numThreads];
    vector<double> tmpMaxY[numThreads];
    vector<double> tmpMaxZ[numThreads];

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        // don't repeat cheating events within window, decrease duplicate events
        // so track all recently grouped and remove old ones
        set<vector<int>> recentlyGrouped;
        map<int64_t, vector<vector<int>>> timeOfGrouping;
        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex = position.getPlayerNameToIndex(gameIndex);

        // iterating over each possible window
        for (int64_t windowStartIndex = position.firstRowAfterWarmup[gameIndex];
             windowStartIndex + GROUPING_WINDOW_SIZE < position.gameStarts[gameIndex+1];
             windowStartIndex++) {

            // track who I currently need
            // double buffer so no need to remove untracked, just won't add them after each frame
            set<vector<int>> possibleGroups[2];
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
                        if (computeDistance(position, playerAIndex, playerBIndex, windowStartIndex, windowStartIndex) < GROUPING_DISTANCE &&
                            computeDistance(position, playerAIndex, playerCIndex, windowStartIndex, windowStartIndex) < GROUPING_DISTANCE &&
                            recentlyGrouped.find({playerAIndex, playerBIndex, playerCIndex}) == recentlyGrouped.end()) {
                            possibleGroups[curReader].insert({playerAIndex, playerBIndex, playerCIndex});
                            AABB region{{1E6, 1E6, 1E6}, {-1E6, -1E6, -1E6}};
                            adjustMinMaxRegion(position, windowStartIndex, region, playerAIndex, playerBIndex, playerCIndex);
                            groupRegions.insert({{playerAIndex, playerBIndex, playerCIndex}, region});
                        }
                    }
                }
            }

            // for each window, as long as any cheaters possibly left in that window
            for (int64_t windowIndex = windowStartIndex; windowIndex < windowStartIndex + GROUPING_WINDOW_SIZE && !possibleGroups[curReader].empty();
                 windowIndex++) {
                // for all needed player supdate only needed players
                for (const auto & possibleGroup : possibleGroups[curReader]) {
                    if (computeDistance(position, possibleGroup[0], possibleGroup[1], windowStartIndex, windowStartIndex) < GROUPING_DISTANCE &&
                        computeDistance(position, possibleGroup[0], possibleGroup[2], windowStartIndex, windowStartIndex) < GROUPING_DISTANCE) {
                        possibleGroups[curWriter].insert(possibleGroup);
                        adjustMinMaxRegion(position, windowIndex, groupRegions[possibleGroup], possibleGroup[0],
                                           possibleGroup[1], possibleGroup[2]);
                    }
                }
                possibleGroups[curReader].clear();
                curReader = (curReader + 1) % 2;
                curWriter = (curWriter + 1) % 2;
            }

            // remove old windows from recentlyGroupd
            if (timeOfGrouping.find(windowStartIndex - GROUPING_WINDOW_SIZE) != timeOfGrouping.end()) {
                for (const auto & oldGroup : timeOfGrouping[windowStartIndex - GROUPING_WINDOW_SIZE]) {
                    recentlyGrouped.erase(oldGroup);
                }
                timeOfGrouping.erase(windowStartIndex - GROUPING_WINDOW_SIZE);
            }

            // save all found groups
            timeOfGrouping[windowStartIndex] = {};
            for (const auto & group : possibleGroups[curReader]) {
                timeOfGrouping[windowStartIndex].push_back(group);
                recentlyGrouped.insert(group);
                tmpIndices[threadNum].push_back(windowStartIndex);
                tmpTeamates[threadNum].push_back(group);
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
