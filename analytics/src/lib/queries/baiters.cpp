#include "queries/baiters.h"
#include "geometry.h"
#include "indices/spotted.h"
#include <omp.h>
#include <set>
#include <map>

using std::set;
using std::map;
/*
static inline __attribute__((always_inline))
bool withinVelocityRadius(const Position &position, int baiter, int victim, int64_t curIndex, int64_t windowStartIndex,
                          int tOffset) {
    double distance = computeDistance(position, baiter, victim, curIndex, windowStartIndex);
    double radius = MAX_RUN_SPEED / TICKS_PER_SECOND * tOffset;
    return distance <= radius;
}

BaitersResult queryBaiters(const Position &position, const Kills &kills, const SpottedIndex &spottedIndex) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int64_t> tmpAllyDeathTicks[numThreads];
    vector<int> tmpBaiters[numThreads];
    vector<int> tmpVictims[numThreads];
    vector<int> tmpKillers[numThreads];

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        // assuming first position is less than first kills
        int64_t positionGameStartIndex = position.gameStarts[gameIndex];

        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex = position.getPlayerNameToIndex(gameIndex);

        // for each kill, get the right position index
        // then going back from that position index, check if someone in prior BAIT_WINDOW_SIZE could've gotten to same spot
        for (int64_t killIndex = kills.gameStarts[gameIndex];
             killIndex < kills.gameStarts[gameIndex + 1];
             killIndex++) {
            int64_t positionWindowStartIndex = positionGameStartIndex +
                                               kills.demoTickNumber[killIndex] -
                                               position.demoTickNumber[positionGameStartIndex];
            // ok if kill happens before game start as skipping those position ticks
            if (position.demoTickNumber[positionWindowStartIndex] != kills.demoTickNumber[killIndex] &&
                kills.demoTickNumber[killIndex] >= position.firstRowAfterWarmup[gameIndex]) {
                std::cerr << "bad kill demo tick number " << kills.demoTickNumber[killIndex] <<
                    " not equal position demo tick number " << position.demoTickNumber[positionWindowStartIndex]
                    << " in game " << position.fileNames[position.demoFile[positionWindowStartIndex]] << std::endl;
                continue;
            }

            int victimIndex = playerNameToIndex[kills.victim[killIndex]];
            int killerIndex = playerNameToIndex[kills.killer[killIndex]];

            // tracking who might be a baiter but hasn't been conclusive marked as such yet
            // so don't need to add them multiple times in a window
            set<int> possibleBaiters[2];
            int curReader = 0, curWriter = 1;
            for (int baiterIndex = 0; baiterIndex < NUM_PLAYERS; baiterIndex++) {
                // require baiter to be on same team and not same person
                if (baiterIndex != victimIndex &&
                    position.players[baiterIndex].team[positionWindowStartIndex] ==
                        position.players[victimIndex].team[positionWindowStartIndex]) {
                    // require baiter not to be seen in window and alive whole time
                    bool seenInWindow = false;
                    bool alive = true;
                    for (int64_t positionWindowIndex = positionWindowStartIndex;
                         positionWindowIndex >= position.firstRowAfterWarmup[gameIndex] &&
                         positionWindowIndex > positionWindowStartIndex - BAIT_WINDOW_SIZE;
                         positionWindowIndex--) {
                        seenInWindow |= spottedIndex.visible[baiterIndex][killerIndex][positionWindowIndex];
                        alive &= position.players[baiterIndex].isAlive[positionWindowIndex];
                    }
                    if (!seenInWindow && alive) {
                        possibleBaiters[curReader].insert(baiterIndex);
                    }
                }
            }


            for (int64_t positionWindowIndex = positionWindowStartIndex;
                 positionWindowIndex >= position.firstRowAfterWarmup[gameIndex] &&
                 positionWindowIndex > positionWindowStartIndex - BAIT_WINDOW_SIZE &&
                 !possibleBaiters[curReader].empty();
                 positionWindowIndex--) {
                int tOffset = positionWindowStartIndex - positionWindowIndex;

                for (const auto & playerIndex : possibleBaiters[curReader]) {
                    // baiting if
                    // 1. on same team - checked by possibleBaiters
                    // 2. didn't already bait - checked by possibleBaiters
                    // 3. baiter never visible to killer during window - checked by possibleBaiters
                    // 4. baiter alive whole time - checked by possibleBaiters
                    // 5. baiter could've gotten to spot where victim died
                    if (withinVelocityRadius(position, playerIndex, victimIndex, positionWindowIndex,
                                             positionWindowStartIndex, tOffset)) {
                        tmpIndices[threadNum].push_back(positionWindowIndex);
                        tmpAllyDeathTicks[threadNum].push_back(position.demoTickNumber[positionWindowStartIndex]);
                        tmpBaiters[threadNum].push_back(playerIndex);
                        tmpVictims[threadNum].push_back(victimIndex);
                        tmpKillers[threadNum].push_back(killerIndex);
                    }
                    else {
                        possibleBaiters[curWriter].insert(playerIndex);
                    }
                }

                possibleBaiters[curReader].clear();
                curReader = (curReader + 1) % 2;
                curWriter = (curWriter + 1) % 2;
            }
        }
    }

    BaitersResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.allyDeathTicks.push_back(tmpAllyDeathTicks[i][j]);
            result.baiters.push_back(tmpBaiters[i][j]);
            result.victimsAndKillers.push_back({tmpVictims[i][j], tmpKillers[i][j]});
        }
    }
    return result;
}
*/