#include "queries/baiters.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
using std::set;
using std::map;
#define BAIT_WINDOW_SIZE 64

static inline __attribute__((always_inline))
bool withinVelocityRadius(const Position & position, int source, int target, int64_t positionIndex, int tOffset) {
    double xDistance = position.players[source].xPosition[positionIndex] - position.players[target].xPosition[positionIndex];
    double yDistance = position.players[source].yPosition[positionIndex] - position.players[target].yPosition[positionIndex];
    double zDistance = position.players[source].zPosition[positionIndex] - position.players[target].zPosition[positionIndex];
    double distance = sqrt(xDistance*xDistance + yDistance*yDistance + zDistance*zDistance);
    double radius = MAX_RUN_SPEED / TICKS_PER_SECOND * tOffset;
    return distance <= radius;
}

BaitersResult queryBaiters(const Position & position, const Kills & kills) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int64_t> tmpMostRecentPossibleHelp[numThreads];
    vector<int> tmpBaiters[numThreads];
    vector<int> tmpVictims[numThreads];

#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        // assuming first position is less than first kills
        int64_t positionGameStartIndex = position.gameStarts[gameIndex];

        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            playerNameToIndex.insert({position.players[i].name[position.firstRowAfterWarmup[gameIndex]], i});

            // for each kill, get the right position index
            // then going back from that position index, check if someone in prior BAIT_WINDOW_SIZE could've gotten to same spot
            for (int64_t killIndex = kills.gameStarts[gameIndex];
                 killIndex < kills.gameStarts[gameIndex + 1];
                 killIndex++) {
                int64_t positionWindowStartIndex = positionGameStartIndex +
                                                   kills.demoTickNumber[killIndex] -
                                                   position.demoTickNumber[positionGameStartIndex];
                if (position.demoTickNumber[positionWindowStartIndex] != kills.demoTickNumber[killIndex]) {
                    std::cerr << "bad position computing for baiters" << std::endl;
                    continue;
                }

                // tracking who already baited so don't need to add them multiple times in a window
                set<SourceAndTarget> alreadyBaited;
                int victimIndex = playerNameToIndex[kills.victim[killIndex]];

                for (int64_t positionWindowIndex = positionWindowStartIndex;
                     positionWindowStartIndex >= position.firstRowAfterWarmup[gameIndex] &&
                     positionWindowIndex >= positionWindowStartIndex - BAIT_WINDOW_SIZE;
                     positionWindowIndex--) {
                    int tOffset = positionWindowStartIndex - positionWindowIndex;

                    for (int playerIndex = 0; playerIndex < NUM_PLAYERS; playerIndex++) {
                        if (alreadyBaited.find({playerIndex, victimIndex}) == alreadyBaited.end() &&
                            withinVelocityRadius(position, playerIndex, victimIndex, positionWindowIndex, tOffset)) {
                            alreadyBaited.insert({playerIndex, victimIndex});
                            tmpIndices[threadNum].push_back(positionWindowStartIndex);
                            tmpMostRecentPossibleHelp[threadNum].push_back(positionWindowIndex);
                            tmpBaiters[threadNum].push_back(playerIndex);
                            tmpVictims[threadNum].push_back(victimIndex);
                        }
                    }

                }
            }
        }
    }

    BaitersResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.mostRecentPossibleHelp.push_back(tmpMostRecentPossibleHelp[i][j]);
            result.baiters.push_back(tmpBaiters[i][j]);
            result.victims.push_back(tmpVictims[i][j]);
        }
    }
    return result;
}
