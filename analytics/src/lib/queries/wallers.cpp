#include "queries/wallers.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
using std::set;
using std::map;


struct CheaterAndVictim {
    int cheater, victim;
    bool operator <(const CheaterAndVictim& cv) const {
        return cheater < cv.cheater || ((cheater == cv.cheater) && victim < cv.victim);
    }
};
/*
WallersResult queryWallers(const Position & position, const Spotted & spotted) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int> tmpCheaters[numThreads];
    vector<int> tmpVictims[numThreads];

    // helpful test props - https://steamcommunity.com/sharedfiles/filedetails/?id=1458241029
    // models - https://developer.valvesoftware.com/wiki/Anatomy_of_a_Model
    // x angle 0 is looking towards large x positions
    // x angle 180 is looking towards smaller x posotions
    // x angle -90 is looking towards smaller y positions
    // x angle 90 is looking towards larger y positions
    // y angle 90 is looking towards smaller z positions
    // y angle -90 is looking towards larger z positions
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        AABB boxes[NUM_PLAYERS];
        Ray eyes[NUM_PLAYERS];
        int64_t spottedIndex = spotted.gameStarts[gameIndex];
        // don't repeat cheating events within 32 ticks, decrease duplicate events
        int64_t ticksSinceLastCheating[NUM_PLAYERS][NUM_PLAYERS];
        // spottedPerWindow[i][j] - is player i visible to player j
        // initially, no one can see anyone else
        bool spottedPerWindow[NUM_PLAYERS][NUM_PLAYERS];
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int j = 0; j < NUM_PLAYERS; j++) {
                spottedPerWindow[i][j] = false;
                ticksSinceLastCheating[NUM_PLAYERS][NUM_PLAYERS] = 1000;
            }
        }
        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex = position.getPlayerNameToIndex(gameIndex);

        // iterating over each possible window
        for (int64_t windowStartIndex = position.firstRowAfterWarmup[gameIndex];
            windowStartIndex + WALL_WINDOW_SIZE < position.gameStarts[gameIndex+1];
            windowStartIndex++) {
            // for all spotted events on cur tick, update the spotted spottedPerWindow
            while (spottedIndex < spotted.size && spotted.demoFile[spottedIndex] == position.demoFile[windowStartIndex] &&
                    spotted.demoTickNumber[spottedIndex] <= position.demoTickNumber[windowStartIndex]) {
                // skip invalid rows
                if (spotted.skipRows.find(spottedIndex) != spotted.skipRows.end()) {
                    spottedIndex++;
                    continue;
                }
                int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndex]];
                for (int i = 0; i < NUM_PLAYERS; i++) {
                    spottedPerWindow[spottedPlayer][i] = spotted.spotters[i].playerSpotter[spottedIndex];
                }
                spottedIndex++;
            }

            // if see 1 time in window, not suspicious, so track if seen ever inside window
            int64_t spottedIndexInWindow = spottedIndex;
            bool spottedInWindow[NUM_PLAYERS][NUM_PLAYERS];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                for (int j = 0; j < NUM_PLAYERS; j++) {
                    spottedInWindow[i][j] = spottedPerWindow[i][j];
                }
            }

            // track who I currently need
            // double buffer so no need to remove untracked, just won't add them after each frame
            set<CheaterAndVictim> windowTracking[2];
            set<int> neededPlayers[2];
            int curReader = 0, curWriter = 1;
            // start tracking all players for each window - everyone is a suspect for trackign everyone else until not
            for (int cheaterIndex = 0; cheaterIndex < NUM_PLAYERS; cheaterIndex++) {
                neededPlayers[curReader].insert(cheaterIndex);
                for (int victimIndex = 0; victimIndex < NUM_PLAYERS; victimIndex++) {
                    windowTracking[curReader].insert({cheaterIndex, victimIndex});
                }
            }

            // for each window, as long as any cheaters possibly left in that window
            for (int64_t windowIndex = windowStartIndex; windowIndex < windowStartIndex + WALL_WINDOW_SIZE && !neededPlayers[curReader].empty();
                windowIndex++) {
                // update the spotted players for this window
                while (spottedIndexInWindow < spotted.size && spotted.demoFile[spottedIndexInWindow] == position.demoFile[windowIndex] &&
                        spotted.demoTickNumber[spottedIndexInWindow] <= position.demoTickNumber[windowIndex]) {
                    // skip invalid rows
                    if (spotted.skipRows.find(spottedIndex) != spotted.skipRows.end()) {
                        spottedIndexInWindow++;
                        continue;
                    }
                    int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndexInWindow]];
                    for (int i = 0; i < NUM_PLAYERS; i++) {
                        spottedInWindow[spottedPlayer][i] |= spotted.spotters[i].playerSpotter[spottedIndexInWindow];
                    }
                    spottedIndexInWindow++;
                }

                // update only needed players
                for (const auto & playerIndex : neededPlayers[curReader]) {
                    boxes[playerIndex] = getAABBForPlayer({position.players[playerIndex].xPosition[windowIndex],
                                                           position.players[playerIndex].yPosition[windowIndex],
                                                           position.players[playerIndex].zPosition[windowIndex]});
                    eyes[playerIndex] = getEyeCoordinatesForPlayer(
                        {position.players[playerIndex].xPosition[windowIndex],
                         position.players[playerIndex].yPosition[windowIndex],
                         position.players[playerIndex].zPosition[windowIndex]},
                        {position.players[playerIndex].xViewDirection[windowIndex],
                         position.players[playerIndex].yViewDirection[windowIndex]}
                    );
                }
                // save for this window if still a suspect -
                // 1. not on same team
                // 2. both alive
                // 3. not visible at any point in window
                // 4. aim locked on
                for (const auto & cv: windowTracking[curReader]) {
                    double t0, t1;
                    if (position.players[cv.cheater].team[windowIndex] != position.players[cv.victim].team[windowIndex] &&
                        !spottedInWindow[cv.victim][cv.cheater] &&
                        position.players[cv.cheater].isAlive[windowIndex] && position.players[cv.victim].isAlive[windowIndex] &&
                        intersectP(boxes[cv.victim], eyes[cv.cheater], t0, t1)) {
                        windowTracking[curWriter].insert({cv.cheater, cv.victim});
                        neededPlayers[curWriter].insert(cv.cheater);
                        neededPlayers[curWriter].insert(cv.victim);
                    }
                }
                // finish double buffering for this frame
                windowTracking[curReader].clear();
                neededPlayers[curReader].clear();
                curReader = (curReader + 1) % 2;
                curWriter = (curWriter + 1) % 2;
            }

            for (int i = 0; i < NUM_PLAYERS; i++) {
                for (int j = 0; j < NUM_PLAYERS; j++) {
                    ticksSinceLastCheating[i][j]++;
                }
            }

            // save all found cheaters in this window who weren't found in last window's worth of ticks
            for (const auto & cv : windowTracking[curReader]) {
                if (ticksSinceLastCheating[cv.cheater][cv.victim] >= WALL_WINDOW_SIZE) {
                    tmpIndices[threadNum].push_back(windowStartIndex);
                    tmpCheaters[threadNum].push_back(cv.cheater);
                    tmpVictims[threadNum].push_back(cv.victim);
                    ticksSinceLastCheating[cv.cheater][cv.victim] = 0;
                }
            }
        }
    }

    WallersResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.cheaters.push_back(tmpCheaters[i][j]);
            result.victims.push_back({tmpVictims[i][j]});
        }
    }
    return result;
}
*/