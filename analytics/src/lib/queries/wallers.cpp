#include "queries/wallers.h"
#include <omp.h>
#include <set>
#include <map>
using std::set;
using std::map;
#define WALL_WINDOW_SIZE 32

const int HEIGHT = 72;
const int WIDTH = 32;
static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    return {{pos.x - WIDTH / 2, pos.y - WIDTH / 2, pos.z},
            {pos.x + WIDTH / 2, pos.x + WIDTH / 2, pos.z + HEIGHT}};
}

const int EYE_HEIGHT = 64;
static inline __attribute__((always_inline))
Vec5 getEyeCoordinatesForPlayer(Vec5 pos) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    return {{pos.pos.x, pos.pos.y, pos.pos.z + EYE_HEIGHT}, {pos.view.x, pos.view.y}};
}

static inline __attribute__((always_inline))
bool rayAABBIntersection(Vec5 ray, AABB box) {
    return false;
}

struct CheaterAndVictim {
    int cheater, victim;
    bool operator <(const CheaterAndVictim& cv) const {
        return cheater < cv.cheater || ((cheater == cv.cheater) && victim < cv.victim);
    }
};

WallersResult queryWallers(const Position & position, const Spotted & spotted) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int> tmpCheaters[numThreads];
    vector<int> tmpVictims[numThreads];

    // helpful test props - https://steamcommunity.com/sharedfiles/filedetails/?id=1458241029
    // models - https://developer.valvesoftware.com/wiki/Anatomy_of_a_Model
    // I can treat all coordinates as top or bottom, doesn't matter
    // x angle 0 is looking towards large x positions
    // x angle 180 is looking towards smaller x posotions
    // x angle -90 is looking towards smaller y positions
    // x angle 90 is looking towards smaller y positions
    // y angle 90 is looking towards smaller z positions
    // y angle -90 is looking towards larger z positions
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        AABB boxes[NUM_PLAYERS];
        Vec5 eyes[NUM_PLAYERS];
        int64_t spottedIndex = spotted.gameStarts[gameIndex];
        // spottedPerWindow[i][j] - is player i visible to player j
        // initially, no one can see anyone else
        bool spottedPerWindow[NUM_PLAYERS][NUM_PLAYERS];
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int j = 0; j < NUM_PLAYERS; j++) {
                spottedPerWindow[i][j] = false;
            }
        }
        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            playerNameToIndex.insert({position.players[i].name[0], i});
        }

        // iterating over each possible window
        for (int64_t windowStartIndex = position.gameStarts[gameIndex];
            windowStartIndex + WALL_WINDOW_SIZE < position.gameStarts[gameIndex+1];
            windowStartIndex++) {
            // for all spotted events on cur tick, update the spotted spottedPerWindow
            while (spotted.demoTickNumber[spottedIndex] <= position.demoTickNumber[windowStartIndex]) {
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
                while (spotted.demoTickNumber[spottedIndexInWindow] <= position.demoTickNumber[windowIndex]) {
                    int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndexInWindow]];
                    for (int i = 0; i < NUM_PLAYERS; i++) {
                        spottedPerWindow[spottedPlayer][i] |= spotted.spotters[i].playerSpotter[spottedIndexInWindow];
                    }
                    spottedIndexInWindow++;
                }

                // update only needed players
                for (const auto & playerIndex : neededPlayers[curReader]) {
                    boxes[playerIndex] = getAABBForPlayer({position.players[playerIndex].xPosition[windowIndex],
                                                           position.players[playerIndex].yPosition[windowIndex],
                                                           position.players[playerIndex].zPosition[windowIndex]});
                    eyes[playerIndex] = getEyeCoordinatesForPlayer({
                        {position.players[playerIndex].xPosition[windowIndex],
                         position.players[playerIndex].yPosition[windowIndex],
                         position.players[playerIndex].zPosition[windowIndex]},
                        {position.players[playerIndex].xViewDirection[windowIndex],
                         position.players[playerIndex].yViewDirection[windowIndex]}
                    });
                }
                // save for this window if still a suspect -
                // 1. aim locked on
                // 2. not on same team
                // 3. not visible at any point in window
                for (const auto & cv: windowTracking[curReader]) {
                    if (rayAABBIntersection(eyes[cv.cheater], boxes[cv.victim]) &&
                        position.players[cv.cheater].team[windowIndex] != position.players[cv.victim].team[windowIndex] &&
                        !spottedInWindow[cv.victim][cv.cheater]) {
                        windowTracking[curWriter].insert({cv.cheater, cv.victim});
                        neededPlayers[curWriter].insert({cv.cheater});
                        neededPlayers[curWriter].insert({cv.victim});
                    }
                }
                // finish double buffering for this frame
                windowTracking[curReader].clear();
                neededPlayers[curReader].clear();
                curReader = (curReader + 1) % 2;
                curWriter = (curWriter + 1) % 2;
            }

            // save all found cheaters in this window
            for (const auto & cv : windowTracking[curReader]) {
                tmpIndices[threadNum].push_back(windowStartIndex);
                tmpCheaters[threadNum].push_back(cv.cheater);
                tmpVictims[threadNum].push_back(cv.victim);
            }
        }
    }

    WallersResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.cheaters.push_back(tmpCheaters[i][j]);
            result.victims.push_back(tmpVictims[i][j]);
        }
    }
    return result;
}
