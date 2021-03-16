#include "queries/wallers.h"
#include <omp.h>
#include <set>
#include <map>
#include <math.h>
#include <limits>
using std::set;
using std::map;
#define WALL_WINDOW_SIZE 64
//https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/mathlib/mathlib.h#L301-L303
#ifndef DEG2RAD
#define DEG2RAD( x  )  ( (double)(x) * (double)(M_PI / 180.) )
#endif

const int HEIGHT = 72;
const int WIDTH = 32;
static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    //making box with these coordinates wraps player perfectly
    AABB result;
    result.min = {pos.x - WIDTH / 2, pos.y - WIDTH / 2, pos.z};
    result.max = {pos.x + WIDTH / 2, pos.x + WIDTH / 2, pos.z + HEIGHT};
    return result;
}

// https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp#L901-L914
Vec3 angleVectors(const Vec2 &angles) {
    Vec3 forward;
    double sp, sy, cp, cy;

    sincos( DEG2RAD( angles.x ), &sy, &cy );
    sincos( DEG2RAD( angles.y ), &sp, &cp );

    forward.x = cp*cy;
    forward.y = cp*sy;
    forward.z = -sp;
    return forward;
}

const int EYE_HEIGHT = 64;
static inline __attribute__((always_inline))
Ray getEyeCoordinatesForPlayer(Vec3 pos, Vec2 view) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    // don't negative view.y as it runs negative (it's pitch, not actually y), used in angleVectors
    return Ray({pos.x, pos.y, pos.z + EYE_HEIGHT}, angleVectors({view.x, view.y}));
}

// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/pstd.h#L26-L31
template <typename T>
static inline __attribute__((always_inline))
void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/vecmath.h#L1555
static inline __attribute__((always_inline))
bool intersectP(const AABB & box, const Ray & ray, double tMax = std::numeric_limits<double>::infinity()) {
    double t0 = 0, t1 = tMax;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        double invRayDir = 1 / ray.dir[i];
        double tNear = (box.min[i] - ray.orig[i]) * invRayDir;
        double tFar = (box.max[i] - ray.orig[i]) * invRayDir;
        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar)
           swap(tNear, tFar);
        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * gamma(3);

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1)
            return false;
    }
    return true;
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
    // x angle 0 is looking towards large x positions
    // x angle 180 is looking towards smaller x posotions
    // x angle -90 is looking towards smaller y positions
    // x angle 90 is looking towards larger y positions
    // y angle 90 is looking towards smaller z positions
    // y angle -90 is looking towards larger z positions
//#pragma omp parallel for
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
        map<string, int> playerNameToIndex;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            playerNameToIndex.insert({position.players[i].name[0], i});
        }
        std::cout << "name index size" << playerNameToIndex.size() << std::endl;

        // iterating over each possible window
        for (int64_t windowStartIndex = position.firstRowAfterWarmup[gameIndex];
            windowStartIndex + WALL_WINDOW_SIZE < position.gameStarts[gameIndex+1];
            windowStartIndex++) {
            // for all spotted events on cur tick, update the spotted spottedPerWindow
            while (spottedIndex < spotted.size && spotted.demoFile[spottedIndex] == position.demoFile[windowStartIndex] &&
                    spotted.demoTickNumber[spottedIndex] <= position.demoTickNumber[windowStartIndex]) {
                int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndex]];
                if (playerNameToIndex.size() > 10) {
                    std::cout << "wut" << std::endl;
                }
                for (int i = 0; i < NUM_PLAYERS; i++) {
                    spottedPerWindow[spottedPlayer][i] = spotted.spotters[i].playerSpotter[spottedIndex];
                    if (i == 1) {
                        std::cout <<" spotted demo tick " << spotted.demoTickNumber[spottedIndex] <<
                            " actual demo tick " << position.demoTickNumber[windowStartIndex] <<
                            " spotted player " << spotted.spottedPlayer[spottedIndex] << " is index " << spottedPlayer <<
                            " is brett visible " << spotted.spotters[1].playerSpotter[spottedIndex] <<
                            " set spotted player visible from brett " << spottedPerWindow[spottedPlayer][i] << std::endl;
                    }
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
                if (position.fileNames[gameIndex].compare("auto0-20210221-232115-1880750554-de_dust2-Counter-Strike__Global_Offensive0c007374-749b-11eb-b224-1622baae68c9.dem") == 0 &&
                    position.demoTickNumber[windowStartIndex] == 4577) {//position.players[playerIndex].name[windowStartIndex][0] == 'W') {
                    std::cout << "hi" << std::endl;
                }
                // update the spotted players for this window
                while (spottedIndexInWindow < spotted.size && spotted.demoFile[spottedIndexInWindow] == position.demoFile[windowIndex] &&
                        spotted.demoTickNumber[spottedIndexInWindow] <= position.demoTickNumber[windowIndex]) {
                    int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndexInWindow]];
                    if (playerNameToIndex.size() > 10) {
                        std::cout << "wut" << std::endl;
                    }
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
                    if (position.fileNames[gameIndex].compare("auto0-20210221-232115-1880750554-de_dust2-Counter-Strike__Global_Offensive0c007374-749b-11eb-b224-1622baae68c9.dem") == 0 &&
                        position.demoTickNumber[windowStartIndex] == 4577 && windowIndex == 4591 && playerIndex == 4) {//position.players[playerIndex].name[windowStartIndex][0] == 'W') {
                        std::cout << "eye pos x: " << eyes[playerIndex].orig.x << ", y: " << eyes[playerIndex].orig.y << ", y: " << eyes[playerIndex].orig.z << std::endl;
                        std::cout << "eye yaw: " << position.players[playerIndex].xViewDirection[windowIndex] << ", pivot: " << position.players[playerIndex].yViewDirection[windowIndex] << std::endl;
                        std::cout << "eye view x: " << eyes[playerIndex].dir.x << ", y: " << eyes[playerIndex].dir.y << ", y: " << eyes[playerIndex].dir.z << std::endl;
                    }
                }
                // save for this window if still a suspect -
                // 1. not on same team
                // 2. both alive
                // 3. not visible at any point in window
                // 4. aim locked on
                for (const auto & cv: windowTracking[curReader]) {
                    if (position.players[cv.cheater].team[windowIndex] != position.players[cv.victim].team[windowIndex] &&
                        !spottedInWindow[cv.victim][cv.cheater] &&
                        position.players[cv.cheater].isAlive[windowIndex] && position.players[cv.victim].isAlive[windowIndex] &&
                        intersectP(boxes[cv.victim], eyes[cv.cheater])) {
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
            result.victims.push_back(tmpVictims[i][j]);
        }
    }
    return result;
}
