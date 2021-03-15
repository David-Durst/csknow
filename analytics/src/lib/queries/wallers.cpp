#include "queries/wallers.h"
#include <omp.h>
#include <set>
#include <map>
#include <math.h>
using std::set;
using std::map;
#define WALL_WINDOW_SIZE 32
//https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/public/mathlib/mathlib.h#L301-L303
#ifndef M_PI
#define M_PI		3.14159265358979323846	// matches value in gcc v2 math.h
#endif
#define M_PI_F		((float)(M_PI))	// Shouldn't collide with anything.
#ifndef DEG2RAD
#define DEG2RAD( x  )  ( (float)(x) * (float)(M_PI_F / 180.f) )
#endif

const int HEIGHT = 72;
const int WIDTH = 32;
static inline __attribute__((always_inline))
AABB getAABBForPlayer(Vec3 pos) {
    //https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Mapper%27s_Reference
    // looks like coordinates are center at feet - tested using getpos_exact and box commands from
    //https://old.reddit.com/r/csmapmakers/comments/58ch3f/useful_console_commands_for_map_making_csgo/
    AABB result;
    result.bounds[0] = {pos.x - WIDTH / 2, pos.y - WIDTH / 2, pos.z};
    result.bounds[1] = {pos.x + WIDTH / 2, pos.x + WIDTH / 2, pos.z + HEIGHT};
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

// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
static inline __attribute__((always_inline))
bool rayAABBIntersection(Ray ray, AABB box) {
    double tmin, tmax, tymin, tymax, tzmin, tzmax;


    tmin = (box.bounds[ray.sign[0]].x - ray.orig.x) * ray.invdir.x;
    tmax = (box.bounds[1-ray.sign[0]].x - ray.orig.x) * ray.invdir.x;
    tymin = (box.bounds[ray.sign[1]].y - ray.orig.y) * ray.invdir.y;
    tymax = (box.bounds[1-ray.sign[1]].y - ray.orig.y) * ray.invdir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (box.bounds[ray.sign[2]].z - ray.orig.z) * ray.invdir.z;
    tzmax = (box.bounds[1-ray.sign[2]].z - ray.orig.z) * ray.invdir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return true; }

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
#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int hitsInFile = 0;
        int threadNum = omp_get_thread_num();
        AABB boxes[NUM_PLAYERS];
        Ray eyes[NUM_PLAYERS];
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
            while (spotted.demoTickNumber[spottedIndex] <= position.demoTickNumber[windowStartIndex] &&
                    spottedIndex < spotted.size) {
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
                while (spotted.demoTickNumber[spottedIndexInWindow] <= position.demoTickNumber[windowIndex] &&
                        spottedIndexInWindow < spotted.size) {
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
                    eyes[playerIndex] = getEyeCoordinatesForPlayer(
                        {position.players[playerIndex].xPosition[windowIndex],
                         position.players[playerIndex].yPosition[windowIndex],
                         position.players[playerIndex].zPosition[windowIndex]},
                        {position.players[playerIndex].xViewDirection[windowIndex],
                         position.players[playerIndex].yViewDirection[windowIndex]}
                    );
                }
                // save for this window if still a suspect -
                // 1. aim locked on
                // 2. not on same team
                // 3. both alive
                // 4. not visible at any point in window
                for (const auto & cv: windowTracking[curReader]) {
                    if (position.players[cv.cheater].team[windowIndex] != position.players[cv.victim].team[windowIndex] &&
                        !spottedInWindow[cv.victim][cv.cheater] &&
                        position.players[cv.cheater].isAlive[windowIndex] && position.players[cv.victim].isAlive[windowIndex] &&
                        rayAABBIntersection(eyes[cv.cheater], boxes[cv.victim])) {
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
                hitsInFile++;
            }
        }
        if (position.fileNames[gameIndex].compare("auto0-20210221-232115-1880750554-de_dust2-Counter-Strike__Global_Offensive0c007374-749b-11eb-b224-1622baae68c9.dem") == 0) {
            std::cout << "tracked file waller events: " << hitsInFile << std::endl;
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
