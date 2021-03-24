#include "queries/netcode.h"
#include "geometry.h"
#include <omp.h>
#include <set>
#include <map>
using std::set;
using std::map;
// TODO: replace this with a reset time per weapon
#define RESET_TIME 32

NetcodeResult queryNetcode(const Position & position, const WeaponFire & weaponFire,
                           const PlayerHurt & playerHurt) {
    int64_t numGames = position.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpIndices[numThreads];
    vector<int> tmpShooters[numThreads];
    vector<int> tmpLuckys[numThreads];

//#pragma omp parallel for
    for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
        int threadNum = omp_get_thread_num();
        AABB boxes[NUM_PLAYERS];
        Ray eyes[NUM_PLAYERS];
        int64_t fireIndex = weaponFire.gameStarts[gameIndex];
        int64_t hurtIndex = playerHurt.gameStarts[gameIndex];
        int32_t lastMoveOrFireTick[NUM_PLAYERS];
        for (int i = 0; i < NUM_PLAYERS; i++) {
            lastMoveOrFireTick[i] = position.demoTickNumber[position.firstRowAfterWarmup[gameIndex]];
        }
        // since spotted tracks names for spotted player, need to map that to the player index
        map<string, int> playerNameToIndex = position.getPlayerNameToIndex(gameIndex);
        // given a team's id, get all enemies
        map<int, vector<int>> enemiesForTeam = position.getEnemiesForTeam(gameIndex);

        // iterating over each possible window
        // TODO: DO I WANT TO ITERATE OVER WEAPON FIRE OR POSITION?
        // WEAPON FIRE IS BIGGER THAN KILLS, SO DIFFERENT TRADE-OFF
        for (int64_t positionIndex = position.firstRowAfterWarmup[gameIndex];
             positionIndex < position.gameStarts[gameIndex+1];
             positionIndex++) {
            // first update for movement on this tick
            // ok to look 1 back since skipping warmup, so always at least 1 in past
            for (int i = 0; i < NUM_PLAYERS; i++) {
                if (abs(position.players[i].xPosition[positionIndex] - position.players[i].xPosition[positionIndex-1]) > 0.05 ||
                    abs(position.players[i].yPosition[positionIndex] - position.players[i].yPosition[positionIndex-1]) > 0.05 ||
                    abs(position.players[i].zPosition[positionIndex] - position.players[i].yPosition[positionIndex-1]) > 0.05) {
                    lastMoveOrFireTick[i] = positionIndex;
                }
            }
            set<SourceAndTarget> hurtEntriesThisTick;
            while (hurtIndex < playerHurt.size && playerHurt.demoFile[hurtIndex] == position.demoFile[positionIndex] &&
                   playerHurt.demoTickNumber[hurtIndex] <= position.demoTickNumber[positionIndex]) {
                if (playerHurt.demoTickNumber[hurtIndex] < position.demoTickNumber[positionIndex]) {
                    std::cerr << "player hurt somehow get behind position" << std::endl;
                }
                else {
                    hurtEntriesThisTick.insert({ playerNameToIndex[playerHurt.attacker[hurtIndex]],
                                                 playerNameToIndex[playerHurt.victimName[hurtIndex]]});
                }
                hurtIndex++;
            }

            // then check for all firing events on this tick
            // for all spotted events on cur tick, update the spotted spottedPerWindow
            while (fireIndex < weaponFire.size && weaponFire.demoFile[fireIndex] == position.demoFile[positionIndex] &&
                   weaponFire.demoTickNumber[fireIndex] <= position.demoTickNumber[positionIndex]) {
                int firingPlayer = playerNameToIndex[weaponFire.shooter[fireIndex]];

                if (weaponFire.demoTickNumber[fireIndex] < position.demoTickNumber[positionIndex]) {
                    std::cerr << "weapon fire somehow get behind position" << std::endl;
                }


                // check for each player that is alive and didn't get shot by attacker this tick if should've hit them
                for (const auto &enemyIndex : enemiesForTeam[position.players[firingPlayer].team[positionIndex]]) {
                    if (position.players[enemyIndex].isAlive &&
                        hurtEntriesThisTick.find({firingPlayer, enemyIndex}) == hurtEntriesThisTick.end()) {
                        AABB victimBox = getAABBForPlayer({position.players[firingPlayer].xPosition[positionIndex],
                                                           position.players[firingPlayer].yPosition[positionIndex],
                                                           position.players[firingPlayer].zPosition[positionIndex]},
                                                          0.1);
                        Ray attackerEyes = getEyeCoordinatesForPlayer(
                                {position.players[enemyIndex].xPosition[positionIndex],
                                 position.players[enemyIndex].yPosition[positionIndex],
                                 position.players[enemyIndex].zPosition[positionIndex]},
                                {position.players[enemyIndex].xViewDirection[positionIndex],
                                 position.players[enemyIndex].yViewDirection[positionIndex]});
                        double t0, t1;
                        if (intersectP(victimBox, attackerEyes, t0, t1)) {
                            tmpIndices[threadNum].push_back(positionIndex);
                            tmpShooters[threadNum].push_back(firingPlayer);
                            tmpLuckys[threadNum].push_back(enemyIndex);
                        }
                    }
                }
                fireIndex++;
            }
        }
    }

    NetcodeResult result;
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpIndices[i].size(); j++) {
            result.positionIndex.push_back(tmpIndices[i][j]);
            result.shooters.push_back(tmpShooters[i][j]);
            result.luckys.push_back({tmpLuckys[i][j]});
        }
    }
    return result;
}
