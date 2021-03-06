#include "load_data.h"
#include <algorithm>
#include <iostream>
#include <assert.h>
using std::cout;
using std::endl;

void buildRangeIndex(const vector<int64_t> primaryKeyCol, int64_t primarySize, const int64_t * foreignKeyCol,
                          int64_t foreignSize, RangeIndex rangeIndexCol) {
    for (int64_t primaryIndex = 0, foreignIndex = 0; primaryIndex < primarySize; primaryIndex++) {
        if (foreignIndex >= foreignSize || foreignKeyCol[foreignIndex] > primaryKeyCol[primaryIndex]) {
            rangeIndexCol[primaryIndex].minId = -1;
            rangeIndexCol[primaryIndex].maxId = -1;
        }
        else {
            // sometimes have mistakes where point to 0 as uninitalized, skip entries
            for(; foreignKeyCol[foreignIndex] <= 0 && foreignKeyCol[foreignIndex] < primaryKeyCol[primaryIndex]; foreignIndex++);
            if (foreignKeyCol[foreignIndex] < primaryKeyCol[primaryIndex]) {
                cout << "bad foreign " << foreignIndex  << " val " << foreignKeyCol[foreignIndex]
                     << " and primary " << primaryIndex << " val " << primaryKeyCol[primaryIndex] << endl;
                assert(foreignKeyCol[foreignIndex] >= primaryKeyCol[primaryIndex]);
            }
            assert(foreignIndex < foreignSize);
            rangeIndexCol[primaryIndex].minId = foreignIndex;
            for (; foreignIndex < foreignSize && foreignKeyCol[foreignIndex] == primaryKeyCol[primaryIndex]; foreignIndex++) ;
            rangeIndexCol[primaryIndex].maxId = foreignIndex - 1;
        }
    }
}

void buildHashmapIndex(const vector<int64_t *> foreignKeyCols, int64_t foreignSize, HashmapIndex hashIndexCol) {
    for (int64_t foreignIndex = 0; foreignIndex < foreignSize; foreignIndex++) {
        // collect all primary key entries that are in range the foreign keys
        int64_t minPrimaryIndex = foreignKeyCols[0][foreignIndex], maxPrimaryIndex = foreignKeyCols[0][foreignIndex];
        for (int col = 1; col < foreignKeyCols.size(); col++) {
            minPrimaryIndex = std::min(minPrimaryIndex, foreignKeyCols[col][foreignIndex]);
            maxPrimaryIndex = std::max(maxPrimaryIndex, foreignKeyCols[col][foreignIndex]);
        }
        // insert into all key cols in range
        for (int64_t primaryIndex = minPrimaryIndex; primaryIndex <= maxPrimaryIndex; primaryIndex++) {
            hashIndexCol[primaryIndex].push_back(foreignIndex);
        }
    }
}

void buildIndexes(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games,
                       Players & players, Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted,
                       WeaponFire & weaponFire, Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed,
                       GrenadeTrajectories & grenadeTrajectories, Plants & plants, Defusals & defusals, Explosions & explosions) {
    cout << "building range indexes" << endl;
    buildRangeIndex(games.id, games.size, rounds.gameId, rounds.size, games.roundsPerGame);
    buildRangeIndex(games.id, games.size, players.gameId, players.size, games.playersPerGame);
    buildRangeIndex(rounds.id, rounds.size, ticks.roundId, ticks.size, rounds.ticksPerRound);
    buildRangeIndex(ticks.id, ticks.size, playerAtTick.tickId, playerAtTick.size, ticks.patPerTick);
    buildRangeIndex(ticks.id, ticks.size, spotted.tickId, spotted.size, ticks.spottedPerTick);
    buildRangeIndex(grenades.id, grenades.size, flashed.grenadeId, flashed.size, grenades.flashedPerGrenade);
    buildRangeIndex(grenades.id, grenades.size, grenadeTrajectories.grenadeId, grenadeTrajectories.size, grenades.trajectoryPerGrenade);
    buildRangeIndex(plants.id, plants.size, defusals.plantId, defusals.size, plants.defusalsPerGrenade);
    buildRangeIndex(plants.id, plants.size, explosions.plantId, explosions.size, plants.explosionsPerGrenade);

    cout << "building hashmap indexes" << endl;
    buildHashmapIndex({weaponFire.tickId}, weaponFire.size, ticks.weaponFirePerTick);
    buildHashmapIndex({kills.tickId}, kills.size, ticks.killsPerTick);
    buildHashmapIndex({hurt.tickId}, hurt.size, ticks.hurtPerTick);
    buildHashmapIndex({grenades.throwTick, grenades.activeTick, grenades.expiredTick, grenades.destroyTick}, grenades.size, ticks.grenadesPerTick);
    buildHashmapIndex({grenades.throwTick}, grenades.size, ticks.grenadesThrowPerTick);
    buildHashmapIndex({grenades.activeTick}, grenades.size, ticks.grenadesActivePerTick);
    buildHashmapIndex({grenades.expiredTick}, grenades.size, ticks.grenadesExpiredPerTick);
    buildHashmapIndex({grenades.activeTick}, grenades.size, ticks.grenadesActivePerTick);
    buildHashmapIndex({flashed.tickId}, flashed.size, ticks.flashedPerTick);
    buildHashmapIndex({plants.startTick, plants.endTick}, plants.size, ticks.plantsPerTick);
    buildHashmapIndex({plants.startTick}, plants.size, ticks.plantsStartPerTick);
    buildHashmapIndex({plants.endTick}, plants.size, ticks.plantsEndPerTick);
    buildHashmapIndex({defusals.startTick, defusals.endTick}, defusals.size, ticks.defusalsPerTick);
    buildHashmapIndex({defusals.startTick}, defusals.size, ticks.defusalsStartPerTick);
    buildHashmapIndex({defusals.endTick}, defusals.size, ticks.defusalsEndPerTick);
    buildHashmapIndex({explosions.tickId}, explosions.size, ticks.explosionsPerTick);
}
