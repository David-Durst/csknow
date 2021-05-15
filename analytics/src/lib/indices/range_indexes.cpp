#include "load_data.h"
#include <algorithm>
#include <iostream>
#include <assert.h>

template <class T>
void buildRangeIndex(const vector<T> primaryKeyCol, int64_t primarySize, const T * foreignKeyCol,
                          int64_t foreignSize, RangeIndex rangeIndexCol) {
    for (int64_t primaryIndex = 0, foreignIndex = 0; primaryIndex < primarySize; primaryIndex++) {
        // sometimes have mistakes where point to 0 as uninitalized, skip entries
        if(foreignKeyCol[foreignIndex] == 0 &&  primaryKeyCol[primaryIndex] > 0) {
            for (; foreignKeyCol[foreignIndex] == 0; foreignIndex++) ;
        }
        assert(foreignKeyCol[foreignIndex] >= primaryKeyCol[primaryIndex]);
        if (primaryIndex >= foreignSize || foreignKeyCol[foreignIndex] > primaryKeyCol[primaryIndex]) {
            rangeIndexCol[primaryIndex].minId = -1;
            rangeIndexCol[primaryIndex].maxId = -1;
        }

        rangeIndexCol[primaryIndex].minId = foreignIndex;
        for (; foreignKeyCol[foreignIndex] == primaryKeyCol[primaryIndex]; foreignIndex++) ;
        rangeIndexCol[primaryIndex].maxId = foreignIndex - 1;
    }
}

template <class T>
void buildHashmapIndex(const vector<T> primaryKeyCol, int64_t primarySize, const T * foreignKeyStartCol,
                       const vector<T *> foreignKeyEndCols, int64_t foreignSize, RangeIndex rangeIndexCol) {
    int64_t primaryIndex = 0, startForeignIndex = 0, endForeignIndex = 0;
    while (primaryIndex < primarySize) {
        // sometimes have mistakes where point to 0 as uninitalized, skip entries
        if(foreignKeyStartCol[startForeignIndex] == 0 &&  primaryKeyCol[primaryIndex] > 0) {
            for (; foreignKeyStartCol[startForeignIndex] == 0; startForeignIndex++) ;
        }
        assert(foreignKeyStartCol[startForeignIndex] >= primaryKeyCol[primaryIndex]);
        if (primaryIndex >= foreignSize || foreignKeyStartCol[startForeignIndex] > primaryKeyCol[primaryIndex]) {
            rangeIndexCol[primaryIndex].minId = -1;
            rangeIndexCol[primaryIndex].maxId = -1;
        }

        rangeIndexCol[primaryIndex].minId = startForeignIndex;
        for (; foreignKeyStartCol[startForeignIndex] == primaryKeyCol[primaryIndex]; primaryIndex++) {
            for (endForeignIndex = startForeignIndex;
                 foreignKeyStartCol[endForeignIndex] != 0 && foreignKeyStartCol[endForeignIndex] <= primaryKeyCol[primaryIndex];
                 endForeignIndex++) ;
            for (const auto foreignKeyEndCol : foreignKeyEndCols) {
                if (foreignKeyEndCol[endForeignIndex] <= prim)
            }
            rangeIndexCol[primaryIndex].maxId = endForeignIndex;
        }
        for (; foreignKeyStartCol[startForeignIndex] == primaryKeyCol[primaryIndex]; startForeignIndex++) ;
    }
}

void buildRangeIndexes(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games,
                       Players & players, Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted,
                       WeaponFire & weaponFire, Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed,
                       GrenadeTrajectories & grenadeTrajectories, Plants & plants, Defusals & defusals, Explosions & explosions) {
    buildRangeIndex(games.id, games.size, rounds.gameId, rounds.size, games.roundsPerGame);
    buildRangeIndex(rounds.id, rounds.size, ticks.roundId, ticks.size, rounds.ticksPerRound);
    buildRangeIndex(ticks.id, ticks.size, playerAtTick.tickId, playerAtTick.size, ticks.playersPerTick);
    buildRangeIndex(ticks.id, ticks.size, spotted.tickId, spotted.size, ticks.spottedPerTick);
    buildRangeIndex(ticks.id, ticks.size, kills.tickId, kills.size, ticks.killsPerTick);
    buildRangeIndex(ticks.id, ticks.size, hurt.tickId, hurt.size, ticks.hurtPerTick);
    buildOverlappingRangeIndex(ticks.id, )
    buildPointRangeIndex(ticks.id, ticks.size, grenades.throwTick, grenades.size, ticks.grenadesThrowPerTick);
    buildPointRangeIndex(ticks.id, ticks.size, grenades.activeTick, grenades.size, ticks.grenadesActivePerTick);
    buildPointRangeIndex(ticks.id, ticks.size, grenades.expiredTick, grenades.size, ticks.grenadesExpiredPerTick);
    buildPointRangeIndex(ticks.id, ticks.size, grenades.destroyTick, grenades.size, ticks.grenadesDestroyedPerTick);
    buildPointRangeIndex(ticks.id, ticks.size, flashed.tickId, flashed.size, ticks.flashedPerTick);
    buildRangeIndex(ticks.id, ticks.size, plants.st, plants.size, ticks.plantsPerTick);
}
