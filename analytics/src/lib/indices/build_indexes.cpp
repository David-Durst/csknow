#include "load_data.h"
#include "load_cover.h"
#include <algorithm>
#include <iostream>
#include <assert.h>
using std::cout;
using std::endl;

void buildRangeIndex(const vector<int64_t> &primaryKeyCol, int64_t primarySize, const int64_t * foreignKeyCol,
                          int64_t foreignSize, RangeIndex rangeIndexCol,
                          string primaryName, string foreignName) {
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
                     << " for foreign column " << foreignName
                     << " and primary " << primaryIndex << " val " << primaryKeyCol[primaryIndex]
                     << " for primary column " << primaryName << endl;
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
                       Players & players, Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep,
                       WeaponFire & weaponFire, Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed,
                       GrenadeTrajectories & grenadeTrajectories, Plants & plants, Defusals & defusals, Explosions & explosions) {
    cout << "building range indexes" << endl;
    buildRangeIndex(games.id, games.size, rounds.gameId, rounds.size, games.roundsPerGame, "games", "rounds");
    buildRangeIndex(games.id, games.size, players.gameId, players.size, games.playersPerGame, "games", "players");
    buildRangeIndex(rounds.id, rounds.size, ticks.roundId, ticks.size, rounds.ticksPerRound, "rounds", "ticks");
    buildRangeIndex(ticks.id, ticks.size, playerAtTick.tickId, playerAtTick.size, ticks.patPerTick, "ticks", "playerAtTick");
    buildRangeIndex(ticks.id, ticks.size, spotted.tickId, spotted.size, ticks.spottedPerTick, "ticks", "spotted");
    buildRangeIndex(ticks.id, ticks.size, footstep.tickId, footstep.size, ticks.footstepPerTick, "ticks", "footstep");
    buildRangeIndex(grenades.id, grenades.size, flashed.grenadeId, flashed.size, grenades.flashedPerGrenade, "grenades", "flashed");
    buildRangeIndex(grenades.id, grenades.size, grenadeTrajectories.grenadeId, grenadeTrajectories.size, grenades.trajectoryPerGrenade, "grenades", "grenadeTrajectories");
    buildRangeIndex(plants.id, plants.size, defusals.plantId, defusals.size, plants.defusalsPerGrenade, "plants", "defusals");
    buildRangeIndex(plants.id, plants.size, explosions.plantId, explosions.size, plants.explosionsPerGrenade, "plants", "explosions");

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

void buildGridIndex(const vector<int64_t> &primaryKeyCol, const Vec3 * points, GridIndex &index) {
    // get min and max values
    index.minValues = points[0];
    index.maxValues = points[0];
    for (int64_t i = 0; i < primaryKeyCol.size(); i++) {
        index.minValues = min(points[i], index.minValues);
        index.maxValues = max(points[i], index.maxValues);
    }

    // compute number of cells
    index.numCells = index.getCellCoordinates(index.maxValues);
    index.numCells.x += 1;
    index.numCells.y += 1;
    index.numCells.z += 1;
    size_t totalCells = index.numCells.x * index.numCells.y * index.numCells.z;
    index.minIdIndex.resize(totalCells, -1);
    index.numIds.resize(totalCells, 0);

    // sort ids by coordinates
    index.sortedIds = primaryKeyCol;
    GridComparator comparator(index);
    std::sort(index.sortedIds.begin(), index.sortedIds.end(), comparator);

    // get ranges within each cell
    IVec3 curCell{-1, -1, -1};
    for (int64_t idIndex = 0; idIndex < index.sortedIds.size(); idIndex++) {
        const int64_t id = index.sortedIds[idIndex];
        IVec3 newCell = index.getCellCoordinates(points[id]);
        int64_t cellIndex = index.getCellIndex(newCell);
        if (curCell != newCell) {
            curCell = newCell;
            index.minIdIndex[cellIndex] = idIndex;
        }
        index.numIds[cellIndex]++;
    }
}

void buildAABBIndex(const RangeIndex rangeIndex, int64_t rangeSize, const AABB * aabbCol, AABBIndex aabbIndexCol) {
    for (int64_t rangeId = 0; rangeId < rangeSize; rangeId++) {
        bool firstAABB = true;
        for (int64_t aabbId = rangeIndex[rangeId].minId; aabbId != -1 && aabbId <= rangeIndex[rangeId].maxId;
            aabbId++) {
            AABB aabb = aabbCol[aabbId];
            if (firstAABB) {
                firstAABB = false;
                aabbIndexCol[rangeId] = aabb;
            }
            else {
                aabbIndexCol[rangeId].min = min(aabbIndexCol[rangeId].min, aabb.min);
                aabbIndexCol[rangeId].max = max(aabbIndexCol[rangeId].max, aabb.max);
            }
        }
    }
}

void buildCoverIndex(CoverOrigins & origins, CoverEdges & edges) {
    cout << "building cover indexes" << endl;
    buildGridIndex(origins.id, origins.origins, origins.originsGrid);
    buildRangeIndex(origins.id, origins.size, edges.originId, edges.size,
                    origins.coverEdgesPerOrigin, "origins", "edges");
    buildAABBIndex(origins.coverEdgesPerOrigin, origins.size, edges.aabbs, origins.coverEdgeBoundsPerOrigin);
}
