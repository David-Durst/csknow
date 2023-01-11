#include "load_data.h"
#include "load_cover.h"
#include <algorithm>
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;

// this is for dense one-to-many relationships (one row in primary column matches many in foreign column)
// dense as have a relationship for every row in primary column (like PAT showing playing position for ever tick)
// since dense, wnat to iterate over times (as will have event for every time)
void buildRangeIndex(const vector<int64_t> &primaryKeyCol, int64_t primarySize, const int64_t * foreignKeyCol,
                          int64_t foreignSize, RangeIndex rangeIndexCol,
                          const string & primaryName, const string & foreignName) {
    for (int64_t primaryIndex = 0, foreignIndex = 0; primaryIndex < primarySize; primaryIndex++) {
        if (foreignIndex >= foreignSize || foreignKeyCol[foreignIndex] > primaryKeyCol[primaryIndex]) {
            rangeIndexCol[primaryIndex].minId = -1;
            rangeIndexCol[primaryIndex].maxId = -1;
        }
        else {
            assert(foreignIndex < foreignSize);
            // sometimes have mistakes where point to 0 as uninitalized, skip entries
            // for example, ticks in between rounds have a round id of -1
            for(; foreignIndex < foreignSize &&
                  foreignKeyCol[foreignIndex] <= 0 &&
                  foreignKeyCol[foreignIndex] < primaryKeyCol[primaryIndex]; foreignIndex++);
            if (foreignIndex >= foreignSize) {
                continue;
            }
            if (foreignKeyCol[foreignIndex] < primaryKeyCol[primaryIndex]) {
                cout << "bad foreign " << foreignIndex  << " val " << foreignKeyCol[foreignIndex]
                     << " for foreign column " << foreignName
                     << " and primary " << primaryIndex << " val " << primaryKeyCol[primaryIndex]
                     << " for primary column " << primaryName << endl;
                assert(foreignKeyCol[foreignIndex] >= primaryKeyCol[primaryIndex]);
            }
            // skip primary key col that have no references to them
            if (foreignKeyCol[foreignIndex] == primaryKeyCol[primaryIndex]) {
                rangeIndexCol[primaryIndex].minId = foreignIndex;
                for (; foreignIndex < foreignSize && foreignKeyCol[foreignIndex] == primaryKeyCol[primaryIndex]; foreignIndex++) ;
                rangeIndexCol[primaryIndex].maxId = foreignIndex - 1;
            }
            else {
                rangeIndexCol[primaryIndex].minId = -1;
                rangeIndexCol[primaryIndex].maxId = -1;
            }
        }
    }
}

// this is for sparse many-to-many relationships (like grenades, one grenade corresponds to many ticks and many grenades at one tick)
// sparse like kills, most rows don't have a kill
// since sparse, want to iterate over the events rather than times when events occur
IntervalIndex buildIntervalIndex(const vector<const int64_t *> &foreignKeyCols, int64_t foreignSize) {
    vector<Interval<int64_t, int64_t>> eventIntervals;
    unordered_map<int64_t, RangeIndexEntry> eventToInterval;
    for (int64_t foreignIndex = 0; foreignIndex < foreignSize; foreignIndex++) {
        // collect all primary key entries that are in range the foreign keys
        int64_t minPrimaryIndex = foreignKeyCols[0][foreignIndex], maxPrimaryIndex = foreignKeyCols[0][foreignIndex];
        for (size_t col = 1; col < foreignKeyCols.size(); col++) {
            minPrimaryIndex = std::min(minPrimaryIndex, foreignKeyCols[col][foreignIndex]);
            maxPrimaryIndex = std::max(maxPrimaryIndex, foreignKeyCols[col][foreignIndex]);
        }
        eventIntervals.push_back({minPrimaryIndex, maxPrimaryIndex, foreignIndex});
        eventToInterval[foreignIndex] = {minPrimaryIndex, maxPrimaryIndex};
    }
    return {IntervalTree{std::move(eventIntervals)}, std::move(eventToInterval)};
}

void buildIndexes(Equipment & equipment [[maybe_unused]], GameTypes & gameTypes [[maybe_unused]], HitGroups & hitGroups [[maybe_unused]], Games & games,
                       Players & players, Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep,
                       WeaponFire & weaponFire, Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed,
                       GrenadeTrajectories & grenadeTrajectories [[maybe_unused]], Plants & plants, Defusals & defusals, Explosions & explosions) {
    cout << "building range indexes" << endl;
    buildRangeIndex(games.id, games.size, rounds.gameId, rounds.size, games.roundsPerGame, "games", "rounds");
    buildRangeIndex(games.id, games.size, players.gameId, players.size, games.playersPerGame, "games", "players");
    buildRangeIndex(rounds.id, rounds.size, ticks.roundId, ticks.size, rounds.ticksPerRound, "rounds", "ticks");
    buildRangeIndex(ticks.id, ticks.size, playerAtTick.tickId, playerAtTick.size, ticks.patPerTick, "ticks", "playerAtTick");
    buildRangeIndex(ticks.id, ticks.size, spotted.tickId, spotted.size, ticks.spottedPerTick, "ticks", "spotted");
    buildRangeIndex(ticks.id, ticks.size, footstep.tickId, footstep.size, ticks.footstepPerTick, "ticks", "footstep");
    // TODO: reenable when golang parser UniqueID2 works so indices are fixed
    //buildRangeIndex(grenades.id, grenades.size, flashed.grenadeId, flashed.size, grenades.flashedPerGrenade, "grenades", "flashed");
    //buildRangeIndex(grenades.id, grenades.size, grenadeTrajectories.grenadeId, grenadeTrajectories.size, grenades.trajectoryPerGrenade, "grenades", "grenadeTrajectories");
    buildRangeIndex(plants.id, plants.size, defusals.plantId, defusals.size, plants.defusalsPerGrenade, "plants", "defusals");
    buildRangeIndex(plants.id, plants.size, explosions.plantId, explosions.size, plants.explosionsPerGrenade, "plants", "explosions");

    cout << "building hashmap indexes" << endl;
    ticks.weaponFirePerTick = buildIntervalIndex({weaponFire.tickId}, weaponFire.size);
    ticks.killsPerTick = buildIntervalIndex({kills.tickId}, kills.size);
    ticks.hurtPerTick = buildIntervalIndex({hurt.tickId}, hurt.size);
    ticks.grenadesPerTick = buildIntervalIndex({grenades.throwTick, grenades.activeTick, grenades.expiredTick, grenades.destroyTick}, grenades.size);
    ticks.grenadesThrowPerTick = buildIntervalIndex({grenades.throwTick}, grenades.size);
    ticks.grenadesActivePerTick = buildIntervalIndex({grenades.activeTick}, grenades.size);
    ticks.grenadesExpiredPerTick = buildIntervalIndex({grenades.expiredTick}, grenades.size);
    ticks.grenadesActivePerTick = buildIntervalIndex({grenades.activeTick}, grenades.size);
    ticks.flashedPerTick = buildIntervalIndex({flashed.tickId}, flashed.size);
    ticks.plantsPerTick = buildIntervalIndex({plants.startTick, plants.endTick}, plants.size);
    ticks.plantsStartPerTick = buildIntervalIndex({plants.startTick}, plants.size);
    ticks.plantsEndPerTick = buildIntervalIndex({plants.endTick}, plants.size);
    ticks.defusalsPerTick = buildIntervalIndex({defusals.startTick, defusals.endTick}, defusals.size);
    ticks.defusalsStartPerTick = buildIntervalIndex({defusals.startTick}, defusals.size);
    ticks.defusalsEndPerTick = buildIntervalIndex({defusals.endTick}, defusals.size);
    ticks.explosionsPerTick = buildIntervalIndex({explosions.tickId}, explosions.size);
}

void buildGridIndex(const vector<int64_t> &primaryKeyCol, const Vec3 * points, GridIndex &index) {
    // get min and max values
    index.minValues = points[0];
    index.maxValues = points[0];
    for (size_t i = 0; i < primaryKeyCol.size(); i++) {
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
    for (size_t idIndex = 0; idIndex < index.sortedIds.size(); idIndex++) {
        const int64_t id = index.sortedIds[idIndex];
        IVec3 newCell = index.getCellCoordinates(points[id]);
        int64_t cellIndex = index.getCellIndex(newCell);
        if (curCell != newCell) {
            curCell = newCell;
            index.minIdIndex[cellIndex] = static_cast<int64_t>(idIndex);
        }
        index.numIds[cellIndex]++;
    }
}

void buildAABBIndex(RangeIndex rangeIndex, int64_t rangeSize, const AABB * aabbCol, AABBIndex aabbIndexCol) {
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

[[maybe_unused]]
void buildCoverIndex(CoverOrigins & origins, CoverEdges & edges) {
    cout << "building cover indexes" << endl;
    buildGridIndex(origins.id, origins.origins, origins.originsGrid);
    buildRangeIndex(origins.id, origins.size, edges.originId, edges.size,
                    origins.coverEdgesPerOrigin, "origins", "edges");
    buildAABBIndex(origins.coverEdgesPerOrigin, origins.size, edges.aabbs, origins.coverEdgeBoundsPerOrigin);
}
