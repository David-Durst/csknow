#include "load_data.h"
#include <algorithm>
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;

// this is for dense one-to-many relationships (one row in primary column matches many in foreign column)
// dense as have a relationship for every row in primary column (like PAT showing playing position for ever tick)
// since dense, wnat to iterate over times (as will have event for every time)
void buildRangeIndex(const vector<int64_t> &primaryKeyCol, int64_t primarySize, const vector<int64_t> &foreignKeyCol,
                          int64_t foreignSize, RangeIndex & rangeIndexCol,
                          const string & primaryName, const string & foreignName) {
    rangeIndexCol.resize(primarySize);
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
IntervalIndex buildIntervalIndex(const vector<std::reference_wrapper<const vector<int64_t>>> &foreignKeyCols, int64_t foreignSize) {
    vector<Interval<int64_t, int64_t>> eventIntervals;
    unordered_map<int64_t, RangeIndexEntry> eventToInterval;
    for (int64_t foreignIndex = 0; foreignIndex < foreignSize; foreignIndex++) {
        // collect all primary key entries that are in range the foreign keys
        int64_t minPrimaryIndex = std::numeric_limits<int64_t>::max(),
            maxPrimaryIndex = std::numeric_limits<int64_t>::max() * -1;
        // don't save events that lack a valid index, otherwise will save wildly large min/small max, creating
        // huge interval to search and thus making index slow
        bool haveValidIndex = false;
        for (size_t col = 0; col < foreignKeyCols.size(); col++) {
            if (foreignKeyCols[col].get()[foreignIndex] > INVALID_ID) {
                haveValidIndex = true;
                minPrimaryIndex = std::min(minPrimaryIndex, foreignKeyCols[col].get()[foreignIndex]);
                maxPrimaryIndex = std::max(maxPrimaryIndex, foreignKeyCols[col].get()[foreignIndex]);
            }
        }
        if (haveValidIndex) {
            eventIntervals.push_back({minPrimaryIndex, maxPrimaryIndex, foreignIndex});
            eventToInterval[foreignIndex] = {minPrimaryIndex, maxPrimaryIndex};
        }
    }
    return {IntervalTree{std::move(eventIntervals)}, std::move(eventToInterval)};
}

void buildIndexes(Equipment & equipment [[maybe_unused]], GameTypes & gameTypes [[maybe_unused]], HitGroups & hitGroups [[maybe_unused]], Games & games,
                       Players & players, Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep,
                       WeaponFire & weaponFire, Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed,
                       GrenadeTrajectories & grenadeTrajectories [[maybe_unused]], Plants & plants, Defusals & defusals, Explosions & explosions,
                       Say & say) {
    cout << "building range indexes" << endl;
    buildRangeIndex(games.id, games.size, rounds.gameId, rounds.size, games.roundsPerGame, "games", "rounds");
    buildRangeIndex(games.id, games.size, players.gameId, players.size, games.playersPerGame, "games", "players");
    buildRangeIndex(rounds.id, rounds.size, ticks.roundId, ticks.size, rounds.ticksPerRound, "rounds", "ticks");
    buildRangeIndex(ticks.id, ticks.size, playerAtTick.tickId, playerAtTick.size, ticks.patPerTick, "ticks", "playerAtTick");
    buildRangeIndex(ticks.id, ticks.size, spotted.tickId, spotted.size, ticks.spottedPerTick, "ticks", "spotted");
    buildRangeIndex(ticks.id, ticks.size, footstep.tickId, footstep.size, ticks.footstepPerTick, "ticks", "footstep");
    // TODO: reenable when golang parser UniqueID2 works so indices are fixed
    //buildRangeIndex(grenades.id, grenades.size, flashed.grenadeId, flashed.size, grenades.flashedPerGrenade, "grenades", "flashed");
    buildRangeIndex(plants.id, plants.size, defusals.plantId, defusals.size, plants.defusalsPerPlant, "plants", "defusals");
    buildRangeIndex(plants.id, plants.size, explosions.plantId, explosions.size, plants.explosionsPerPlant, "plants", "explosions");

    cout << "building hashmap indexes" << endl;
    ticks.weaponFirePerTick = buildIntervalIndex({weaponFire.tickId}, weaponFire.size);
    ticks.killsPerTick = buildIntervalIndex({kills.tickId}, kills.size);
    ticks.hurtPerTick = buildIntervalIndex({hurt.tickId}, hurt.size);
    ticks.grenadesPerTick = buildIntervalIndex({grenades.throwTick, grenades.activeTick, grenades.expiredTick, grenades.destroyTick}, grenades.size);
    ticks.grenadesThrowPerTick = buildIntervalIndex({grenades.throwTick}, grenades.size);
    ticks.grenadesActivePerTick = buildIntervalIndex({grenades.activeTick}, grenades.size);
    ticks.grenadesExpiredPerTick = buildIntervalIndex({grenades.expiredTick}, grenades.size);
    ticks.grenadesDestroyedPerTick = buildIntervalIndex({grenades.destroyTick}, grenades.size);
    ticks.flashedPerTick = buildIntervalIndex({flashed.tickId}, flashed.size);
    ticks.plantsPerTick = buildIntervalIndex({plants.startTick, plants.endTick}, plants.size);
    ticks.plantsStartPerTick = buildIntervalIndex({plants.startTick}, plants.size);
    ticks.plantsEndPerTick = buildIntervalIndex({plants.endTick}, plants.size);
    ticks.defusalsPerTick = buildIntervalIndex({defusals.startTick, defusals.endTick}, defusals.size);
    ticks.defusalsStartPerTick = buildIntervalIndex({defusals.startTick}, defusals.size);
    ticks.defusalsEndPerTick = buildIntervalIndex({defusals.endTick}, defusals.size);
    ticks.explosionsPerTick = buildIntervalIndex({explosions.tickId}, explosions.size);
    ticks.sayPerTick = buildIntervalIndex({say.tickId}, say.size);
    grenades.trajectoryPerGrenade = buildIntervalIndex({grenadeTrajectories.grenadeId}, grenadeTrajectories.size);
}