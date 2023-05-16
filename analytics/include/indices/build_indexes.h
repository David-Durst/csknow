//
// Created by durst on 9/12/22.
//

#ifndef CSKNOW_BUILD_INDEXES_H
#define CSKNOW_BUILD_INDEXES_H

#include "load_data.h"

void buildRangeIndex(const vector<int64_t> &primaryKeyCol, int64_t primarySize, const vector<int64_t> &foreignKeyCol,
                     int64_t foreignSize, RangeIndex rangeIndexCol, string primaryName, string foreignName);

IntervalIndex buildIntervalIndex(const vector<std::reference_wrapper<const vector<int64_t>>> &foreignKeyCols, int64_t foreignSize);

void buildIndexes(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games,
                  Players & players, Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep,
                  WeaponFire & weaponFire, Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed,
                  GrenadeTrajectories & grenadeTrajectories, Plants & plants, Defusals & defusals, Explosions & explosions,
                  Say & say);

#endif //CSKNOW_BUILD_INDEXES_H
