//
// Created by durst on 7/11/23.
//

#ifndef CSKNOW_FEATURE_STORE_TEAM_EXTRACTOR_H
#define CSKNOW_FEATURE_STORE_TEAM_EXTRACTOR_H

#include "queries/moments/engagement.h"
#include "queries/nearest_nav_cell.h"

namespace csknow::feature_store {

    TeamFeatureStoreResult featureStoreTeamExtraction(const string & navPath, const nav_mesh::nav_file & navFile,
                                                      const std::vector<csknow::orders::QueryOrder> & orders,
                                                      const VisPoints & visPoints,
                                                      const DistanceToPlacesResult & distanceToPlaces,
                                                      const nearest_nav_cell::NearestNavCell & nearestNavCell,
                                                      const Players & players, const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                                      const WeaponFire & weaponFire, const Hurt & hurt,
                                                      const Plants & plants, const Defusals & defusals,
                                                      const csknow::key_retake_events::KeyRetakeEvents & keyRetakeEvents,
                                                      bool requireBothTeamsAlive);
}

#endif //CSKNOW_FEATURE_STORE_TEAM_EXTRACTOR_H
