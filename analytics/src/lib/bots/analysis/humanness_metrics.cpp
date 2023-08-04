//
// Created by durst on 8/3/23.
//

#include "bots/analysis/humanness_metrics.h"

namespace csknow::humanness_metrics {
    AreaBits getVisibleAreasByTeam(const VisPoints & visPoints, vector<AreaId> areaIds) {
        AreaBits result;
        for (const auto & areaId : areaIds) {
            result |= visPoints.getVisibilityRelativeToSrc(areaId);
        }
        return result;
    }

    HumannessMetrics::HumannessMetrics(const csknow::feature_store::TeamFeatureStoreResult &teamFeatureStoreResult,
                                       const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                       const Hurt & hurt, const WeaponFire & weaponFire,
                                       const ReachableResult & reachable, const VisPoints & visPoints) {
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                if (teamFeatureStoreResult.nonDecimatedValidRetakeTicks[tickIndex]) {
                    set<int64_t> shootersThisTick;
                    for (const auto & [_0, _1, weaponFireIndex] :
                        ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        shootersThisTick.insert(weaponFire.shooter[weaponFireIndex]);
                    }

                    set<int64_t> victimsThisTick;
                    map<int64_t, int64_t> attackerForVictimsThisTick;
                    for (const auto & [_0, _1, hurtIndex] :
                            ticks.hurtPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        if (isDemoEquipmentAGun(hurt.weapon[hurtIndex])) {
                            victimsThisTick.insert(hurt.victim[hurtIndex]);
                            attackerForVictimsThisTick[hurt.victim[hurtIndex]] = hurt.attacker[hurtIndex];
                        }
                    }

                    // skip tick if no key events
                    if (shootersThisTick.empty() && victimsThisTick.empty()) {
                        continue;
                    }

                    map<int64_t, int64_t> playerToAreaIndex;
                    map<int64_t, int64_t> playerToAreaId;
                    for (int i = 0; i < csknow::feature_store::max_enemies; i++) {
                        if (teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex] != INVALID_ID) {
                            playerToAreaIndex[teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex]] =
                                    teamFeatureStoreResult.nonDecimatedCTData[i].areaIndex[tickIndex];
                            playerToAreaId[teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex]] =
                                    teamFeatureStoreResult.nonDecimatedCTData[i].areaId[tickIndex];
                        }
                        if (teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex] != INVALID_ID) {
                            playerToAreaIndex[teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex]] =
                                    teamFeatureStoreResult.nonDecimatedTData[i].areaIndex[tickIndex];
                            playerToAreaId[teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex]] =
                                    teamFeatureStoreResult.nonDecimatedTData[i].areaId[tickIndex];
                        }
                    }

                    map<int64_t, int16_t> playerToTeam;
                    vector<AreaId> ctAreaIds, tAreaIds;
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        playerToTeam[playerAtTick.playerId[patIndex]] = playerAtTick.team[patIndex];
                        if (playerAtTick.isAlive[patIndex] && playerToAreaId.count(playerAtTick.playerId[patIndex])) {
                            if (playerAtTick.team[patIndex] == ENGINE_TEAM_CT) {
                                ctAreaIds.push_back(playerToAreaId[playerAtTick.playerId[patIndex]]);
                            }
                            if (playerAtTick.team[patIndex] == ENGINE_TEAM_T) {
                                tAreaIds.push_back(playerToAreaId[playerAtTick.playerId[patIndex]]);
                            }
                        }
                    }
                    AreaBits coverForT = getVisibleAreasByTeam(visPoints, ctAreaIds),
                        coverForCT = getVisibleAreasByTeam(visPoints, tAreaIds);
                    coverForT.flip();
                    coverForCT.flip();


                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        int64_t playerId = playerAtTick.playerId[patIndex];
                        int16_t teamId = playerAtTick.team[patIndex];

                        // compute velocity metrics
                        float playerVelocity = static_cast<float>(computeMagnitude(Vec3{
                            playerAtTick.velX[tickIndex], playerAtTick.velY[tickIndex], playerAtTick.velZ[tickIndex]}));
                        if (shootersThisTick.count(playerId) > 0) {
                            velocityWhenFiring.push_back(playerVelocity);
                        }
                        if (victimsThisTick.count(playerId) > 0) {
                            velocityWhenShot.push_back(playerVelocity);
                        }

                        if (shootersThisTick.count(playerId) > 0 || victimsThisTick.count(playerId) > 0) {
                            // compute distance to teammate/enemy/attacker metrics
                            float nearestTeammateDistance = std::numeric_limits<float>::max();
                            float nearestEnemyDistance = std::numeric_limits<float>::max();
                            float attackerForVictimDistance = std::numeric_limits<float>::max();

                            for (const auto & [otherPlayerId, otherTeamId] : playerToTeam) {
                                float otherPlayerDistance = static_cast<float>(
                                        reachable.getDistance(playerToAreaIndex[playerId], playerToAreaIndex[otherPlayerId]));

                                if (attackerForVictimsThisTick.count(playerId) > 0 &&
                                    attackerForVictimsThisTick[playerId] == otherPlayerId) {
                                    attackerForVictimDistance = otherPlayerDistance;
                                }

                                if (teamId == otherTeamId) {
                                    nearestTeammateDistance = std::min(nearestTeammateDistance, otherPlayerDistance);
                                }
                                else {
                                    nearestEnemyDistance = std::min(nearestEnemyDistance, otherPlayerDistance);
                                }
                            }

                            if (shootersThisTick.count(playerId)) {
                                distanceToNearestTeammateWhenFiring.push_back(nearestTeammateDistance);
                                distanceToNearestEnemyWhenFiring.push_back(nearestEnemyDistance);
                            }
                            if (victimsThisTick.count(playerId)) {
                                distanceToNearestTeammateWhenShot.push_back(nearestTeammateDistance);
                                distanceToAttackerWhenShot.push_back(attackerForVictimDistance);
                            }

                            // compute distance to cover
                            float minDistanceToCover = std::numeric_limits<float>::max();
                            const AreaBits & cover = teamId == ENGINE_TEAM_CT ? coverForCT : coverForT;
                            for (size_t i = 0; i < visPoints.getAreaVisPoints().size(); i++) {
                                if (cover[i]) {
                                    minDistanceToCover = std::min(minDistanceToCover, static_cast<float>(
                                            reachable.getDistance(playerToAreaIndex[playerId], i)));
                                }
                            }
                            if (shootersThisTick.count(playerId)) {
                                distanceToCoverWhenFiring.push_back(minDistanceToCover);
                            }
                            if (victimsThisTick.count(playerId)) {
                                distanceToCoverWhenShot.push_back(minDistanceToCover);
                            }
                        }
                    }
                }
            }
        }
    }
}