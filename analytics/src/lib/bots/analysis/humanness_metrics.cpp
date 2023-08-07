//
// Created by durst on 8/3/23.
//

#include "bots/analysis/humanness_metrics.h"
#include "file_helpers.h"
#include <atomic>

namespace csknow::humanness_metrics {
    AreaBits getVisibleAreasByTeam(const VisPoints & visPoints, vector<AreaId> areaIds) {
        AreaBits result;
        for (const auto & areaId : areaIds) {
            result |= visPoints.getVisibilityRelativeToSrc(areaId);
        }
        return result;
    }

    HumannessMetrics::HumannessMetrics(const csknow::feature_store::TeamFeatureStoreResult &teamFeatureStoreResult,
                                       const Games & games, const Rounds & rounds, const Players & players, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick, const Hurt & hurt, const WeaponFire & weaponFire,
                                       const ReachableResult & reachable, const VisPoints & visPoints) {
        std::atomic<int64_t> roundsProcessed = 0;
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            // record round metrics only if round has at least one valid tick
            bool roundHasValidTick = false, ctWon = false;
            size_t numValidTicksCT = 0, numValidTicksT = 0;
            size_t numMaxSpeedTicksCT = 0, numMaxSpeedTicksT = 0, numStillTicksCT = 0, numStillTicksT = 0;

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                if (teamFeatureStoreResult.nonDecimatedValidRetakeTicks[tickIndex]) {
                    // on first valid tick, set winner
                    if (!roundHasValidTick) {
                        roundHasValidTick = true;
                        ctWon = rounds.winner[roundIndex] == ENGINE_TEAM_CT;
                    }

                    // compute key events
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

                    // get positions in area id form and enemy visible
                    map<int64_t, int64_t> playerToAreaIndex;
                    map<int64_t, int64_t> playerToAreaId;
                    set<int64_t> playerCanSeeEnemy;
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
                        if (teamFeatureStoreResult.nonDecimatedCTData[i].enemyVisible[tickIndex]) {
                            playerCanSeeEnemy
                        }
                    }

                    // get who's alive and their
                    map<int64_t, int16_t> playerToTeam;
                    map<int64_t, bool> playerToAlive;
                    vector<AreaId> ctAreaIds, tAreaIds;
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        playerToTeam[playerAtTick.playerId[patIndex]] = playerAtTick.team[patIndex];
                        playerToAlive[playerAtTick.playerId[patIndex]] = playerAtTick.isAlive[patIndex];
                        if (playerAtTick.isAlive[patIndex] && playerToAreaId.count(playerAtTick.playerId[patIndex])) {
                            if (playerAtTick.team[patIndex] == ENGINE_TEAM_CT) {
                                ctAreaIds.push_back(playerToAreaId[playerAtTick.playerId[patIndex]]);
                            }
                            if (playerAtTick.team[patIndex] == ENGINE_TEAM_T) {
                                tAreaIds.push_back(playerToAreaId[playerAtTick.playerId[patIndex]]);
                            }
                        }
                    }

                    // copmute cover
                    AreaBits coverForT = getVisibleAreasByTeam(visPoints, ctAreaIds),
                            coverForCT = getVisibleAreasByTeam(visPoints, tAreaIds);
                    coverForT.flip();
                    coverForCT.flip();

                    // record ticks for entire round metrics
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        if (playerAtTick.isAlive[patIndex]) {
                            int64_t playerId = playerAtTick.playerId[patIndex];
                            int16_t teamId = playerAtTick.team[patIndex];

                            // compute speed metrics
                            float curUnscaledSpeed = static_cast<float>(computeMagnitude(Vec2{
                                    playerAtTick.velX[patIndex], playerAtTick.velY[patIndex]
                            }));

                            csknow::weapon_speed::StatureOptions statureOption =
                                    csknow::weapon_speed::StatureOptions::Standing;
                            if (playerAtTick.duckingKeyPressed[patIndex]) {
                                statureOption = csknow::weapon_speed::StatureOptions::Ducking;
                            }
                            else if (playerAtTick.isWalking[patIndex]) {
                                statureOption = csknow::weapon_speed::StatureOptions::Walking;
                            }
                            double maxSpeed = csknow::weapon_speed::engineWeaponIdToMaxSpeed(
                                    demoEquipmentTypeToEngineWeaponId(playerAtTick.activeWeapon[patIndex]),
                                    statureOption, playerAtTick.isScoped[patIndex]);
                            // anything over max speed is due to jump strafing, just cap at regular max
                            curUnscaledSpeed = std::max(static_cast<float>(maxSpeed), curUnscaledSpeed);
                            float curScaledSpeed = static_cast<float>(curUnscaledSpeed / maxSpeed);

                            double maxRunSpeed = csknow::weapon_speed::engineWeaponIdToMaxSpeed(
                                    demoEquipmentTypeToEngineWeaponId(playerAtTick.activeWeapon[patIndex]),
                                    csknow::weapon_speed::StatureOptions::Standing, playerAtTick.isScoped[patIndex]);
                            float curWeaponOnlyScaledSpeed = static_cast<float>(curUnscaledSpeed / maxRunSpeed);

                            bool runningAtMaxSpeed = curScaledSpeed >= csknow::weapon_speed::speed_threshold;
                            bool standingStill = curScaledSpeed <= (1 - csknow::weapon_speed::speed_threshold);

                            if (teamId == ENGINE_TEAM_CT) {
                                numValidTicksCT++;
                                if (runningAtMaxSpeed) {
                                    numMaxSpeedTicksCT++;
                                }
                                if (standingStill) {
                                    numStillTicksCT++;
                                }
                            }
                            if (teamId == ENGINE_TEAM_T) {
                                numValidTicksT++;
                                if (runningAtMaxSpeed) {
                                    numMaxSpeedTicksT++;
                                }
                                if (standingStill) {
                                    numStillTicksT++;
                                }
                            }

                            if (curUnscaledSpeed > 400.) {
                                std::cout << "demo file " << games.demoFile[rounds.gameId[roundIndex]]
                                          << ", game tick number " << ticks.gameTickNumber[tickIndex]
                                          << ", player "  << players.name[players.idOffset + playerAtTick.playerId[patIndex]]
                                          << ", cur speed " << curUnscaledSpeed
                                          << ", vel x " << playerAtTick.velX[patIndex] << ", vel y " << playerAtTick.velY[patIndex]
                                          << ", weapon " << demoEquipmentTypeToString(playerAtTick.activeWeapon[patIndex])
                                          << ", weapon id " << playerAtTick.activeWeapon[patIndex]
                                          << ", player alive " << playerAtTick.isAlive[patIndex] << std::endl;
                            }
                            unscaledSpeed.push_back(curUnscaledSpeed);
                            scaledSpeed.push_back(curScaledSpeed);
                            weaponOnlyScaledSpeed.push_back(curWeaponOnlyScaledSpeed);
                            if (shootersThisTick.count(playerId) > 0) {
                                unscaledSpeedWhenFiring.push_back(curUnscaledSpeed);
                                scaledSpeedWhenFiring.push_back(curScaledSpeed);
                                weaponOnlyScaledSpeedWhenFiring.push_back(curWeaponOnlyScaledSpeed);
                            }
                            if (victimsThisTick.count(playerId) > 0) {
                                unscaledSpeedWhenShot.push_back(curUnscaledSpeed);
                                scaledSpeedWhenShot.push_back(curScaledSpeed);
                                weaponOnlyScaledSpeedWhenShot.push_back(curWeaponOnlyScaledSpeed);
                            }


                            // compute distance to teammate/enemy/attacker metrics
                            float nearestTeammateDistance = std::numeric_limits<float>::max();
                            float nearestEnemyDistance = std::numeric_limits<float>::max();
                            float attackerForVictimDistance = std::numeric_limits<float>::max();

                            for (const auto &[otherPlayerId, otherTeamId]: playerToTeam) {
                                if (playerId == otherPlayerId) {
                                    continue;
                                }
                                if (!playerToAlive[otherPlayerId] && !victimsThisTick.count(otherPlayerId) &&
                                    !shootersThisTick.count(otherPlayerId)) {
                                    continue;
                                }

                                float otherPlayerDistance = static_cast<float>(
                                        reachable.getDistance(playerToAreaIndex[playerId],
                                                              playerToAreaIndex[otherPlayerId]));

                                if (attackerForVictimsThisTick.count(playerId) > 0 &&
                                    attackerForVictimsThisTick[playerId] == otherPlayerId) {
                                    attackerForVictimDistance = otherPlayerDistance;
                                }

                                if (teamId == otherTeamId) {
                                    nearestTeammateDistance = std::min(nearestTeammateDistance,
                                                                       otherPlayerDistance);
                                } else {
                                    nearestEnemyDistance = std::min(nearestEnemyDistance, otherPlayerDistance);
                                }
                            }

                            if (nearestTeammateDistance != std::numeric_limits<float>::max()) {
                                distanceToNearestTeammate.push_back(nearestTeammateDistance);
                            }
                            if (shootersThisTick.count(playerId)) {
                                // filter out times when no teammate exists
                                if (nearestTeammateDistance != std::numeric_limits<float>::max()) {
                                    distanceToNearestTeammateWhenFiring.push_back(nearestTeammateDistance);
                                }
                                distanceToNearestEnemyWhenFiring.push_back(nearestEnemyDistance);
                            }

                            distanceToNearestEnemy.push_back(nearestEnemyDistance);
                            if (victimsThisTick.count(playerId)) {
                                if (nearestTeammateDistance != std::numeric_limits<float>::max()) {
                                    distanceToNearestTeammateWhenShot.push_back(nearestTeammateDistance);
                                }
                                distanceToNearestEnemyWhenShot.push_back(nearestEnemyDistance);
                                distanceToAttackerWhenShot.push_back(attackerForVictimDistance);
                            }


                            // compute distance to cover metrics
                            float minDistanceToCover = std::numeric_limits<float>::max();
                            const AreaBits &cover = teamId == ENGINE_TEAM_CT ? coverForCT : coverForT;
                            for (size_t i = 0; i < visPoints.getAreaVisPoints().size(); i++) {
                                if (cover[i]) {
                                    float newDistanceToCover =
                                            static_cast<float>( reachable.getDistance(playerToAreaIndex[playerId], i));
                                    // filter out invalid distances
                                    if (newDistanceToCover >= 0.) {
                                        minDistanceToCover = std::min(minDistanceToCover, newDistanceToCover);
                                    }
                                }
                            }
                            distanceToCover.push_back(minDistanceToCover);
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

            // finish round metrics
            if (roundHasValidTick) {
                pctTimeMaxSpeedCT.push_back(static_cast<float>(numMaxSpeedTicksCT) / static_cast<float>(numValidTicksCT));
                pctTimeStillCT.push_back(static_cast<float>(numStillTicksCT) / static_cast<float>(numValidTicksCT));
                pctTimeMaxSpeedT.push_back(static_cast<float>(numMaxSpeedTicksT) / static_cast<float>(numValidTicksT));
                pctTimeStillT.push_back(static_cast<float>(numStillTicksT) / static_cast<float>(numValidTicksT));
                ctWins.push_back(ctWon);
            }

            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }

        // different columns have different sizes, so size not that valid here
        size = static_cast<int64_t>(unscaledSpeedWhenFiring.size());
    }

    void HumannessMetrics::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;

        file.createDataSet("/data/unscaled speed", unscaledSpeed, hdf5FlatCreateProps);
        file.createDataSet("/data/unscaled speed when firing", unscaledSpeedWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/unscaled speed when shot", unscaledSpeedWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/scaled speed", scaledSpeed, hdf5FlatCreateProps);
        file.createDataSet("/data/scaled speed when firing", scaledSpeedWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/scaled speed when shot", scaledSpeedWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/weapon only scaled speed", weaponOnlyScaledSpeed, hdf5FlatCreateProps);
        file.createDataSet("/data/weapon only scaled speed when firing", weaponOnlyScaledSpeedWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/weapon only scaled speed when shot", weaponOnlyScaledSpeedWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/distance to nearest teammate", distanceToNearestTeammate, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to nearest teammate when firing", distanceToNearestTeammateWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to nearest teammate when shot", distanceToNearestTeammateWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/distance to nearest enemy", distanceToNearestEnemy, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to nearest enemy when firing", distanceToNearestEnemyWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to nearest enemy when shot", distanceToNearestEnemyWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/distance to attacker when shot", distanceToAttackerWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/distance to cover", distanceToCover, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to cover when firing", distanceToCoverWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to cover when shot", distanceToCoverWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/pct time max speed ct", pctTimeMaxSpeedCT, hdf5FlatCreateProps);
        file.createDataSet("/data/pct time max speed t", pctTimeMaxSpeedT, hdf5FlatCreateProps);
        file.createDataSet("/data/pct time still ct", pctTimeStillCT, hdf5FlatCreateProps);
        file.createDataSet("/data/pct time still t", pctTimeStillT, hdf5FlatCreateProps);
        file.createDataSet("/data/ct wins", ctWins, hdf5FlatCreateProps);
    }
}