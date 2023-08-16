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
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);

            // record round metrics only if round has at least one valid tick
            bool roundHasValidTick = false, ctWon = false;
            size_t numValidTicksCT = 0, numValidTicksT = 0;
            size_t numMaxSpeedTicksCT = 0, numMaxSpeedTicksT = 0, numStillTicksCT = 0, numStillTicksT = 0;

            map<TeamId, map<int64_t, int64_t>> teamToPlayerToLastTimeFiring, teamToPlayerToLastTimeShot;
            teamToPlayerToLastTimeFiring[ENGINE_TEAM_CT] = {};
            teamToPlayerToLastTimeShot[ENGINE_TEAM_CT] = {};
            teamToPlayerToLastTimeFiring[ENGINE_TEAM_T] = {};
            teamToPlayerToLastTimeShot[ENGINE_TEAM_T] = {};

            // keep player to team across ticks because need it on frame when players die. Dead players aren't
            // recorded by team feature store. OFC victim on frame when die, and can shoot on frame when dying
            map<int64_t, TeamId> playerToTeamId;

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

                    // get positions in area id form, enemy visible, and team
                    map<int64_t, int64_t> playerToAreaIndex;
                    map<int64_t, int64_t> playerToAreaId;
                    set<int64_t> playerCanSeeEnemyNoFOV, playerCanSeeEnemyFOV;
                    for (int i = 0; i < csknow::feature_store::max_enemies; i++) {
                        if (teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex] != INVALID_ID) {
                            int64_t playerId = teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex];
                            playerToAreaIndex[playerId] =
                                    teamFeatureStoreResult.nonDecimatedCTData[i].areaIndex[tickIndex];
                            playerToAreaId[playerId] =
                                    teamFeatureStoreResult.nonDecimatedCTData[i].areaId[tickIndex];
                            playerToTeamId[playerId] = ENGINE_TEAM_CT;
                        }
                        if (teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex] != INVALID_ID) {
                            int64_t playerId = teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex];
                            playerToAreaIndex[playerId] =
                                    teamFeatureStoreResult.nonDecimatedTData[i].areaIndex[tickIndex];
                            playerToAreaId[playerId] =
                                    teamFeatureStoreResult.nonDecimatedTData[i].areaId[tickIndex];
                            playerToTeamId[playerId] = ENGINE_TEAM_T;
                        }
                        if (teamFeatureStoreResult.nonDecimatedCTData[i].noFOVEnemyVisible[tickIndex]) {
                            playerCanSeeEnemyNoFOV.insert(teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex]);
                        }
                        if (teamFeatureStoreResult.nonDecimatedCTData[i].fovEnemyVisible[tickIndex]) {
                            playerCanSeeEnemyFOV.insert(teamFeatureStoreResult.nonDecimatedCTData[i].playerId[tickIndex]);
                        }
                        if (teamFeatureStoreResult.nonDecimatedTData[i].noFOVEnemyVisible[tickIndex]) {
                            playerCanSeeEnemyNoFOV.insert(teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex]);
                        }
                        if (teamFeatureStoreResult.nonDecimatedTData[i].fovEnemyVisible[tickIndex]) {
                            playerCanSeeEnemyFOV.insert(teamFeatureStoreResult.nonDecimatedTData[i].playerId[tickIndex]);
                        }
                    }

                    // start time events based on firing/shoot -> teammate seeing enemy
                    for (const auto & shooter : shootersThisTick) {
                        if (!playerToTeamId.count(shooter)) {
                            std::cout << "shooter " << players.name[players.idOffset + shooter] << " without team, game tick id " << ticks.gameTickNumber[tickIndex] << std::endl;
                        }
                        TeamId teamId = playerToTeamId[shooter];
                        // don't add if already started event from prior shot that hasn't finished
                        if (!teamToPlayerToLastTimeFiring[teamId].count(shooter)) {
                            teamToPlayerToLastTimeFiring[teamId][shooter] = tickIndex;
                        }
                    }

                    for (const auto & victim : victimsThisTick) {
                        if (!playerToTeamId.count(victim)) {
                            std::cout << "victim " << players.name[players.idOffset + victim] << " without team, game tick id " << ticks.gameTickNumber[tickIndex] << std::endl;
                        }
                        TeamId teamId = playerToTeamId[victim];
                        // don't add if already started event from prior shot that hasn't finished
                        if (!teamToPlayerToLastTimeShot[teamId].count(victim)) {
                            teamToPlayerToLastTimeShot[teamId][victim] = tickIndex;
                        }
                    }

                    // finish time events based on firing/shoot -> teammate seeing enemy
                    // finish after start so can handle 0 tick events where fire/shot on same tick
                    // won't close events where shot but never have teammate see enemey because those are
                    // just end of round when last players are alive, not baits
                    for (const auto & playerSeeingEnemy : playerCanSeeEnemyFOV) {
                        TeamId teamId = playerToTeamId[playerSeeingEnemy];
                        vector<int64_t> firingTeammatesToClear;
                        for (const auto & [firingTeammateId, startTickIndex] : teamToPlayerToLastTimeFiring[teamId]) {
                            // don't count if same person sees and shoots enemy
                            if (firingTeammateId != playerSeeingEnemy) {
                                timeFromFiringToTeammateSeeingEnemyFOV.push_back(static_cast<float>(
                                        secondsBetweenTicks(tickRates, startTickIndex, tickIndex)));
                                firingTeammatesToClear.push_back(firingTeammateId);
                            }
                        }
                        for (const auto & firingTeammateId : firingTeammatesToClear) {
                            teamToPlayerToLastTimeFiring[teamId].erase(firingTeammateId);
                        }

                        vector<int64_t> shotTeammatesToClear;
                        for (const auto & [shotTeammateId, startTickIndex] : teamToPlayerToLastTimeShot[teamId]) {
                            // don't count if same person sees and shoots enemy
                            if (shotTeammateId != playerSeeingEnemy) {
                                timeFromShotToTeammateSeeingEnemyFOV.push_back(static_cast<float>(
                                        secondsBetweenTicks(tickRates, startTickIndex, tickIndex)));
                                shotTeammatesToClear.push_back(shotTeammateId);
                            }
                        }
                        for (const auto & shotTeammateId : shotTeammatesToClear) {
                            teamToPlayerToLastTimeShot[teamId].erase(shotTeammateId);
                        }
                    }

                    // get who's alive and their area ids
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

                    // compute metrics that require player at tick data (positions/velocities)
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
                            curUnscaledSpeed = std::min(static_cast<float>(maxSpeed), curUnscaledSpeed);
                            float curScaledSpeed = static_cast<float>(curUnscaledSpeed / maxSpeed);

                            double maxRunSpeed = csknow::weapon_speed::engineWeaponIdToMaxSpeed(
                                    demoEquipmentTypeToEngineWeaponId(playerAtTick.activeWeapon[patIndex]),
                                    csknow::weapon_speed::StatureOptions::Standing, false);
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

                            /*
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
                            */
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

                            if (nearestEnemyDistance > 1e7) {
                                std::cout << "bad nearest enemy distance " << nearestEnemyDistance
                                    << " game tick number " << ticks.gameTickNumber[tickIndex] << std::endl;
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
                            if (playerCanSeeEnemyNoFOV.count(playerId)) {
                                distanceToCoverWhenEnemyVisibleNoFOV.push_back(minDistanceToCover);
                            }
                            if (playerCanSeeEnemyFOV.count(playerId)) {
                                distanceToCoverWhenEnemyVisibleFOV.push_back(minDistanceToCover);
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
        file.createDataSet("/data/distance to cover when enemy visible no fov", distanceToCoverWhenEnemyVisibleNoFOV, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to cover when enemy visible fov", distanceToCoverWhenEnemyVisibleFOV, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to cover when firing", distanceToCoverWhenFiring, hdf5FlatCreateProps);
        file.createDataSet("/data/distance to cover when shot", distanceToCoverWhenShot, hdf5FlatCreateProps);

        file.createDataSet("/data/time from firing to teammate seeing enemy fov", timeFromFiringToTeammateSeeingEnemyFOV,
                           hdf5FlatCreateProps);
        file.createDataSet("/data/time from shot to teammate seeing enemy fov", timeFromShotToTeammateSeeingEnemyFOV,
                           hdf5FlatCreateProps);

        file.createDataSet("/data/pct time max speed ct", pctTimeMaxSpeedCT, hdf5FlatCreateProps);
        file.createDataSet("/data/pct time max speed t", pctTimeMaxSpeedT, hdf5FlatCreateProps);
        file.createDataSet("/data/pct time still ct", pctTimeStillCT, hdf5FlatCreateProps);
        file.createDataSet("/data/pct time still t", pctTimeStillT, hdf5FlatCreateProps);
        file.createDataSet("/data/ct wins", ctWins, hdf5FlatCreateProps);
    }
}