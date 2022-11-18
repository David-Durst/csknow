//
// Created by durst on 9/26/22.
//

#include "queries/training_moments/training_engagement_aim.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include "bots/analysis/vis_geometry.h"
#include <omp.h>

struct EngagementFireData {
    int16_t numShotsFired = 0;
    int16_t ticksSinceLastFire = std::numeric_limits<int16_t>::max();
    int64_t lastShotFiredTickId = INVALID_ID;
};

TrainingEngagementAimResult queryTrainingEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                       const PlayerAtTick & playerAtTick, const WeaponFire & weaponFire,
                                                       const EngagementResult & engagementResult,
                                                       const csknow::fire_history::FireHistoryResult & fireHistoryResult,
                                                       const VisPoints & visPoints) {
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpRoundId(numThreads);
    vector<vector<int64_t>> tmpTickId(numThreads);
    vector<vector<int64_t>> tmpDemoTickId(numThreads);
    vector<vector<int64_t>> tmpGameTickId(numThreads);
    vector<vector<int64_t>> tmpEngagementId(numThreads);
    vector<vector<int64_t>> tmpAttackerPlayerId(numThreads);
    vector<vector<int64_t>> tmpVictimPlayerId(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpAttackerViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpIdealViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpDeltaRelativeFirstHitHeadViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpDeltaRelativeCurHeadViewAngle(numThreads);
    vector<vector<array<float, TOTAL_AIM_TICKS>>> tmpRecoilIndex(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpScaledRecoilAngle(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksSinceLastFire(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksSinceLastHoldingAttack(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksUntilNextFire(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksUntilNextHoldingAttack(numThreads);
    vector<vector<array<bool, TOTAL_AIM_TICKS>>> tmpEnemyVisible(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpEnemyRelativeFirstHitHeadMinViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpEnemyRelativeFirstHitHeadMaxViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpEnemyRelativeFirstHitHeadCurHeadAngle(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpAttackerEyePos(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpVictimEyePos(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpAttackerVel(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpVictimVel(numThreads);
    vector<vector<AimWeaponType>> tmpWeaponType(numThreads);
    vector<vector<double>> tmpDistanceNormalization(numThreads);

    // for each round
    // for each tick
    // for each engagement in each tick
    // record where supposed to aim vs where aiming and distance
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        RollingWindow rollingWindow(rounds, ticks, playerAtTick);
        // minus 1 as need to include current tick in window size
        rollingWindow.setTemporalRange(rounds.ticksPerRound[roundIndex].minId + PAST_AIM_TICKS, tickRates,
                                       {DurationType::Ticks, 0, 0, PAST_AIM_TICKS, FUTURE_AIM_TICKS});
        const PlayerToPATWindows & playerToPatWindows = rollingWindow.getWindows();

        map<int64_t, Vec3> engagementToFirstHitVictimHeadPos;

        for (int64_t windowEndTickIndex = rollingWindow.lastReadTickId();
             windowEndTickIndex <= rounds.ticksPerRound[roundIndex].maxId; windowEndTickIndex = rollingWindow.readNextTick()) {
            int64_t tickIndex = rollingWindow.lastCurTickId();
            for (const auto & [_0, _1, engagementIndex] :
                engagementResult.engagementsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                tmpRoundId[threadNum].push_back(roundIndex);
                tmpTickId[threadNum].push_back(tickIndex);
                tmpDemoTickId[threadNum].push_back(ticks.demoTickNumber[tickIndex]);
                tmpGameTickId[threadNum].push_back(ticks.gameTickNumber[tickIndex]);
                tmpEngagementId[threadNum].push_back(engagementIndex);

                int64_t attackerId = engagementResult.playerId[engagementIndex][0];
                tmpAttackerPlayerId[threadNum].push_back(attackerId);
                int64_t victimId = engagementResult.playerId[engagementIndex][1];
                tmpVictimPlayerId[threadNum].push_back(victimId);

                // if first time dealing with this engagement, get the PAT for the victim on first hit
                if (engagementToFirstHitVictimHeadPos.find(engagementIndex) == engagementToFirstHitVictimHeadPos.end()) {
                    int64_t firstHitTickIndex = engagementResult.hurtTickIds[engagementIndex][0];
                    for (int64_t patIndex = ticks.patPerTick[firstHitTickIndex].minId;
                         patIndex <= ticks.patPerTick[firstHitTickIndex].maxId; patIndex++) {
                        if (playerAtTick.playerId[patIndex] == victimId) {
                            engagementToFirstHitVictimHeadPos[engagementIndex] =
                                getCenterHeadCoordinatesForPlayer({
                                    playerAtTick.posX[patIndex],
                                    playerAtTick.posY[patIndex],
                                    playerAtTick.eyePosZ[patIndex]
                                }, {
                                    playerAtTick.viewX[patIndex],
                                    playerAtTick.viewY[patIndex]
                                }, playerAtTick.duckAmount[patIndex]);
                            break;
                        }
                    }
                }

                tmpAttackerViewAngle[threadNum].push_back({});
                tmpIdealViewAngle[threadNum].push_back({});
                tmpDeltaRelativeFirstHitHeadViewAngle[threadNum].push_back({});
                tmpDeltaRelativeCurHeadViewAngle[threadNum].push_back({});
                tmpRecoilIndex[threadNum].push_back({});
                tmpScaledRecoilAngle[threadNum].push_back({});
                tmpTicksSinceLastFire[threadNum].push_back({});
                tmpTicksSinceLastHoldingAttack[threadNum].push_back({});
                tmpTicksUntilNextFire[threadNum].push_back({});
                tmpTicksUntilNextHoldingAttack[threadNum].push_back({});
                tmpEnemyVisible[threadNum].push_back({});
                tmpEnemyRelativeFirstHitHeadMinViewAngle[threadNum].push_back({});
                tmpEnemyRelativeFirstHitHeadMaxViewAngle[threadNum].push_back({});
                tmpEnemyRelativeFirstHitHeadCurHeadAngle[threadNum].push_back({});
                tmpAttackerEyePos[threadNum].push_back({});
                tmpVictimEyePos[threadNum].push_back({});
                tmpAttackerVel[threadNum].push_back({});
                tmpVictimVel[threadNum].push_back({});

                for (size_t i = 0; i < TOTAL_AIM_TICKS; i++) {
                    const int64_t & attackerPATId = playerToPatWindows.at(attackerId).fromOldest(static_cast<int64_t>(i));
                    const int64_t & victimPATId = playerToPatWindows.at(victimId).fromOldest(static_cast<int64_t>(i));

                    Vec3 attackerEyePos {
                        playerAtTick.posX[attackerPATId],
                        playerAtTick.posY[attackerPATId],
                        playerAtTick.eyePosZ[attackerPATId]
                    };

                    Vec3 victimEyePos {
                        playerAtTick.posX[victimPATId],
                        playerAtTick.posY[victimPATId],
                        playerAtTick.eyePosZ[victimPATId]
                    };

                    Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(
                        victimEyePos,
                        {
                            playerAtTick.viewX[victimPATId],
                            playerAtTick.viewY[victimPATId]
                        },
                        playerAtTick.duckAmount[victimPATId]
                    );

                    Vec2 curViewAngle {
                        playerAtTick.viewX[attackerPATId],
                        playerAtTick.viewY[attackerPATId]
                    };
                    curViewAngle.normalize();
                    Vec2 idealViewAngle = viewFromOriginToDest(attackerEyePos, victimHeadPos);

                    tmpAttackerViewAngle[threadNum].back()[i] = curViewAngle;
                    tmpIdealViewAngle[threadNum].back()[i] = idealViewAngle;

                    tmpDeltaRelativeFirstHitHeadViewAngle[threadNum].back()[i] =
                        deltaViewFromOriginToDest(attackerEyePos,
                                                  engagementToFirstHitVictimHeadPos[engagementIndex], curViewAngle);
                    tmpDeltaRelativeCurHeadViewAngle[threadNum].back()[i] =
                        deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, curViewAngle);

                    tmpRecoilIndex[threadNum].back()[i] = playerAtTick.recoilIndex[attackerPATId];

                    // mul recoil by -1 as flipping all angles internally
                    Vec2 recoil {
                        playerAtTick.aimPunchX[attackerPATId],
                        -1 * playerAtTick.aimPunchY[attackerPATId]
                    };

                    tmpScaledRecoilAngle[threadNum].back()[i] = recoil * WEAPON_RECOIL_SCALE;

                    tmpTicksSinceLastFire[threadNum].back()[i] = fireHistoryResult.ticksSinceLastFire[attackerPATId];
                    tmpTicksSinceLastHoldingAttack[threadNum].back()[i] =
                        fireHistoryResult.ticksSinceLastHoldingAttack[attackerPATId];
                    tmpTicksUntilNextFire[threadNum].back()[i] = fireHistoryResult.ticksUntilNextFire[attackerPATId];
                    tmpTicksUntilNextHoldingAttack[threadNum].back()[i] =
                        fireHistoryResult.ticksUntilNextHoldingAttack[attackerPATId];


                    vector<CellIdAndDistance> attackerCellIdsByDistances = visPoints.getCellVisPointsByDistance(
                        attackerEyePos);
                    vector<CellIdAndDistance> victimCellIdsByDistances = visPoints.getCellVisPointsByDistance(
                        victimEyePos);
                    vector<CellVisPoint> victimTwoClosestCellVisPoints = {
                        visPoints.getCellVisPoints()[victimCellIdsByDistances[0].cellId],
                        visPoints.getCellVisPoints()[victimCellIdsByDistances[1].cellId]
                    };
                    bool enemyInFOV = getCellsInFOV(victimTwoClosestCellVisPoints, attackerEyePos,
                                                    curViewAngle);
                    // vis from either of attackers two closest cell vis points
                    bool enemyVisNoFOV = false;
                    for (size_t i = 0; i < 2; i++) {
                        for (size_t j = 0; j < 2; j++) {
                            enemyVisNoFOV |= visPoints.getCellVisPoints()[attackerCellIdsByDistances[i].cellId]
                                .visibleFromCurPoint[victimCellIdsByDistances[j].cellId];
                        }
                    }
                    tmpEnemyVisible[threadNum].back()[i] = enemyInFOV && enemyVisNoFOV;

                    tmpAttackerEyePos[threadNum].back()[i] = attackerEyePos;
                    tmpVictimEyePos[threadNum].back()[i] = victimEyePos;
                    tmpAttackerVel[threadNum].back()[i] = {
                        playerAtTick.velX[attackerPATId],
                        playerAtTick.velY[attackerPATId],
                        playerAtTick.velZ[attackerPATId]
                    };
                    tmpVictimVel[threadNum].back()[i] = {
                        playerAtTick.velX[victimPATId],
                        playerAtTick.velY[victimPATId],
                        playerAtTick.velZ[victimPATId]
                    };

                }

                // compute normalization constants, used to visualize inference
                const int64_t & curAttackerPATId = playerToPatWindows.at(attackerId).fromNewest(FUTURE_AIM_TICKS);
                const int64_t & curVictimPATId = playerToPatWindows.at(victimId).fromNewest(FUTURE_AIM_TICKS);

                Vec3 curAttackerEyePos = {
                    playerAtTick.posX[curAttackerPATId],
                    playerAtTick.posY[curAttackerPATId],
                    playerAtTick.eyePosZ[curAttackerPATId]
                };

                Vec3 victimBotPos = {
                    playerAtTick.posX[curVictimPATId],
                    playerAtTick.posY[curVictimPATId],
                    playerAtTick.posZ[curVictimPATId]
                };
                Vec2 viewAngleToBotPos = vectorAngles(victimBotPos - curAttackerEyePos);
                Vec3 victimTopPos = victimBotPos;
                victimTopPos.z += PLAYER_HEIGHT;
                Vec2 topVsBotViewAngle = deltaViewFromOriginToDest(curAttackerEyePos, victimTopPos, viewAngleToBotPos);
                tmpDistanceNormalization[threadNum].push_back(std::abs(topVsBotViewAngle.y));

                // assume attacker has same weapon for all ticks
                AimWeaponType aimWeaponType;
                DemoEquipmentType demoEquipmentType =
                        static_cast<DemoEquipmentType>(playerAtTick.activeWeapon[curAttackerPATId]);
                if (demoEquipmentType < DemoEquipmentType::EQ_PISTOL_END) {
                    aimWeaponType = AimWeaponType::Pistol;
                }
                else if (demoEquipmentType < DemoEquipmentType::EQ_SMG_END) {
                    aimWeaponType = AimWeaponType::SMG;
                }
                else if (demoEquipmentType < DemoEquipmentType::EQ_HEAVY_END) {
                    aimWeaponType = AimWeaponType::Heavy;
                }
                else if (demoEquipmentType < DemoEquipmentType::EQ_RIFLE_END) {
                    if (demoEquipmentType == DemoEquipmentType::EqScout || demoEquipmentType == DemoEquipmentType::EqAWP) {
                        aimWeaponType = AimWeaponType::Sniper;
                    }
                    else {
                        aimWeaponType = AimWeaponType::AR;
                    }
                }
                else {
                    aimWeaponType = AimWeaponType::Unknown;
                }
                tmpWeaponType[threadNum].push_back(aimWeaponType);
            }
        }

        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
    }

    TrainingEngagementAimResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.tickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.roundId.push_back(tmpRoundId[minThreadId][tmpRowId]);
                           result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                           result.demoTickId.push_back(tmpDemoTickId[minThreadId][tmpRowId]);
                           result.gameTickId.push_back(tmpGameTickId[minThreadId][tmpRowId]);
                           result.engagementId.push_back(tmpEngagementId[minThreadId][tmpRowId]);
                           result.attackerPlayerId.push_back(tmpAttackerPlayerId[minThreadId][tmpRowId]);
                           result.victimPlayerId.push_back(tmpVictimPlayerId[minThreadId][tmpRowId]);
                           result.numShotsFired.push_back(tmpNumShotsFired[minThreadId][tmpRowId]);
                           result.ticksSinceLastFire.push_back(tmpTicksSinceLastFire[minThreadId][tmpRowId]);
                           result.lastShotFiredTickId.push_back(tmpLastShotFiredTickId[minThreadId][tmpRowId]);
                           result.deltaViewAngle.push_back(tmpDeltaViewAngle[minThreadId][tmpRowId]);
                           result.recoilAngle.push_back(tmpRecoilAngle[minThreadId][tmpRowId]);
                           result.deltaViewAngleRecoilAdjusted.push_back(tmpDeltaViewAngleRecoilAdjusted[minThreadId][tmpRowId]);
                           result.deltaPosition.push_back(tmpDeltaPosition[minThreadId][tmpRowId]);
                           result.eyeToHeadDistance.push_back(tmpEyeToHeadDistance[minThreadId][tmpRowId]);
                           result.distanceNormalization.push_back(tmpDistanceNormalization[minThreadId][tmpRowId]);
                           result.weaponType.push_back(tmpWeaponType[minThreadId][tmpRowId]);
                       });
    return result;
}
