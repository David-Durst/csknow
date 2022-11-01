//
// Created by durst on 9/26/22.
//

#include "queries/training_moments/training_engagement_aim.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include <omp.h>

struct EngagementFireData {
    int16_t numShotsFired = 0;
    int16_t ticksSinceLastFire = std::numeric_limits<int16_t>::max();
    int64_t lastShotFiredTickId = INVALID_ID;
};

TrainingEngagementAimResult queryTrainingEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                       const PlayerAtTick & playerAtTick, const WeaponFire & weaponFire,
                                                       const EngagementResult & engagementResult) {
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpRoundId(numThreads);
    vector<vector<int64_t>> tmpTickId(numThreads);
    vector<vector<int64_t>> tmpEngagementId(numThreads);
    vector<vector<int64_t>> tmpAttackerPlayerId(numThreads);
    vector<vector<int64_t>> tmpVictimPlayerId(numThreads);
    vector<vector<int16_t>> tmpNumShotsFired(numThreads);
    vector<vector<int16_t>> tmpTicksSinceLastFire(numThreads);
    vector<vector<int64_t>> tmpLastShotFiredTickId(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpDeltaViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpRecoilAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpDeltaViewAngleRecoilAdjusted(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpDeltaPosition(numThreads);
    vector<vector<array<double, TOTAL_AIM_TICKS>>> tmpEyeToHeadDistance(numThreads);
    vector<vector<double>> tmpDistanceNormalization(numThreads);
    vector<vector<AimWeaponType>> tmpWeaponType(numThreads);

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
        map<int64_t, EngagementFireData> engagementToFireData;

        for (int64_t windowEndTickIndex = rollingWindow.lastReadTickId();
             windowEndTickIndex <= rounds.ticksPerRound[roundIndex].maxId; windowEndTickIndex = rollingWindow.readNextTick()) {
            int64_t tickIndex = rollingWindow.lastCurTickId();
            map<int64_t, int64_t> playerToFirePerTick;
            for (const auto & [_0, _1, fireIndex] :
                // use the prior tick for fire data so respecting causality
                ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex-1, tickIndex-1)) {
                playerToFirePerTick[weaponFire.shooter[fireIndex]] = fireIndex;
            }
            for (const auto & [_0, _1, engagementIndex] :
                engagementResult.engagementsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                tmpRoundId[threadNum].push_back(roundIndex);
                tmpTickId[threadNum].push_back(tickIndex);
                tmpEngagementId[threadNum].push_back(engagementIndex);
                if (engagementToFireData.find(engagementIndex) == engagementToFireData.end()) {
                    engagementToFireData[engagementIndex] = {
                        0, std::numeric_limits<int16_t>::max(), INVALID_ID                    };
                }

                int64_t attackerId = engagementResult.playerId[engagementIndex][0];
                tmpAttackerPlayerId[threadNum].push_back(attackerId);
                int64_t victimId = engagementResult.playerId[engagementIndex][1];
                tmpVictimPlayerId[threadNum].push_back(victimId);

                if (playerToFirePerTick.find(attackerId) != playerToFirePerTick.end()) {
                    engagementToFireData[attackerId].numShotsFired++;
                    engagementToFireData[attackerId].ticksSinceLastFire = 0;
                    engagementToFireData[attackerId].lastShotFiredTickId = tickIndex;
                }
                else if (engagementToFireData[attackerId].ticksSinceLastFire != std::numeric_limits<int16_t>::max()){
                    engagementToFireData[attackerId].ticksSinceLastFire++;
                }
                tmpNumShotsFired[threadNum].push_back(engagementToFireData[attackerId].numShotsFired);
                tmpTicksSinceLastFire[threadNum].push_back(engagementToFireData[attackerId].ticksSinceLastFire);
                tmpLastShotFiredTickId[threadNum].push_back(engagementToFireData[attackerId].lastShotFiredTickId);

                tmpDeltaViewAngle[threadNum].push_back({});
                tmpRecoilAngle[threadNum].push_back({});
                tmpDeltaViewAngleRecoilAdjusted[threadNum].push_back({});
                tmpEyeToHeadDistance[threadNum].push_back({});
                tmpDeltaPosition[threadNum].push_back({});

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

                    Vec2 deltaViewAngle = deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, curViewAngle);

                    tmpDeltaViewAngle[threadNum].back()[i] = deltaViewAngle;
                    tmpEyeToHeadDistance[threadNum].back()[i] = computeDistance(attackerEyePos, victimHeadPos);
                    tmpDeltaPosition[threadNum].back()[i] = attackerEyePos - victimEyePos;

                    // mul recoil by -1 as flipping all angles internally
                    Vec2 recoil {
                        playerAtTick.aimPunchX[attackerPATId],
                        -1 * playerAtTick.aimPunchY[attackerPATId]
                    };

                    tmpRecoilAngle[threadNum].back()[i] = recoil;
                    tmpDeltaViewAngleRecoilAdjusted[threadNum].back()[i] = deltaViewAngle + recoil * WEAPON_RECOIL_SCALE;
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
                           result.roundId.push_back(tmpTickId[minThreadId][tmpRowId]);
                           result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
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
