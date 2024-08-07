//
// Created by durst on 9/26/22.
//

#include "queries/training_moments/training_engagement_aim.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include "bots/analysis/vis_geometry.h"
#include "file_helpers.h"
#include "bots/analysis/streaming_manager.h"
#include <omp.h>
#include <atomic>

struct EngagementHurtData {
    int64_t engagementIndex;
    bool hitRemaining;
};

TrainingEngagementAimResult queryTrainingEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                       const PlayerAtTick & playerAtTick,
                                                       const EngagementResult & engagementResult,
                                                       const csknow::fire_history::FireHistoryResult & fireHistoryResult,
                                                       const VisPoints & visPoints,
                                                       const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell,
                                                       bool parallelize) {
    int numThreads = omp_get_max_threads();
    std::atomic<int64_t> roundsProcessed = 0;
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpRoundId(numThreads);
    vector<vector<int64_t>> tmpTickId(numThreads);
    vector<vector<int64_t>> tmpDemoTickId(numThreads);
    vector<vector<int64_t>> tmpGameTickId(numThreads);
    vector<vector<int64_t>> tmpGameTime(numThreads);
    vector<vector<int64_t>> tmpEngagementId(numThreads);
    vector<vector<int64_t>> tmpAttackerPlayerId(numThreads);
    vector<vector<int64_t>> tmpVictimPlayerId(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpAttackerViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpIdealViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpDeltaRelativeFirstHeadViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpDeltaRelativeCurHeadViewAngle(numThreads);
    vector<vector<array<bool, TOTAL_AIM_TICKS>>> tmpHitVictim(numThreads);
    vector<vector<array<float, TOTAL_AIM_TICKS>>> tmpRecoilIndex(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpScaledRecoilAngle(numThreads);
    vector<vector<array<bool, TOTAL_AIM_TICKS>>> tmpHoldingAttack(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksSinceLastFire(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksSinceLastHoldingAttack(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksUntilNextFire(numThreads);
    vector<vector<array<int64_t, TOTAL_AIM_TICKS>>> tmpTicksUntilNextHoldingAttack(numThreads);
    vector<vector<array<bool, TOTAL_AIM_TICKS>>> tmpVictimVisible(numThreads);
    vector<vector<array<bool, TOTAL_AIM_TICKS>>> tmpVictimVisibleYet(numThreads);
    vector<vector<array<bool, TOTAL_AIM_TICKS>>> tmpVictimAlive(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimMinViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimMaxViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimCurHeadAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimRelativeFirstHeadMinViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimRelativeFirstHeadMaxViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimRelativeFirstHeadCurHeadAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimRelativeCurHeadMinViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimRelativeCurHeadMaxViewAngle(numThreads);
    vector<vector<array<Vec2, TOTAL_AIM_TICKS>>> tmpVictimRelativeCurHeadCurHeadAngle(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpAttackerEyePos(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpVictimEyePos(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpDeltaEyePos(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpAttackerVel(numThreads);
    vector<vector<array<Vec3, TOTAL_AIM_TICKS>>> tmpVictimVel(numThreads);
    vector<vector<array<float, TOTAL_AIM_TICKS>>> tmpAttackerDuckAmount(numThreads);
    vector<vector<array<float, TOTAL_AIM_TICKS>>> tmpNextPrimaryAttack(numThreads);
    vector<vector<array<float, TOTAL_AIM_TICKS>>> tmpNextSecondaryAttack(numThreads);
    vector<vector<array<float, TOTAL_AIM_TICKS>>> tmpAttackerGameTime(numThreads);
    vector<vector<array<DemoEquipmentType, TOTAL_AIM_TICKS>>> tmpWeaponId(numThreads);
    vector<vector<AimWeaponType>> tmpWeaponType(numThreads);

    // for each round
    // for each tick
    // for each engagement in each tick
    // record where supposed to aim vs where aiming and distance
#pragma omp parallel for if(parallelize)
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

        map<int64_t, Vec2> engagementToFirstTickIdealViewAngle;
        map<int64_t, int64_t> engagementToVictimLastAlivePATId;
        map<int64_t, int64_t> engagementToVictimFirstVisiblePATId;

        for (int64_t windowEndTickIndex = rollingWindow.lastReadTickId();
             windowEndTickIndex <= rounds.ticksPerRound[roundIndex].maxId; windowEndTickIndex = rollingWindow.readNextTick()) {
            int64_t tickIndex = rollingWindow.lastCurTickId();
            // late start and early termination of engagements
            // each player can be attacker in at most one engagement at a time
            // late start: sort priority by
            // 1. at least hit remaining in engagement
            // 2. earliest starting of those with at least 1 hit remaining
            // 3. if all remaining engagments have no active hits, take the engagement with the most ticks left
            // early termination: filter out engagements where attacker dead during window of computation
            map<int64_t, vector<EngagementHurtData>> playerToAttackingEngagements;
            for (const auto & [_0, _1, engagementIndex] :
                engagementResult.engagementsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                int64_t attackerId = engagementResult.playerId[engagementIndex][0];

                // if attacker isn't alive during window, skip
                bool attackerDeadDuringWindow = false;
                for (size_t i = 0; i < TOTAL_AIM_TICKS; i++) {
                    const int64_t & attackerPATId =
                        playerToPatWindows.at(attackerId).fromOldest(static_cast<int64_t>(i));
                    if (!playerAtTick.isAlive[attackerPATId]) {
                        attackerDeadDuringWindow = true;
                        break;
                    }
                }
                if (attackerDeadDuringWindow) {
                    continue;
                }

                // otherwise add for sorting
                bool hitRemaining = false;
                for (const auto & hurtTickId : engagementResult.hurtTickIds[engagementIndex]) {
                    if (hurtTickId >= tickIndex) {
                        hitRemaining = true;
                        break;
                    }
                }
                playerToAttackingEngagements[attackerId].push_back({engagementIndex, hitRemaining});
            }
            vector<int64_t> selectedEngagements;
            for (auto & [_, attackingEngagements] : playerToAttackingEngagements) {
                std::sort(attackingEngagements.begin(), attackingEngagements.end(),
                          [engagementResult](const EngagementHurtData & a, const EngagementHurtData & b) {
                    bool aStartFirst =
                        (engagementResult.startTickId[a.engagementIndex] < engagementResult.startTickId[b.engagementIndex]) ||
                        (engagementResult.startTickId[a.engagementIndex] == engagementResult.startTickId[b.engagementIndex] &&
                            a.engagementIndex < b.engagementIndex);
                    bool aEndLater =
                      (engagementResult.endTickId[a.engagementIndex] > engagementResult.endTickId[b.engagementIndex]) ||
                      (engagementResult.endTickId[a.engagementIndex] == engagementResult.endTickId[b.engagementIndex] &&
                       a.engagementIndex > b.engagementIndex);
                    return (a.hitRemaining && !b.hitRemaining) ||
                        (a.hitRemaining && b.hitRemaining && aStartFirst) ||
                        (!a.hitRemaining && !b.hitRemaining && aEndLater);
                });
                selectedEngagements.push_back(attackingEngagements[0].engagementIndex);
            }

            // compute data for relevant engagements
            for (const auto & engagementIndex : selectedEngagements) {
                tmpRoundId[threadNum].push_back(roundIndex);
                tmpTickId[threadNum].push_back(tickIndex);
                tmpDemoTickId[threadNum].push_back(ticks.demoTickNumber[tickIndex]);
                tmpGameTickId[threadNum].push_back(ticks.gameTickNumber[tickIndex]);
                tmpGameTime[threadNum].push_back(ticks.gameTime[tickIndex]);
                tmpEngagementId[threadNum].push_back(engagementIndex);

                int64_t attackerId = engagementResult.playerId[engagementIndex][0];
                tmpAttackerPlayerId[threadNum].push_back(attackerId);
                int64_t victimId = engagementResult.playerId[engagementIndex][1];
                tmpVictimPlayerId[threadNum].push_back(victimId);


                // if first time dealing with this engagement, get the first tick head pos
                if (engagementToFirstTickIdealViewAngle.find(engagementIndex) ==
                    engagementToFirstTickIdealViewAngle.end()) {
                    const int64_t & attackerPATId =
                        playerToPatWindows.at(attackerId).fromOldest(static_cast<int64_t>(PAST_AIM_TICKS));
                    // first tick in engagmenet, so no need to check for last frame where victim is alive
                    const int64_t & victimPATId =
                        playerToPatWindows.at(victimId).fromOldest(static_cast<int64_t>(PAST_AIM_TICKS));

                    Vec3 attackerEyePos {
                        playerAtTick.posX[attackerPATId],
                        playerAtTick.posY[attackerPATId],
                        playerAtTick.eyePosZ[attackerPATId]
                    };

                    Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(
                        {
                            playerAtTick.posX[victimPATId],
                            playerAtTick.posY[victimPATId],
                            playerAtTick.eyePosZ[victimPATId]
                        },
                        {
                            playerAtTick.viewX[victimPATId],
                            playerAtTick.viewY[victimPATId]
                        },
                        playerAtTick.duckAmount[victimPATId]
                    );

                    Vec2 idealViewAngle = viewFromOriginToDest(attackerEyePos, victimHeadPos);
                    engagementToFirstTickIdealViewAngle[engagementIndex] = idealViewAngle;
                }

                tmpAttackerViewAngle[threadNum].push_back({});
                tmpIdealViewAngle[threadNum].push_back({});
                tmpDeltaRelativeFirstHeadViewAngle[threadNum].push_back({});
                tmpDeltaRelativeCurHeadViewAngle[threadNum].push_back({});
                tmpHitVictim[threadNum].push_back({});
                tmpRecoilIndex[threadNum].push_back({});
                tmpScaledRecoilAngle[threadNum].push_back({});
                tmpHoldingAttack[threadNum].push_back({});
                tmpTicksSinceLastFire[threadNum].push_back({});
                tmpTicksSinceLastHoldingAttack[threadNum].push_back({});
                tmpTicksUntilNextFire[threadNum].push_back({});
                tmpTicksUntilNextHoldingAttack[threadNum].push_back({});
                tmpVictimVisible[threadNum].push_back({});
                tmpVictimVisibleYet[threadNum].push_back({});
                tmpVictimAlive[threadNum].push_back({});
                tmpVictimMinViewAngle[threadNum].push_back({});
                tmpVictimMaxViewAngle[threadNum].push_back({});
                tmpVictimCurHeadAngle[threadNum].push_back({});
                tmpVictimRelativeFirstHeadMinViewAngle[threadNum].push_back({});
                tmpVictimRelativeFirstHeadMaxViewAngle[threadNum].push_back({});
                tmpVictimRelativeFirstHeadCurHeadAngle[threadNum].push_back({});
                tmpVictimRelativeCurHeadMinViewAngle[threadNum].push_back({});
                tmpVictimRelativeCurHeadMaxViewAngle[threadNum].push_back({});
                tmpVictimRelativeCurHeadCurHeadAngle[threadNum].push_back({});
                tmpAttackerEyePos[threadNum].push_back({});
                tmpVictimEyePos[threadNum].push_back({});
                tmpDeltaEyePos[threadNum].push_back({});
                tmpAttackerVel[threadNum].push_back({});
                tmpVictimVel[threadNum].push_back({});
                tmpAttackerDuckAmount[threadNum].push_back({});
                tmpNextPrimaryAttack[threadNum].push_back({});
                tmpNextSecondaryAttack[threadNum].push_back({});
                tmpAttackerGameTime[threadNum].push_back({});
                tmpWeaponId[threadNum].push_back({});

                for (size_t i = 0; i < TOTAL_AIM_TICKS; i++) {
                    const int64_t & attackerPATId =
                        playerToPatWindows.at(attackerId).fromOldest(static_cast<int64_t>(i));
                    // need to check for last frame where victim is alive
                    const int64_t & uncheckedVictimPATId =
                        playerToPatWindows.at(victimId).fromOldest(static_cast<int64_t>(i));
                    int64_t victimPATId = uncheckedVictimPATId;
                    // assume alive on first tick, must be true as otherwise no engagement
                    if (engagementToVictimLastAlivePATId.find(engagementIndex) ==
                        engagementToVictimLastAlivePATId.end() ||
                        (playerAtTick.isAlive[victimPATId] &&
                        victimPATId > engagementToVictimLastAlivePATId[engagementIndex])) {
                        engagementToVictimLastAlivePATId[engagementIndex] = victimPATId;
                    }
                    else if (!playerAtTick.isAlive[victimPATId]) {
                        victimPATId = engagementToVictimLastAlivePATId[engagementIndex];
                    }
                    // do nothing if victim is alive but not newest recorded alive time

                    Vec3 attackerEyePos {
                        playerAtTick.posX[attackerPATId],
                        playerAtTick.posY[attackerPATId],
                        playerAtTick.eyePosZ[attackerPATId]
                    };

                    Vec3 victimFootPos {
                        playerAtTick.posX[victimPATId],
                        playerAtTick.posY[victimPATId],
                        playerAtTick.posZ[victimPATId]
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

                    tmpDeltaRelativeFirstHeadViewAngle[threadNum].back()[i] =
                        wrappedAngleDifference(curViewAngle, engagementToFirstTickIdealViewAngle[engagementIndex]);
                    tmpDeltaRelativeCurHeadViewAngle[threadNum].back()[i] =
                        deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, curViewAngle);

                    tmpHitVictim[threadNum].back()[i] = fireHistoryResult.hitEnemy[attackerPATId] &&
                        fireHistoryResult.victims[attackerPATId].find(victimId) != fireHistoryResult.victims[attackerPATId].end();

                    tmpRecoilIndex[threadNum].back()[i] = playerAtTick.recoilIndex[attackerPATId];

                    // mul recoil by -1 as flipping all angles internally
                    // DON'T DO THIS, MAKES CALCULATIONS A PAIN
                    Vec2 recoil {
                        playerAtTick.aimPunchX[attackerPATId],
                        playerAtTick.aimPunchY[attackerPATId]
                    };

                    tmpScaledRecoilAngle[threadNum].back()[i] = recoil * WEAPON_RECOIL_SCALE;

                    tmpHoldingAttack[threadNum].back()[i] =
                        fireHistoryResult.ticksSinceLastHoldingAttack[attackerPATId] == 0;
                    tmpTicksSinceLastFire[threadNum].back()[i] =
                        std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, fireHistoryResult.ticksSinceLastFire[attackerPATId]);
                    tmpTicksSinceLastHoldingAttack[threadNum].back()[i] =
                        std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, fireHistoryResult.ticksSinceLastHoldingAttack[attackerPATId]);
                    tmpTicksUntilNextFire[threadNum].back()[i] =
                        std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, fireHistoryResult.ticksUntilNextFire[attackerPATId]);
                    tmpTicksUntilNextHoldingAttack[threadNum].back()[i] =
                        std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, fireHistoryResult.ticksUntilNextHoldingAttack[attackerPATId]);


                    bool curTickVictimVisible = demoIsVisible(playerAtTick, attackerPATId, victimPATId,
                                                              nearestNavCell, visPoints).fov;
                    tmpVictimVisible[threadNum].back()[i] = curTickVictimVisible;
                    if (curTickVictimVisible &&
                        engagementToVictimFirstVisiblePATId.find(engagementIndex) ==
                        engagementToVictimFirstVisiblePATId.end()) {
                        engagementToVictimFirstVisiblePATId[engagementIndex] = victimPATId;
                    }
                    tmpVictimVisibleYet[threadNum].back()[i] =
                        engagementToVictimFirstVisiblePATId.find(engagementIndex) !=
                        engagementToVictimFirstVisiblePATId.end() &&
                        engagementToVictimFirstVisiblePATId[engagementIndex] <= victimPATId;

                    tmpVictimAlive[threadNum].back()[i] = playerAtTick.isAlive[uncheckedVictimPATId];

                    AABB victimAABB = getAABBForPlayer(victimFootPos, playerAtTick.duckAmount[victimPATId]);
                    vector<Vec3> aabbCorners = getAABBCorners(victimAABB);
                    Vec2 victimMinViewAngle{std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max()};
                    Vec2 victimMaxViewAngle{-1*std::numeric_limits<double>::max(),
                                            -1*std::numeric_limits<double>::max()};
                    Vec2 victimMinViewAngleFirstHead{std::numeric_limits<double>::max(),
                                                   std::numeric_limits<double>::max()};
                    Vec2 victimMaxViewAngleFirstHead{-1*std::numeric_limits<double>::max(),
                                                   -1*std::numeric_limits<double>::max()};
                    Vec2 victimMinViewAngleCur{std::numeric_limits<double>::max(),
                                              std::numeric_limits<double>::max()};
                    Vec2 victimMaxViewAngleCur{-1*std::numeric_limits<double>::max(),
                                              -1*std::numeric_limits<double>::max()};
                    for (const auto & aabbCorner : aabbCorners) {
                        Vec2 aabbViewAngle = viewFromOriginToDest(attackerEyePos, aabbCorner);
                        Vec2 deltaAABBViewAngleFirstHead =
                            wrappedAngleDifference(aabbViewAngle, engagementToFirstTickIdealViewAngle[engagementIndex]);
                        Vec2 deltaAABBViewAngleCur =
                            deltaViewFromOriginToDest(attackerEyePos,
                                                      victimHeadPos,
                                                      aabbViewAngle);
                        victimMinViewAngle = min(victimMinViewAngle, aabbViewAngle);
                        victimMaxViewAngle = max(victimMaxViewAngle, aabbViewAngle);
                        victimMinViewAngleFirstHead = min(victimMinViewAngleFirstHead, deltaAABBViewAngleFirstHead);
                        victimMaxViewAngleFirstHead = max(victimMaxViewAngleFirstHead, deltaAABBViewAngleFirstHead);
                        victimMinViewAngleCur = min(victimMinViewAngleCur, deltaAABBViewAngleCur);
                        victimMaxViewAngleCur = max(victimMaxViewAngleCur, deltaAABBViewAngleCur);
                    }

                    tmpVictimMinViewAngle[threadNum].back()[i] = victimMinViewAngle;
                    tmpVictimMaxViewAngle[threadNum].back()[i] = victimMaxViewAngle;
                    tmpVictimCurHeadAngle[threadNum].back()[i] = idealViewAngle;
                    tmpVictimRelativeFirstHeadMinViewAngle[threadNum].back()[i] = victimMinViewAngleFirstHead;
                    tmpVictimRelativeFirstHeadMaxViewAngle[threadNum].back()[i] = victimMaxViewAngleFirstHead;
                    tmpVictimRelativeFirstHeadCurHeadAngle[threadNum].back()[i] =
                        wrappedAngleDifference(idealViewAngle, engagementToFirstTickIdealViewAngle[engagementIndex]);
                    tmpVictimRelativeCurHeadMinViewAngle[threadNum].back()[i] = victimMinViewAngleCur;
                    tmpVictimRelativeCurHeadMaxViewAngle[threadNum].back()[i] = victimMaxViewAngleCur;
                    tmpVictimRelativeCurHeadCurHeadAngle[threadNum].back()[i] =
                        deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, idealViewAngle);

                    tmpAttackerEyePos[threadNum].back()[i] = attackerEyePos;
                    tmpVictimEyePos[threadNum].back()[i] = victimEyePos;
                    tmpDeltaEyePos[threadNum].back()[i] = victimEyePos - attackerEyePos;
                    tmpAttackerVel[threadNum].back()[i] = {
                        playerAtTick.velX[attackerPATId],
                        playerAtTick.velY[attackerPATId],
                        playerAtTick.velZ[attackerPATId]
                    };
                    if (playerAtTick.isAlive[uncheckedVictimPATId]) {
                        tmpVictimVel[threadNum].back()[i] = {
                            playerAtTick.velX[uncheckedVictimPATId],
                            playerAtTick.velY[uncheckedVictimPATId],
                            playerAtTick.velZ[uncheckedVictimPATId]
                        };
                    }
                    else {
                        tmpVictimVel[threadNum].back()[i] = {0., 0., 0.};
                    }
                    tmpAttackerDuckAmount[threadNum].back()[i] = playerAtTick.duckAmount[attackerPATId];
                    tmpNextPrimaryAttack[threadNum].back()[i] = playerAtTick.nextPrimaryAttack[attackerPATId];
                    tmpNextSecondaryAttack[threadNum].back()[i] = playerAtTick.nextSecondaryAttack[attackerPATId];
                    tmpAttackerGameTime[threadNum].back()[i] = playerAtTick.gameTime[attackerPATId];
                    tmpWeaponId[threadNum].back()[i] =
                        static_cast<DemoEquipmentType>(playerAtTick.activeWeapon[attackerPATId]);
                }

                // assume attacker has same weapon for all ticks
                const int64_t &curAttackerPATId = playerToPatWindows.at(attackerId).fromNewest(FUTURE_AIM_TICKS);
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
                    else if (demoEquipmentType == DemoEquipmentType::EqAK47) {
                        aimWeaponType = AimWeaponType::AK;
                    }
                    else if (demoEquipmentType == DemoEquipmentType::EqM4A1) {
                        aimWeaponType = AimWeaponType::M4A1;
                    }
                    else {
                        aimWeaponType = AimWeaponType::AROther;
                    }
                }
                else {
                    aimWeaponType = AimWeaponType::Unknown;
                }
                tmpWeaponType[threadNum].push_back(aimWeaponType);
            }
        }

        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        roundsProcessed++;
        printProgress(roundsProcessed, rounds.size);
    }

    TrainingEngagementAimResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.tickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.roundId.push_back(tmpRoundId[minThreadId][tmpRowId]);
                           result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                           result.demoTickId.push_back(tmpDemoTickId[minThreadId][tmpRowId]);
                           result.gameTickId.push_back(tmpGameTickId[minThreadId][tmpRowId]);
                           result.gameTime.push_back(tmpGameTime[minThreadId][tmpRowId]);
                           result.engagementId.push_back(tmpEngagementId[minThreadId][tmpRowId]);
                           result.attackerPlayerId.push_back(tmpAttackerPlayerId[minThreadId][tmpRowId]);
                           result.victimPlayerId.push_back(tmpVictimPlayerId[minThreadId][tmpRowId]);
                           result.attackerViewAngle.push_back(tmpAttackerViewAngle[minThreadId][tmpRowId]);
                           result.idealViewAngle.push_back(tmpIdealViewAngle[minThreadId][tmpRowId]);
                           result.deltaRelativeFirstHeadViewAngle.push_back(
                               tmpDeltaRelativeFirstHeadViewAngle[minThreadId][tmpRowId]);
                           result.deltaRelativeCurHeadViewAngle.push_back(
                               tmpDeltaRelativeCurHeadViewAngle[minThreadId][tmpRowId]);
                           result.holdingAttack.push_back(tmpHoldingAttack[minThreadId][tmpRowId]);
                           result.hitVictim.push_back(tmpHitVictim[minThreadId][tmpRowId]);
                           result.recoilIndex.push_back(tmpRecoilIndex[minThreadId][tmpRowId]);
                           result.scaledRecoilAngle.push_back(tmpScaledRecoilAngle[minThreadId][tmpRowId]);
                           result.ticksSinceLastFire.push_back(tmpTicksSinceLastFire[minThreadId][tmpRowId]);
                           result.ticksSinceLastHoldingAttack.push_back(
                               tmpTicksSinceLastHoldingAttack[minThreadId][tmpRowId]);
                           result.ticksUntilNextFire.push_back(tmpTicksUntilNextFire[minThreadId][tmpRowId]);
                           result.ticksUntilNextHoldingAttack.push_back(
                               tmpTicksUntilNextHoldingAttack[minThreadId][tmpRowId]);
                           result.victimVisible.push_back(tmpVictimVisible[minThreadId][tmpRowId]);
                           result.victimVisibleYet.push_back(tmpVictimVisibleYet[minThreadId][tmpRowId]);
                           result.victimAlive.push_back(tmpVictimAlive[minThreadId][tmpRowId]);
                           result.victimMinViewAngle.push_back(
                               tmpVictimMinViewAngle[minThreadId][tmpRowId]);
                           result.victimMaxViewAngle.push_back(
                               tmpVictimMaxViewAngle[minThreadId][tmpRowId]);
                           result.victimCurHeadViewAngle.push_back(
                               tmpVictimCurHeadAngle[minThreadId][tmpRowId]);
                           result.victimRelativeFirstHeadMinViewAngle.push_back(
                               tmpVictimRelativeFirstHeadMinViewAngle[minThreadId][tmpRowId]);
                           result.victimRelativeFirstHeadMaxViewAngle.push_back(
                               tmpVictimRelativeFirstHeadMaxViewAngle[minThreadId][tmpRowId]);
                           result.victimRelativeFirstHeadCurHeadViewAngle.push_back(
                               tmpVictimRelativeFirstHeadCurHeadAngle[minThreadId][tmpRowId]);
                           result.victimRelativeCurHeadMinViewAngle.push_back(
                               tmpVictimRelativeCurHeadMinViewAngle[minThreadId][tmpRowId]);
                           result.victimRelativeCurHeadMaxViewAngle.push_back(
                               tmpVictimRelativeCurHeadMaxViewAngle[minThreadId][tmpRowId]);
                           result.victimRelativeCurHeadCurHeadViewAngle.push_back(
                               tmpVictimRelativeCurHeadCurHeadAngle[minThreadId][tmpRowId]);
                           result.attackerEyePos.push_back(tmpAttackerEyePos[minThreadId][tmpRowId]);
                           result.victimEyePos.push_back(tmpVictimEyePos[minThreadId][tmpRowId]);
                           result.deltaEyePos.push_back(tmpDeltaEyePos[minThreadId][tmpRowId]);
                           result.attackerVel.push_back(tmpAttackerVel[minThreadId][tmpRowId]);
                           result.victimVel.push_back(tmpVictimVel[minThreadId][tmpRowId]);
                           result.attackerDuckAmount.push_back(tmpAttackerDuckAmount[minThreadId][tmpRowId]);
                           result.weaponId.push_back(tmpWeaponId[minThreadId][tmpRowId]);
                           result.attackerNextPrimaryAttack.push_back(tmpNextPrimaryAttack[minThreadId][tmpRowId]);
                           result.attackerNextSecondaryAttack.push_back(tmpNextSecondaryAttack[minThreadId][tmpRowId]);
                           result.attackerGameTime.push_back(tmpAttackerGameTime[minThreadId][tmpRowId]);
                           result.weaponType.push_back(tmpWeaponType[minThreadId][tmpRowId]);
                       });
    return result;
}
