//
// Created by durst on 9/11/22.
//

#include "queries/moments/engagement_per_tick_aim.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include <omp.h>
#include <atomic>

EngagementPerTickAimResult queryEngagementPerTickAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                     const PlayerAtTick & playerAtTick, const WeaponFire & first,
                                                     const Hurt & hurt, const EngagementResult & engagementResult) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpTickId[numThreads];
    vector<int64_t> tmpEngagementId[numThreads];
    vector<int64_t> tmpAttackerPlayerId[numThreads];
    vector<int64_t> tmpVictimPlayerId[numThreads];
    vector<double> tmpSecondsToHit[numThreads];
    vector<Vec2> tmpDeltaViewAngle[numThreads];
    vector<Vec2> tmpRawViewAngleVelocity[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;

    // for each round
    // for each tick
    // for each engagement in each tick
    // record where supposed to aim vs where aiming and if it's a fire/hit
    // clear out all hurt events on end of round
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpTickId[threadNum].size());

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            map<int64_t, int64_t> curPlayerToPAT = getPATIdForPlayerId(ticks, playerAtTick, tickIndex);

            // for view angle velocity, only defined if there's a prior tick in the round
            map<int64_t, int64_t> priorPlayerToPAT;
            if (tickIndex > rounds.ticksPerRound[roundIndex].minId) {
                priorPlayerToPAT = getPATIdForPlayerId(ticks, playerAtTick, tickIndex - 1);
            }

            for (const auto & [_0, _1, engagementIndex] :
                engagementResult.engagementsPerTick.findOverlapping(tickIndex, tickIndex)) {

                tmpTickId[threadNum].push_back(tickIndex);
                tmpEngagementId[threadNum].push_back(engagementIndex);

                int64_t attackerId = engagementResult.playerId[engagementIndex][0];
                tmpAttackerPlayerId[threadNum].push_back(attackerId);
                int64_t victimId = engagementResult.playerId[engagementIndex][1];
                tmpVictimPlayerId[threadNum].push_back(victimId);

                // find the nearest hit in the future
                int64_t soonestHurtId = INVALID_ID;
                double secondsToSoonestHurtId = std::numeric_limits<double>::max();
                for (const auto hurtId : engagementResult.hurtIds[engagementIndex]) {
                    double secondsToHurt = secondsBetweenTicks(ticks, tickRates, tickIndex, hurt.tickId[hurtId]);
                    if (secondsToHurt > 0 && secondsToHurt < secondsToSoonestHurtId) {
                        soonestHurtId = hurtId;
                        secondsToSoonestHurtId = secondsToHurt;
                    }
                }
                tmpSecondsToHit[threadNum].push_back(secondsToSoonestHurtId);

                // compute ideal view angle at time of hurt
                // should account for network latency in future, as attack aim at enemy's position in past
                // player stands at CL_INTERP back and sees enemy and CL_INTERP * 2 + LAG back
                map<int64_t, int64_t> nextHurtPlayerToPAT =
                        getPATIdForPlayerId(ticks, playerAtTick, hurt.tickId[soonestHurtId]);

                Vec3 attackerEyePos = {
                        playerAtTick.posX[nextHurtPlayerToPAT[attackerId]],
                        playerAtTick.posY[nextHurtPlayerToPAT[attackerId]],
                        playerAtTick.eyePosZ[nextHurtPlayerToPAT[attackerId]]
                };

                Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(
                        {
                            playerAtTick.posX[nextHurtPlayerToPAT[victimId]],
                            playerAtTick.posY[nextHurtPlayerToPAT[victimId]],
                            playerAtTick.eyePosZ[nextHurtPlayerToPAT[victimId]]
                        },
                        {
                            playerAtTick.viewX[nextHurtPlayerToPAT[victimId]],
                            playerAtTick.viewY[nextHurtPlayerToPAT[victimId]]
                        },
                        playerAtTick.duckAmount[nextHurtPlayerToPAT[victimId]]
                );

                Vec3 targetVector = victimHeadPos - attackerEyePos;
                Vec2 targetViewAngle = vectorAngles(targetVector);
                // want to be + if too large, - if too small, so do current - target
                Vec2 deltaViewAngle = {
                        playerAtTick.viewX[curPlayerToPAT[attackerId]] - targetViewAngle.x,
                        playerAtTick.viewY[curPlayerToPAT[attackerId]] - targetViewAngle.y,
                };
                tmpDeltaViewAngle[threadNum].push_back(deltaViewAngle);

                // compute view angle velocity if there is a prior tick in the round
                if (tickIndex > rounds.ticksPerRound[roundIndex].minId) {
                    tmpRawViewAngleVelocity[threadNum].push_back({
                        playerAtTick.viewX[curPlayerToPAT[attackerId]] - playerAtTick.viewX[priorPlayerToPAT[attackerId]],
                        playerAtTick.viewY[curPlayerToPAT[attackerId]] - playerAtTick.viewY[priorPlayerToPAT[attackerId]],
                    });
                }
                else {
                    tmpRawViewAngleVelocity[threadNum].push_back({0., 0.});
                }
            }
        }

        tmpRoundSizes[threadNum].push_back(tmpTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
    }

    EngagementPerTickAimResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.tickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                           result.engagementId.push_back(tmpEngagementId[minThreadId][tmpRowId]);
                           result.attackerPlayerId.push_back(tmpAttackerPlayerId[minThreadId][tmpRowId]);
                           result.victimPlayerId.push_back(tmpVictimPlayerId[minThreadId][tmpRowId]);
                           result.secondsToHit.push_back(tmpSecondsToHit[minThreadId][tmpRowId]);
                           result.deltaViewAngle.push_back(tmpDeltaViewAngle[minThreadId][tmpRowId]);
                           result.rawViewAngleVelocity.push_back(tmpRawViewAngleVelocity[minThreadId][tmpRowId]);
                       });
    return result;
}
