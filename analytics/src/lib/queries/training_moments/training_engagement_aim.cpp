//
// Created by durst on 9/26/22.
//

#include "queries/training_moments/training_engagement_aim.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include <omp.h>

TrainingEngagementAimResult queryEngagementAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                               const PlayerAtTick & playerAtTick,
                                               const EngagementResult & engagementResult) {
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpTickId(numThreads);
    vector<vector<int64_t>> tmpEngagementId(numThreads);
    vector<vector<int64_t>> tmpAttackerPlayerId(numThreads);
    vector<vector<int64_t>> tmpVictimPlayerId(numThreads);
    vector<vector<array<Vec2, NUM_TICKS>>> tmpDeltaViewAngle(numThreads);
    vector<vector<array<double, NUM_TICKS>>> tmpEyeToHeadDistance(numThreads);

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
        rollingWindow.setTemporalRange(rounds.ticksPerRound[roundIndex].minId + NUM_TICKS - 1, tickRates,
                                       {DurationType::Ticks, 0, 0, NUM_TICKS - 1, 0});
        const PlayerToPATWindows & playerToPatWindows = rollingWindow.getWindows();

        for (int64_t tickIndex = rollingWindow.lastReadTickId();
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++, rollingWindow.readNextTick()) {
            for (const auto & [_0, _1, engagementIndex] :
                engagementResult.engagementsPerTick.findOverlapping(tickIndex, tickIndex)) {
                tmpTickId[threadNum].push_back(tickIndex);
                tmpEngagementId[threadNum].push_back(engagementIndex);

                int64_t attackerId = engagementResult.playerId[engagementIndex][0];
                tmpAttackerPlayerId[threadNum].push_back(attackerId);
                int64_t victimId = engagementResult.playerId[engagementIndex][1];
                tmpVictimPlayerId[threadNum].push_back(victimId);

                tmpDeltaViewAngle[threadNum].push_back({});
                tmpEyeToHeadDistance[threadNum].push_back({});

                for (size_t i = 0; i < NUM_TICKS; i++) {

                    const int64_t & attackerPATId = playerToPatWindows.at(attackerId).fromNewest(i);
                    const int64_t & victimPATId = playerToPatWindows.at(victimId).fromNewest(i);

                    Vec3 attackerEyePos = {
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

                    Vec2 curViewAngle = {
                        playerAtTick.viewX[attackerPATId],
                        playerAtTick.viewY[attackerPATId]
                    };

                    Vec2 deltaViewAngle = deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, curViewAngle);

                    tmpDeltaViewAngle[threadNum].back()[i] = deltaViewAngle;
                    tmpEyeToHeadDistance[threadNum].back()[i] = computeDistance(attackerEyePos, victimHeadPos);
                }
            }
        }

        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
    }

    TrainingEngagementAimResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.tickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                           result.engagementId.push_back(tmpEngagementId[minThreadId][tmpRowId]);
                           result.attackerPlayerId.push_back(tmpAttackerPlayerId[minThreadId][tmpRowId]);
                           result.victimPlayerId.push_back(tmpVictimPlayerId[minThreadId][tmpRowId]);
                           result.deltaViewAngle.push_back(tmpDeltaViewAngle[minThreadId][tmpRowId]);
                           result.eyeToHeadDistance.push_back(tmpEyeToHeadDistance[minThreadId][tmpRowId]);
                       });
    return result;
}
