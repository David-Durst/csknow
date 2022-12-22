//
// Created by durst on 9/11/22.
//

#include "queries/moments/engagement_per_tick_aim.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include <omp.h>
#include <atomic>

EngagementPerTickAimResult queryEngagementPerTickAim(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                     const EngagementResult & engagementResult,
                                                     const TrainingEngagementAimResult & trainingEngagementAimResult) {

    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpTickId(numThreads);
    vector<vector<int64_t>> tmpEngagementId(numThreads);
    vector<vector<int64_t>> tmpAttackerPlayerId(numThreads);
    vector<vector<int64_t>> tmpVictimPlayerId(numThreads);
    vector<vector<double>> tmpSecondsToHit(numThreads);
    vector<vector<Vec2>> tmpDeltaViewAngle(numThreads);
    vector<vector<double>> tmpRawViewAngleSpeed(numThreads);
    vector<vector<double>> tmpSecondsSinceEngagementStart(numThreads);

    // for each round
    // for each tick
    // for each engagement in each tick
    // record where supposed to aim vs where aiming and if it's a fire/hit
    // clear out all hurt events on end of round
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        for (int64_t engagementAimIndex = trainingEngagementAimResult.rowIndicesPerRound[roundIndex].minId;
             engagementAimIndex <= trainingEngagementAimResult.rowIndicesPerRound[roundIndex].maxId;
             engagementAimIndex++) {

            int64_t tickIndex = trainingEngagementAimResult.tickId[engagementAimIndex];
            tmpTickId[threadNum].push_back(tickIndex);
            int64_t engagementIndex = trainingEngagementAimResult.engagementId[engagementAimIndex];
            tmpEngagementId[threadNum].push_back(engagementIndex);

            tmpAttackerPlayerId[threadNum].push_back(trainingEngagementAimResult.attackerPlayerId[engagementAimIndex]);
            tmpVictimPlayerId[threadNum].push_back(trainingEngagementAimResult.victimPlayerId[engagementAimIndex]);
            // fix this later if ever use it
            tmpSecondsToHit[threadNum].push_back(INVALID_ID);
            tmpDeltaViewAngle[threadNum].push_back(
                trainingEngagementAimResult.deltaRelativeFirstHeadViewAngle[engagementAimIndex][PAST_AIM_TICKS]);


            tmpSecondsSinceEngagementStart[threadNum].push_back(
                secondsBetweenTicks(ticks, tickRates, engagementResult.startTickId[engagementIndex],
                                    tickIndex));

            Vec2 tminus2ViewAngle =
                trainingEngagementAimResult.attackerViewAngle[engagementAimIndex][PAST_AIM_TICKS-2];
            Vec2 tminus1ViewAngle =
                trainingEngagementAimResult.attackerViewAngle[engagementAimIndex][PAST_AIM_TICKS-1];
            Vec2 tViewAngle =
                trainingEngagementAimResult.attackerViewAngle[engagementAimIndex][PAST_AIM_TICKS];
            Vec2 tplus1ViewAngle =
                trainingEngagementAimResult.attackerViewAngle[engagementAimIndex][PAST_AIM_TICKS+1];
            double minus1Speed = computeMagnitude(tminus1ViewAngle - tminus2ViewAngle);
            double curSpeed = computeMagnitude(tViewAngle - tminus1ViewAngle);
            double plus1Speed = computeMagnitude(tplus1ViewAngle - tViewAngle);
            vector<double> speeds{minus1Speed, curSpeed, plus1Speed};
            std::sort(speeds.begin(), speeds.end());
            tmpRawViewAngleSpeed[threadNum].push_back(speeds[1]);
        }

        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
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
                           result.rawViewAngleSpeed.push_back(tmpRawViewAngleSpeed[minThreadId][tmpRowId]);
                           result.secondsSinceEngagementStart.push_back(tmpSecondsSinceEngagementStart[minThreadId][tmpRowId]);
                       });
    return result;
}
