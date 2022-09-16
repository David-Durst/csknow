//
// Created by durst on 9/15/22.
//
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/lookback.h"
#include "queries/base_tables.h"
#include "queries/rolling_window.h"
#include <omp.h>
#include <atomic>

struct TrajectoryData {
    int64_t startTickId;
};

NonEngagementTrajectoryResult queryNonEngagementTrajectory(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                                           const PlayerAtTick & playerAtTick,
                                                           const EngagementResult & engagementResult) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpStartTickId[numThreads];
    vector<int64_t> tmpEndTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<int64_t> tmpPlayerId[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;
    double maxSpeed = -1;

    // for each round
    // for each tick
    // record all players in engagement
    // if a player is in engagement, then terminate their trajectory (if one was in progress) without recording
    // otherwise, if player has 0 velocity start trajectory and record old one (if one was start)
    // no need to clear out at end of round, as just drop never terminated trajectories
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpStartTickId[threadNum].size());

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        double stoppedPerTickSpeed = perSecondRateToPerDemoTickRate(tickRates, STOPPED_SPEED_THRESHOLD);
        double startPerTickSpeed = perSecondRateToPerDemoTickRate(tickRates, START_SPEED_THRESHOLD);

        map<int64_t, TrajectoryData> playerToCurTrajectory;

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {


            for (const auto & [_0, _1, engagementIndex] :
                    engagementResult.engagementsPerTick.findOverlapping(tickIndex, tickIndex)) {
                for (const auto playerId : engagementResult.playerId[engagementIndex]) {
                    if (playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                        playerToCurTrajectory.erase(playerId);
                    }
                }
            }


            if (tickIndex - 1 >= rounds.ticksPerRound[roundIndex].minId) {
                map<int64_t, int64_t> tminus1PlayerToPAT = getPATIdForPlayerId(ticks, playerAtTick, tickIndex - 1);

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t playerId = playerAtTick.playerId[patIndex];
                    int64_t tminus1PatIndex = tminus1PlayerToPAT[playerId];
                    Vec3 tminus1Pos = {playerAtTick.posX[tminus1PatIndex], playerAtTick.posY[tminus1PatIndex],
                                       playerAtTick.posZ[tminus1PatIndex]};
                    Vec3 curPos = {playerAtTick.posX[patIndex], playerAtTick.posY[patIndex],
                                   playerAtTick.posZ[patIndex]};
                    double speed = computeMagnitude(curPos - tminus1Pos);
                    maxSpeed = std::max(speed, maxSpeed);
                    if (speed < stoppedPerTickSpeed && playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                        tmpStartTickId[threadNum].push_back(playerToCurTrajectory[playerId].startTickId);
                        tmpEndTickId[threadNum].push_back(tickIndex);
                        tmpLength[threadNum].push_back(tmpEndTickId[threadNum].back() - tmpStartTickId[threadNum].back() + 1);
                        tmpPlayerId[threadNum].push_back(playerId);
                        playerToCurTrajectory.erase(playerId);
                    }
                    else if (speed > startPerTickSpeed && playerToCurTrajectory.find(playerId) == playerToCurTrajectory.end()) {
                        playerToCurTrajectory[playerId] = {tickIndex};
                    }
                }
            }
        }
        tmpRoundSizes[threadNum].push_back(tmpStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
    }

    NonEngagementTrajectoryResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.startTickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                           result.endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                       });
    return result;
}
