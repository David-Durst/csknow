//
// Created by durst on 9/18/22.
//

#include "queries/moments/trajectory_segments.h"
#include "queries/lookback.h"
#include "queries/rolling_window.h"
#include <omp.h>
#include <atomic>

struct TrajectoryData {
    int64_t segmentStartTickId;
    Vec2 segmentStart2DPos;
};

TrajectorySegmentResult queryAllTrajectories(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                             const PlayerAtTick & playerAtTick,
                                             const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpSegmentStartTickId[numThreads];
    vector<int64_t> tmpSegmentEndTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<int64_t> tmpPlayerId[numThreads];
    vector<string> tmpPlayerName[numThreads];
    vector<Vec2> tmpSegmentStart2DPos;
    vector<Vec2> tmpSegmentEnd2DPos;
    std::atomic<int64_t> roundsProcessed = 0;

    // for each round
    // for each tick
    // if a player is in trajectory, start a segment for them if no active segment
    // if a player is in a segment, end it if past segment time
    // clear out at end of round with early termination
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpSegmentStartTickId[threadNum].size());

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<int64_t, TrajectoryData> playerToCurTrajectory;


        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            map<int64_t, int64_t> curPlayerToPAT = getPATIdForPlayerId(ticks, playerAtTick, tickIndex);

            for (const auto & [_0, _1, trajectoryIndex] :
                    nonEngagementTrajectoryResult.trajectoriesPerTick.findOverlapping(tickIndex, tickIndex)) {
                int64_t curPlayerId = nonEngagementTrajectoryResult.playerId[trajectoryIndex];
                if (playerToCurTrajectory.find(curPlayerId) != playerToCurTrajectory.end()) {
                    int64_t curPATId = curPlayerToPAT[curPlayerId];
                    // probably not necessary, but just be defensive
                    if (playerAtTick.isAlive[curPATId]) {
                        playerToCurTrajectory[nonEngagementTrajectoryResult.playerId[trajectoryIndex]] = {
                                tickIndex,
                                {playerAtTick.posX[curPATId], playerAtTick.posY[curPATId]}
                        };
                    }
                }
            }

            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                int64_t playerId = playerAtTick.playerId[patIndex];
                // write if dead
                if (playerAtTick.isAlive[playerId] && playerToCurTrajectory.find(playerId) == playerToCurTrajectory.end() &&
                    inEngagement.find(playerId) == inEngagement.end()) {
                    playerToCurTrajectory[playerId] = {tickIndex};
                }
            }
        }

        for (const auto [playerId, tData] : playerToCurTrajectory) {
            finishEngagement(tmpStartTickId, tmpEndTickId, tmpLength, tmpPlayerId, threadNum, rounds.ticksPerRound[roundIndex].maxId,
                             playerId, playerToCurTrajectory.find(playerId)->second, playerToCurTrajectory, false);

        }
        tmpRoundSizes[threadNum].push_back(tmpStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
        //roundsProcessed++;
        //printProgress((roundsProcessed * 1.0) / rounds.size);
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
