//
// Created by durst on 9/18/22.
//

#include "queries/moments/trajectory_segments.h"
#include "queries/lookback.h"
#include "queries/rolling_window.h"
#include <omp.h>
#include <atomic>

struct TSData {
    int64_t trajectoryId, playerId;
    int64_t segmentStartTickId, segmentEndTickId;
    int64_t segmentStartPATId, segmentEndPATId;
};

void finishSegment(int64_t playerId, int64_t tickIndex, int64_t patId, map<int64_t, TSData> & playerToCurTrajectory,
                   vector<TSData> & finishedSegmentPerRound, bool remove = true) {
    TSData tsData = playerToCurTrajectory.at(playerId);
    tsData.segmentEndTickId = tickIndex;
    tsData.segmentEndPATId = patId;
    finishedSegmentPerRound.push_back(tsData);
    if (remove) {
        playerToCurTrajectory.erase(playerId);
    }
}

void recordSegments(vector<vector<int64_t>> & tmpTrajectoryId,
                    vector<vector<int64_t>> & tmpSegmentStartTickId, vector<vector<int64_t>> & tmpSegmentEndTickId,
                    vector<vector<int64_t>> & tmpLength, vector<vector<int64_t>> & tmpPlayerId, vector<vector<string>> & tmpPlayerName,
                    vector<vector<Vec3>> & tmpSegmentStart2DPos, vector<vector<Vec3>> & tmpSegmentEnd2DPos,
                    int threadNum, const Players & players, const PlayerAtTick & playerAtTick,
                    const vector<TSData> & finishedSegmentPerRound) {
    for (const auto & tsData : finishedSegmentPerRound) {
        tmpTrajectoryId[threadNum].push_back(tsData.trajectoryId);
        tmpSegmentStartTickId[threadNum].push_back(tsData.segmentStartTickId);
        tmpSegmentEndTickId[threadNum].push_back(tsData.segmentEndTickId);
        tmpLength[threadNum].push_back(tmpSegmentEndTickId[threadNum].back() - tmpSegmentStartTickId[threadNum].back() + 1);
        tmpPlayerId[threadNum].push_back(tsData.playerId);
        tmpPlayerName[threadNum].push_back(players.name[players.idOffset + tsData.playerId]);
        tmpSegmentStart2DPos[threadNum].push_back({playerAtTick.posX[tsData.segmentStartPATId],
                                                   playerAtTick.posY[tsData.segmentStartPATId],
                                                   playerAtTick.posZ[tsData.segmentStartPATId]});
        tmpSegmentEnd2DPos[threadNum].push_back({playerAtTick.posX[tsData.segmentEndPATId],
                                                 playerAtTick.posY[tsData.segmentEndPATId],
                                                 playerAtTick.posZ[tsData.segmentEndPATId]});
    }
}

TrajectorySegmentResult queryAllTrajectories(const Players & players, const Games & games, const Rounds & rounds,
                                             const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                             const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult) {
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpTrajectoryId(numThreads);
    vector<vector<int64_t>> tmpSegmentStartTickId(numThreads);
    vector<vector<int64_t>> tmpSegmentEndTickId(numThreads);
    vector<vector<int64_t>> tmpLength(numThreads);
    vector<vector<int64_t>> tmpPlayerId(numThreads);
    vector<vector<string>> tmpPlayerName(numThreads);
    vector<vector<Vec3>> tmpSegmentStart2DPos(numThreads);
    vector<vector<Vec3>> tmpSegmentEnd2DPos(numThreads);

    // for each round
    // for each tick
    // if a player is in trajectory, start a segment for them if no active segment
    // if a player is in a segment, end it if past segment time
    // clear out at end of round with early termination
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpSegmentStartTickId[threadNum].size()));

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<int64_t, TSData> playerToCurTrajectory;
        vector<TSData> finishedSegmentPerRound;
        RollingWindow rollingWindow(rounds, ticks, playerAtTick);

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId + 1;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            map<int64_t, int64_t> curPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex);
            map<int64_t, int64_t> priorPlayerToPAT = rollingWindow.getPATIdForPlayerId(tickIndex-1);

            // this tracks if a player is in a trajectory from non-engagemnet trajectory result
            // compared to internal playerToCurTrajectory/finishedSegmentPerRound to determine how to update internal
            // values
            set<int64_t> playerInTrajectory;

            for (const auto & [_0, _1, trajectoryIndex] :
                    nonEngagementTrajectoryResult.trajectoriesPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                int64_t curPlayerId = nonEngagementTrajectoryResult.playerId[trajectoryIndex];
                playerInTrajectory.insert(curPlayerId);
                if (playerToCurTrajectory.find(curPlayerId) == playerToCurTrajectory.end()) {
                    int64_t curPATId = curPlayerToPAT[curPlayerId];
                    // probably not necessary, but just be defensive
                    if (playerAtTick.isAlive[curPATId]) {
                        playerToCurTrajectory[curPlayerId] = {
                            trajectoryIndex, curPlayerId,
                            tickIndex, INVALID_ID,
                            curPATId, INVALID_ID
                        };
                    }

                }
            }

            // write if trajectory ended
            vector<int64_t> playerEndingTrajectory;
            for (const auto & [playerId, _] : playerToCurTrajectory) {
                if (playerInTrajectory.find(playerId) == playerInTrajectory.end()) {
                    playerEndingTrajectory.push_back(playerId);
                }
            }
            for (const auto & playerId : playerEndingTrajectory) {
                finishSegment(playerId, tickIndex-1, priorPlayerToPAT[playerId],
                              playerToCurTrajectory, finishedSegmentPerRound);

            }

            // write if dead or finished a segment
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                int64_t playerId = playerAtTick.playerId[patIndex];
                if (playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                    double secondsSinceSegmentStart =
                            secondsBetweenTicks(ticks, tickRates,
                                                playerToCurTrajectory.find(playerId)->second.segmentStartTickId,
                                                tickIndex);
                    if (!playerAtTick.isAlive[patIndex] || secondsSinceSegmentStart > SEGMENT_SECONDS) {
                        finishSegment(playerId, tickIndex-1, priorPlayerToPAT[playerId],
                                      playerToCurTrajectory, finishedSegmentPerRound);
                    }
                }
            }
        }

        // write all segments at end of round, then sort all in round and write actual results
        int64_t maxTickInRound = rounds.ticksPerRound[roundIndex].maxId;
        map<int64_t, int64_t> endPlayerToPAT = rollingWindow.getPATIdForPlayerId(maxTickInRound);
        for (const auto & [playerId, tData] : playerToCurTrajectory) {
            finishSegment(playerId, maxTickInRound, endPlayerToPAT[playerId],
                          playerToCurTrajectory, finishedSegmentPerRound, false);
        }
        std::sort(finishedSegmentPerRound.begin(), finishedSegmentPerRound.end(),
                  [](const TSData & a, const TSData & b) {
            return a.trajectoryId < b.trajectoryId ||
                (a.trajectoryId == b.trajectoryId && a.segmentStartTickId < b.segmentStartTickId);
        });
        recordSegments(tmpTrajectoryId,
                       tmpSegmentStartTickId, tmpSegmentEndTickId,
                       tmpLength, tmpPlayerId, tmpPlayerName,
                       tmpSegmentStart2DPos, tmpSegmentEnd2DPos,
                       threadNum, players, playerAtTick,
                       finishedSegmentPerRound);
        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpSegmentStartTickId[threadNum].size()) -
                                            tmpRoundStarts[threadNum].back());
        //roundsProcessed++;
        //printProgress((roundsProcessed * 1.0) / rounds.size);
    }

    TrajectorySegmentResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.segmentStartTickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.trajectoryId.push_back(tmpTrajectoryId[minThreadId][tmpRowId]);
                           result.segmentStartTickId.push_back(tmpSegmentStartTickId[minThreadId][tmpRowId]);
                           result.segmentEndTickId.push_back(tmpSegmentEndTickId[minThreadId][tmpRowId]);
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                           result.playerName.push_back(tmpPlayerName[minThreadId][tmpRowId]);
                           result.segmentStart2DPos.push_back(tmpSegmentStart2DPos[minThreadId][tmpRowId]);
                           result.segmentEnd2DPos.push_back(tmpSegmentEnd2DPos[minThreadId][tmpRowId]);
                       });
    return result;


}
