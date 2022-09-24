//
// Created by durst on 9/18/22.
//

#include "queries/moments/trajectory_segments.h"
#include "queries/lookback.h"
#include "queries/rolling_window.h"
#include <omp.h>
#include <atomic>

struct SegmentData {
    int64_t segmentStartTickId;
    Vec2 segmentStart2DPos;
};

void finishSegment(vector<vector<int64_t>> & tmpSegmentStartTickId, vector<vector<int64_t>> & tmpSegmentEndTickId,
                   vector<vector<int64_t>> & tmpLength, vector<vector<int64_t>> & tmpPlayerId, vector<vector<string>> & tmpPlayerName,
                   vector<vector<Vec2>> & tmpSegmentStart2DPos, vector<vector<Vec2>> & tmpSegmentEnd2DPos,
                   int threadNum, int64_t tickIndex, int64_t playerId, int64_t patIndex,
                   const Players & players, const PlayerAtTick & playerAtTick, const SegmentData & sData,
                   map<int64_t, SegmentData> & playerToCurTrajectory, bool remove = true) {
    tmpSegmentStartTickId[threadNum].push_back(sData.segmentStartTickId);
    tmpSegmentEndTickId[threadNum].push_back(tickIndex);
    tmpLength[threadNum].push_back(tmpSegmentEndTickId[threadNum].back() - tmpSegmentStartTickId[threadNum].back() + 1);
    tmpPlayerId[threadNum].push_back(playerId);
    tmpPlayerName[threadNum].push_back(players.name[players.idOffset + playerId]);
    tmpSegmentStart2DPos[threadNum].push_back({sData.segmentStart2DPos});
    tmpSegmentEnd2DPos[threadNum].push_back({playerAtTick.posX[patIndex], playerAtTick.posY[patIndex]});
    if (remove) {
        playerToCurTrajectory.erase(playerId);
    }
}

TrajectorySegmentResult queryAllTrajectories(const Players & players, const Games & games, const Rounds & rounds,
                                             const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                             const NonEngagementTrajectoryResult & nonEngagementTrajectoryResult) {
    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpSegmentStartTickId(numThreads);
    vector<vector<int64_t>> tmpSegmentEndTickId(numThreads);
    vector<vector<int64_t>> tmpLength(numThreads);
    vector<vector<int64_t>> tmpPlayerId(numThreads);
    vector<vector<string>> tmpPlayerName(numThreads);
    vector<vector<Vec2>> tmpSegmentStart2DPos(numThreads);
    vector<vector<Vec2>> tmpSegmentEnd2DPos(numThreads);

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

        //TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<int64_t, SegmentData> playerToCurTrajectory;
        map<int64_t, int64_t> hi;


        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            map<int64_t, int64_t> curPlayerToPAT = getPATIdForPlayerId(ticks, playerAtTick, tickIndex);

            for (const auto & [_0, _1, trajectoryIndex] :
                    nonEngagementTrajectoryResult.trajectoriesPerTick.findOverlapping(tickIndex, tickIndex)) {
                int64_t curPlayerId = nonEngagementTrajectoryResult.playerId[trajectoryIndex];
                if (playerToCurTrajectory.find(curPlayerId) == playerToCurTrajectory.end()) {
                    if (curPlayerToPAT.find(curPlayerId) == curPlayerToPAT.end()) {
                        int x = 1;
                    }
                    int64_t curPATId = curPlayerToPAT[curPlayerId];
                    // probably not necessary, but just be defensive
                    if (curPATId < 0 || curPATId >= playerAtTick.size) {
                        int x = 1;
                    }
                    if (playerAtTick.isAlive[curPATId]) {
                        //playerToCurTrajectory[curPlayerId];
                        hi.insert({1, 1});
                        auto dude = playerToCurTrajectory.size();
                        //std::pair<int64_t , SegmentData> x = {curPlayerId, {tickIndex, {playerAtTick.posX[curPATId], playerAtTick.posY[curPATId]}}};
                        //playerToCurTrajectory.insert(x);
                        //playerToCurTrajectory[curPlayerId];
                        std::pair<int64_t , SegmentData> x = {curPlayerId, {tickIndex, {playerAtTick.posX[curPATId], playerAtTick.posY[curPATId]}}};
                        playerToCurTrajectory.insert(x);
                        /*
                        playerToCurTrajectory[curPlayerId] = {
                                tickIndex,
                                {playerAtTick.posX[curPATId], playerAtTick.posY[curPATId]}
                        };
                         */
                        //playerToCurTrajectory.insert({curPlayerId, {1}});
                        //playerToCurTrajectory.insert({curPlayerId, {tickIndex, {}}});
                        /*
                        playerToCurTrajectory[curPlayerId] = {
                                tickIndex, {}
                        };
                        playerToCurTrajectory[curPlayerId] = {
                                tickIndex,
                                {playerAtTick.posX[curPATId], playerAtTick.posY[curPATId]}
                        };
                         */
                    }

                }
            }

            /*
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                int64_t playerId = playerAtTick.playerId[patIndex];
                if (playerToCurTrajectory.find(playerId) != playerToCurTrajectory.end()) {
                    // write if dead or finished a segment
                    double secondsSinceSegmentStart =
                            secondsBetweenTicks(ticks, tickRates,
                                                playerToCurTrajectory.find(playerId)->second.segmentStartTickId,
                                                tickIndex);
                    if (!playerAtTick.isAlive[playerId] || secondsSinceSegmentStart > SEGMENT_SECONDS) {
                        finishSegment(tmpSegmentStartTickId, tmpSegmentEndTickId,
                                      tmpLength, tmpPlayerId, tmpPlayerName,
                                      tmpSegmentStart2DPos, tmpSegmentEnd2DPos,
                                      threadNum, tickIndex, playerId, patIndex,
                                      players, playerAtTick, playerToCurTrajectory[playerId],
                                      playerToCurTrajectory);
                    }
                }
            }
             */
        }

        int64_t maxTickInRound = rounds.ticksPerRound[roundIndex].maxId;
        map<int64_t, int64_t> endPlayerToPAT = getPATIdForPlayerId(ticks, playerAtTick, maxTickInRound);
        for (const auto & [playerId, tData] : playerToCurTrajectory) {
            /*
            finishSegment(tmpSegmentStartTickId, tmpSegmentEndTickId,
                          tmpLength, tmpPlayerId, tmpPlayerName,
                          tmpSegmentStart2DPos, tmpSegmentEnd2DPos,
                          threadNum, maxTickInRound, playerId, endPlayerToPAT[playerId],
                          players, playerAtTick, playerToCurTrajectory[playerId],
                          playerToCurTrajectory, false);
                          */
        }
        tmpRoundSizes[threadNum].push_back(tmpSegmentStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
        //roundsProcessed++;
        //printProgress((roundsProcessed * 1.0) / rounds.size);
    }

    TrajectorySegmentResult result;
    /*
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.segmentStartTickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.segmentStartTickId.push_back(tmpSegmentStartTickId[minThreadId][tmpRowId]);
                           result.segmentEndTickId.push_back(tmpSegmentEndTickId[minThreadId][tmpRowId]);
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                           result.playerName.push_back(tmpPlayerName[minThreadId][tmpRowId]);
                           result.segmentStart2DPos.push_back(tmpSegmentStart2DPos[minThreadId][tmpRowId]);
                           result.segmentEnd2DPos.push_back(tmpSegmentEnd2DPos[minThreadId][tmpRowId]);
                       });
                       */
    return result;


}
