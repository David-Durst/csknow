//
// Created by durst on 7/7/22.
//

#include "queries/moments/aggression_event.h"
#include "file_helpers.h"
#include <omp.h>
#include <atomic>

void assignRolesToPlayersPerTeam(const ReachableResult & reachableResult, vector<vector<int64_t>> * tmpPlayerId,
                                 vector<vector<AggressionRole>> * tmpRole, int threadNum,
                                 const map<int64_t, AreaId> & tAreas, const set<int64_t> & tVisiblePlayers,
                                 const set<AreaId> & tVisibleAreas, const nav_mesh::nav_file & navFile) {
    for (const auto & [tPlayerId, tAreaId] : tAreas) {
        tmpPlayerId[threadNum].back().push_back(tPlayerId);
        if (tVisiblePlayers.find(tPlayerId) != tVisiblePlayers.end()) {
            tmpRole[threadNum].back().push_back(AggressionRole::Pusher);
        }
        else {
            bool isBaiter = false;
            for (AreaId tVisibleArea : tVisibleAreas) {
                if (reachableResult.getDistance(tAreaId, tVisibleArea, navFile) <= MAX_BAITER_DISTANCE) {
                    tmpRole[threadNum].back().push_back(AggressionRole::Baiter);
                    isBaiter = true;
                }
            }
            if (!isBaiter) {
                tmpRole[threadNum].back().push_back(AggressionRole::Lurker);
            }
        }
    }
}

AggressionEventResult queryAggressionRoles(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                           const PlayerAtTick & playerAtTick,
                                           const nav_mesh::nav_file & navFile, const VisPoints & visPoints, const ReachableResult & reachableResult) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpStartTickId[numThreads];
    vector<int64_t> tmpEndTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<vector<int64_t>> tmpPlayerId[numThreads];
    vector<vector<AggressionRole>> tmpRole[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;
// for each round
        // for each player - identify current path - if in any regions of a path, or next region they will be in
        // for each time step in path - first player to shoot/be shot by enemy is pusher. baiter is everyone on same path
        // lurker is someone on different path
#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < std::min(6L, rounds.size); roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpStartTickId[threadNum].size());

        int64_t ticksSinceLastEngagement = 10000; // just some large number that will enable first engagement in each round
        // assuming first position is less than first kills
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            // compute areas for each alive player in tick
            map<int64_t, AreaId> tAreas, ctAreas;
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                if (playerAtTick.isAlive[patIndex]) {
                    AreaId areaId = navFile.get_nearest_area_by_position(
                            vec3Conv({playerAtTick.posX[patIndex], playerAtTick.posY[patIndex], playerAtTick.posZ[patIndex]}))
                                    .get_id();
                    if (playerAtTick.team[patIndex] == ENGINE_TEAM_T) {
                        tAreas[playerAtTick.playerId[patIndex]] = areaId;
                    }
                    else if (playerAtTick.team[patIndex] == ENGINE_TEAM_CT) {
                        ctAreas[playerAtTick.playerId[patIndex]] = areaId;
                    }
                }
            }

            // compute visible pairs of ts and cts
            bool anyVisiblePairs = false;
            set<int64_t> tVisiblePlayers, ctVisiblePlayers;
            set<AreaId> tVisibleAreas, ctVisibleAreas;
            for (const auto & [tPlayerId, tAreaId] : tAreas) {
                for (const auto & [ctPlayerId, ctAreaId] : ctAreas) {
                    if (visPoints.isVisibleAreaId(tAreaId, ctAreaId)) {
                        tVisiblePlayers.insert(tPlayerId);
                        tVisibleAreas.insert(tAreaId);
                        ctVisiblePlayers.insert(ctPlayerId);
                        ctVisibleAreas.insert(ctAreaId);
                        anyVisiblePairs = true;
                    }
                }
            }

            // determine if in engagement, if newly in one assign roles
            ticksSinceLastEngagement++;
            double secondsSinceLastEngagement = ticksSinceLastEngagement / games.gameTickRate[rounds.gameId[roundIndex]];
            if (anyVisiblePairs && secondsSinceLastEngagement > NOT_VISIBLE_END_SECONDS) {
                tmpStartTickId[threadNum].push_back(tickIndex);
                tmpEndTickId[threadNum].push_back(tickIndex);
                tmpLength[threadNum].push_back(1);
                tmpPlayerId[threadNum].push_back({});
                tmpRole[threadNum].push_back({});
                assignRolesToPlayersPerTeam(reachableResult, tmpPlayerId, tmpRole, threadNum, tAreas, tVisiblePlayers,
                                             tVisibleAreas, navFile);
                assignRolesToPlayersPerTeam(reachableResult, tmpPlayerId, tmpRole, threadNum, ctAreas, ctVisiblePlayers,
                                            ctVisibleAreas, navFile);
                ticksSinceLastEngagement = 0;
            }
            else if (anyVisiblePairs) {
                tmpEndTickId[threadNum].back() = tickIndex;
                tmpLength[threadNum].back() = tmpEndTickId[threadNum].back() - tmpStartTickId[threadNum].back() + 1;
                ticksSinceLastEngagement = 0;
            }
        }

        tmpRoundSizes[threadNum].push_back(tmpStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
        roundsProcessed++;
        printProgress((roundsProcessed * 1.0) / rounds.size);
    }

    AggressionEventResult result;
    vector<int64_t> roundsProcessedPerThread(numThreads, 0);
    while (true) {
        bool roundToProcess = false;
        int64_t minThreadId = -1;
        int64_t minRoundId = -1;
        for (int64_t threadId = 0; threadId < numThreads; threadId++) {
            if (roundsProcessedPerThread[threadId] < tmpRoundIds[threadId].size()) {
                roundToProcess = true;
                if (minThreadId == -1 || tmpRoundIds[threadId][roundsProcessedPerThread[threadId]] < minRoundId) {
                    minThreadId = threadId;
                    minRoundId = tmpRoundIds[minThreadId][roundsProcessedPerThread[minThreadId]];
                }

            }
        }
        if (!roundToProcess) {
            break;
        }
        result.rowIndicesPerRound.push_back({});
        result.rowIndicesPerRound[minRoundId].minId = result.startTickId.size();
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            result.startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
            result.endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
            result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
            result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
            result.role.push_back(tmpRole[minThreadId][tmpRowId]);
        }
        result.rowIndicesPerRound[minRoundId].maxId = result.startTickId.size() - 1;
        roundsProcessedPerThread[minThreadId]++;
    }
    result.size = result.startTickId.size();
    return result;
}

