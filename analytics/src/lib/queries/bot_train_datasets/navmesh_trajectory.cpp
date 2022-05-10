//
// Created by durst on 3/24/22.
//

#include "queries/bot_train_dataset/navmesh_trajectory.h"
#include "queries/lookback.h"
#include "geometryNavConversions.h"
#include "bots/thinker.h"
#include <utility>
#include <cassert>

void
computeTrajectoryPerRound(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                          const nav_mesh::nav_file & navFile, const int64_t roundId,
                          map<int64_t, NavmeshTrajectoryResult::Trajectory> perRoundPlayerTrajectory) {
    // first key is shooter, second is target
    for (int64_t tickIndex = rounds.ticksPerRound[roundId].minId;
         tickIndex != -1 && tickIndex <= rounds.ticksPerRound[roundId].maxId; tickIndex++) {
        int64_t gameTickNumber = ticks.gameTickNumber[tickIndex];
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            int64_t playerId = playerAtTick.playerId[patIndex];
            const nav_mesh::nav_area & navMeshArea = navFile.get_nearest_area_by_position(vec3Conv({
                playerAtTick.posX[patIndex],
                playerAtTick.posY[patIndex],
                playerAtTick.posZ[patIndex]
            }));
            // update end every tick, so when die no need to do further processing
            if (playerAtTick.isAlive[patIndex]) {
                if (perRoundPlayerTrajectory.find(playerId) == perRoundPlayerTrajectory.end()) {
                    perRoundPlayerTrajectory[playerId].target = NavmeshTrajectoryResult::TrajectoryTarget::NOT_YET_KNOWN;
                    perRoundPlayerTrajectory[playerId].startEndTickIds = {tickIndex, tickIndex};
                    perRoundPlayerTrajectory[playerId].startEndGameTickNumbers = {gameTickNumber, gameTickNumber};
                    perRoundPlayerTrajectory[playerId].navMeshArea = {navMeshArea.get_id()};
                    perRoundPlayerTrajectory[playerId].navMeshPlace = {navMeshArea.m_place};
                    perRoundPlayerTrajectory[playerId].areaEntryPATId = {patIndex};
                }
                else {
                    perRoundPlayerTrajectory[playerId].startEndTickIds.maxId = tickIndex;
                    perRoundPlayerTrajectory[playerId].startEndGameTickNumbers.maxId = gameTickNumber;
                    if (perRoundPlayerTrajectory[playerId].navMeshArea.back() != navMeshArea.get_id()) {
                        perRoundPlayerTrajectory[playerId].navMeshArea.push_back(navMeshArea.get_id());
                        perRoundPlayerTrajectory[playerId].navMeshPlace.push_back(navMeshArea.m_place);
                        perRoundPlayerTrajectory[playerId].navMeshPlaceName.push_back(navFile.get_place(navMeshArea.m_place));
                        perRoundPlayerTrajectory[playerId].areaEntryPATId.push_back(patIndex);
                    }
                }
            }

        }
    }
}


NavmeshTrajectoryResult queryNavmeshTrajectoryDataset(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks, const Players & players,
                                                      const PlayerAtTick & playerAtTick,
                                                      const std::map<std::string, const nav_mesh::nav_file> & mapNavs) {
    int numThreads = omp_get_max_threads();
    // stored per trajectory
    vector<int64_t> tmpSourcePlayerId[numThreads];
    vector<string> tmpSourcePlayerName[numThreads];
    vector<string> tmpDemoName[numThreads];
    vector<NavmeshTrajectoryResult::Trajectory> tmpTrajectory[numThreads];
    // stored once for all tracjectories in a round
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        string mapName = games.mapName[rounds.gameId[roundIndex]];
        const nav_mesh::nav_file & navFile = mapNavs.at(mapName);

        // find all the bombsite A and B locations
        vector<uint32_t> aLocations, bLocations;
        for (const auto & navMeshArea : navFile.m_areas) {
            if (navFile.get_place(navMeshArea.m_place) == "BombsiteA") {
                aLocations.push_back(navMeshArea.get_id());
            }
            else if (navFile.get_place(navMeshArea.m_place) == "BombsiteB") {
                bLocations.push_back(navMeshArea.get_id());
            }
        }

        // compute all the per round trajectories
        map<int64_t, NavmeshTrajectoryResult::Trajectory> perRoundPlayerTrajectory;
        computeTrajectoryPerRound(rounds, ticks, playerAtTick, navFile, roundIndex, perRoundPlayerTrajectory);
    }


    NavmeshTrajectoryResult result;
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
        int64_t startTrajectoryEntry = result.roundId.size();
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            result.roundId.push_back(minRoundId);
            result.sourcePlayerId.push_back(tmpSourcePlayerId[minThreadId][tmpRowId]);
            result.sourcePlayerName.push_back(tmpSourcePlayerName[minThreadId][tmpRowId]);
            result.demoName.push_back(tmpDemoName[minThreadId][tmpRowId]);
            result.trajectory.push_back(tmpTrajectory[minThreadId][tmpRowId]);
        }
        result.trajectoryPerRound.push_back({startTrajectoryEntry, static_cast<int64_t>(result.roundId.size())});
        roundsProcessedPerThread[minThreadId]++;
    }
    result.size = result.roundId.size();
    return result;
}