//
// Created by durst on 7/11/23.
//

#include "queries/moments/feature_store_team_extractor.h"
#include "bots/analysis/streaming_manager.h"
#include "file_helpers.h"
#include <atomic>

namespace csknow::feature_store {
    TeamFeatureStoreResult featureStoreTeamExtraction(const string & navPath, const nav_mesh::nav_file & navFile,
                                                      const std::vector<csknow::orders::QueryOrder> & orders,
                                                      const VisPoints & visPoints,
                                                      const DistanceToPlacesResult & distanceToPlaces,
                                                      const nearest_nav_cell::NearestNavCell & nearestNavCell,
                                                      const Players & players, const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                                      const WeaponFire & weaponFire, const Hurt & hurt,
                                                      const Plants & plants, const Defusals & defusals,
                                                      const csknow::key_retake_events::KeyRetakeEvents & keyRetakeEvents) {
        TeamFeatureStoreResult teamFeatureStoreResult(ticks.size, orders, ticks, keyRetakeEvents);
        int numThreads = omp_get_max_threads();
        std::atomic<int64_t> roundsProcessed = 0;
        vector<feature_store::FeatureStorePreCommitBuffer> tmpPreCommitBuffer(numThreads);
        string navFolderPath(std::filesystem::path(navPath).remove_filename().string());

#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            // skip rounds not on de_dust2, occasionally this happens due to wrong demo collection
            string mapName = games.mapName[rounds.gameId[roundIndex]];
            bool mapValid = mapName.find("de_dust2") != std::string::npos ||
                            mapName.find("DE_DUST2") != std::string::npos;

            RoundPlantDefusal roundPlantDefusal = processRoundPlantDefusals(rounds, ticks, plants, defusals, roundIndex);
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            StreamingManager streamingManager(navFolderPath);

            bool commitValidRowInRound = false;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 mapValid && tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                // clear history until commit a valid row
                // this ensures that history always starts fresh on first save row (aka ignoring teleports during setup, and history from before round starts)
                // so when deploy in retakes game without no history, match training data
                if (!commitValidRowInRound) {
                    tmpPreCommitBuffer[threadNum].clearHistory();
                }

                streamingManager.update(games, roundPlantDefusal, rounds, players, ticks, weaponFire, hurt,
                                        playerAtTick, tickIndex, nearestNavCell, visPoints, tickRates, false);
                const ServerState & curState = streamingManager.db.batchData.fromNewest();

                tmpPreCommitBuffer[threadNum].updateFeatureStoreBufferPlayers(curState);
                tmpPreCommitBuffer[threadNum].updateCurTeamData(curState, navFile);

                bool newCommitValidRowInRound =
                        teamFeatureStoreResult.commitTeamRow(tmpPreCommitBuffer[threadNum],
                                                             distanceToPlaces, navFile, roundIndex, tickIndex);
                commitValidRowInRound = commitValidRowInRound || newCommitValidRowInRound;
            }
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }

        return teamFeatureStoreResult;
    }
}
