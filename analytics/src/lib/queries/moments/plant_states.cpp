//
// Created by durst on 4/27/23.
//

#include "queries/moments/plant_states.h"
#include "indices/build_indexes.h"


namespace csknow::plant_states {
    void PlantStatesResult::runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                     const Plants & plants, const Defusals & defusals) {

        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpPlantTickId(numThreads);
        vector<vector<int64_t>> tmpRoundEndTickId(numThreads);
        vector<vector<int64_t>> tmpLength(numThreads);
        vector<vector<int64_t>> tmpRoundId(numThreads);
        vector<vector<int64_t>> tmpPlantId(numThreads);
        vector<vector<int64_t>> tmpDefusalId(numThreads);
        vector<vector<Vec3>> tmpC4Pos(numThreads);
        vector<vector<TeamId>> tmpWinnerTam(numThreads);
        vector<vector<bool>> tmpC4Defused(numThreads);
        vector<array<PlayerState, max_players_per_team>> tmpCTPlayerStates, tmpTPlayerStates;

        // for each round
        // track events for each pairs of player.
        // start a new event for a pair when hurt event with no prior one or far away prior one
        // clear out all hurt events on end of round
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpPlantTickId[threadNum].size()));
            int64_t plantId = INVALID_ID;
            int64_t defusalId = INVALID_ID;

            map<int64_t, int64_t> latentEventIndexToTmpEngagementIndex;
            bool foundFirstPlantInRound = false;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                bool curTickIsPlant = false;
                for (const auto & [_0, _1, plantIndex] :
                    ticks.plantsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (plants.succesful[plantIndex]) {
                        plantId = plantIndex;
                        curTickIsPlant = true;
                    }
                }

                // don't add multiple player states per round
                if (curTickIsPlant && !foundFirstPlantInRound) {
                    for (size_t i = 0; i < max_players_per_team; i++) {
                        tmpCTPlayerStates[threadNum][i].alive.push_back(false);
                        tmpTPlayerStates[threadNum][i].alive.push_back(false);
                        tmpCTPlayerStates[threadNum][i].pos.push_back({0., 0., 0.});
                        tmpTPlayerStates[threadNum][i].viewAngle.push_back({0., 0.});
                    }

                    size_t ctPlayerIndex = 0, tPlayerIndex = 0;
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        if (playerAtTick.isAlive[patIndex]) {
                            Vec3 playerPos = {
                                playerAtTick.posX[patIndex],
                                playerAtTick.posY[patIndex],
                                playerAtTick.posZ[patIndex]
                            };
                            Vec2 playerViewAngle = {
                                playerAtTick.viewX[patIndex],
                                playerAtTick.viewY[patIndex]
                            };
                            if (playerAtTick.team[patIndex] == ENGINE_TEAM_CT) {
                                tmpCTPlayerStates[threadNum][ctPlayerIndex].alive.back() = true;
                                tmpCTPlayerStates[threadNum][ctPlayerIndex].pos.back() = playerPos;
                                tmpCTPlayerStates[threadNum][ctPlayerIndex].viewAngle.back() = playerViewAngle;
                                ctPlayerIndex++;
                            }
                            else if (playerAtTick.team[patIndex] == ENGINE_TEAM_T) {
                                tmpTPlayerStates[threadNum][tPlayerIndex].alive.back() = true;
                                tmpTPlayerStates[threadNum][tPlayerIndex].pos.back() = playerPos;
                                tmpTPlayerStates[threadNum][tPlayerIndex].viewAngle.back() = playerViewAngle;
                                tPlayerIndex++;
                            }
                        }

                    }
                }

                if (curTickIsPlant) {
                    foundFirstPlantInRound = true;
                }

                for (const auto & [_0, _1, defusalIndex] :
                    ticks.defusalsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (defusals.succesful[defusalIndex]) {
                        defusalId = defusalIndex;
                    }
                }
            }

            if (plantId != INVALID_ID) {
                int64_t curPlantTickId = plants.endTick[plantId];
                tmpPlantTickId[threadNum].push_back(curPlantTickId);
                int64_t curRoundEndTickId = rounds.endTick[roundIndex];
                tmpRoundEndTickId[threadNum].push_back(curRoundEndTickId);
                tmpLength[threadNum].push_back(curRoundEndTickId - curPlantTickId + 1);
                tmpRoundId[threadNum].push_back(roundIndex);
                tmpPlantId[threadNum].push_back(plantId);
                tmpDefusalId[threadNum].push_back(defusalId);
                tmpWinnerTam[threadNum].push_back(rounds.winner[roundIndex]);
                tmpC4Defused[threadNum].push_back(defusalId != INVALID_ID);
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpPlantTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }

        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           plantTickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               plantTickId.push_back(tmpPlantTickId[minThreadId][tmpRowId]);
                               roundEndTickId.push_back(tmpRoundEndTickId[minThreadId][tmpRowId]);
                               tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                               roundId.push_back(tmpRoundId[minThreadId][tmpRowId]);
                               plantId.push_back(tmpPlantId[minThreadId][tmpRowId]);
                               defusalId.push_back(tmpDefusalId[minThreadId][tmpRowId]);
                               winnerTeam.push_back(tmpWinnerTam[minThreadId][tmpRowId]);
                               c4Defused.push_back(tmpC4Defused[minThreadId][tmpRowId]);
                               for (size_t i = 0; i < max_players_per_team; i++) {
                                   ctPlayerStates[i].alive.push_back(tmpCTPlayerStates[minThreadId][i].alive[tmpRowId]);
                                   ctPlayerStates[i].pos.push_back(tmpCTPlayerStates[minThreadId][i].pos[tmpRowId]);
                                   ctPlayerStates[i].viewAngle.push_back(tmpCTPlayerStates[minThreadId][i].viewAngle[tmpRowId]);
                                   tPlayerStates[i].alive.push_back(tmpTPlayerStates[minThreadId][i].alive[tmpRowId]);
                                   tPlayerStates[i].pos.push_back(tmpTPlayerStates[minThreadId][i].pos[tmpRowId]);
                                   tPlayerStates[i].viewAngle.push_back(tmpTPlayerStates[minThreadId][i].viewAngle[tmpRowId]);
                               }
                           });
        vector<const int64_t *> foreignKeyCols{plantTickId.data(), roundEndTickId.data()};
        plantStatesPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}