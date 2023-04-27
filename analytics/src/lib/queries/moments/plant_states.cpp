//
// Created by durst on 4/27/23.
//

#include "queries/moments/plant_states.h"
#include "indices/build_indexes.h"


namespace csknow::plant_states {
    void PlantStatesResult::runQuery(const Rounds & rounds, const Ticks & ticks, const Plants & plants,
                                     const Defusals & defusals) {

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
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                for (const auto & [_0, _1, plantIndex] :
                    ticks.plantsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (plants.succesful[plantIndex]) {
                        plantId = plantIndex;
                    }
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
                           });
        vector<const int64_t *> foreignKeyCols{plantTickId.data(), roundEndTickId.data()};
        plantStatesPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}