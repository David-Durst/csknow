//
// Created by durst on 2/27/23.
//

#include "queries/grenade/player_flashed.h"
#include "indices/build_indexes.h"
#include "queries/lookback.h"
#include <omp.h>
#include <map>

namespace csknow::player_flashed {
    struct FlashedState {
        int64_t lastFlashTickId;
        double timeRemainingAtLastFlashTime;
    };

    void PlayerFlashedResult::runQuery(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick, const Flashed & flashed) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpTickId(numThreads);
        vector<vector<int64_t>> tmpVictimId(numThreads);
        vector<vector<double>> tmpFlashAmount(numThreads);

//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            std::map<int64_t, FlashedState> playerToFlashState;

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

                std::set<int64_t> playerFlashedThisTick;
                bool newFlashedThisTick =
                    !ticks.flashedPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex).empty();

                if (newFlashedThisTick) {
                    std::map<int64_t, double> playerToFlashTimeRemaining;
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        playerToFlashTimeRemaining[playerAtTick.playerId[patIndex]] =
                            playerAtTick.flashDuration[patIndex];
                    }

                    for (const auto & [_0, _1, flashIndex] :
                        ticks.flashedPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {

                        playerToFlashState[flashed.victim[flashIndex]] = {
                            tickIndex,
                            playerToFlashTimeRemaining[flashed.victim[flashIndex]]
                        };
                    }
                }

                vector<int64_t> noLongerFlashedPlayers;
                for (const auto & [playerId, flashedState] : playerToFlashState) {
                    // 3 is magic constant from CSGO
                    double timeSinceFlash = secondsBetweenTicks(ticks, tickRates, flashedState.lastFlashTickId,
                                                                tickIndex);
                    double timeLeft = flashedState.timeRemainingAtLastFlashTime - timeSinceFlash;
                    if (timeLeft >= 0.) {
                        tmpTickId[threadNum].push_back(tickIndex);
                        tmpVictimId[threadNum].push_back(playerId);
                        if (timeLeft >= 3.)  {
                            tmpFlashAmount[threadNum].push_back(1.);
                        }
                        else {
                            tmpFlashAmount[threadNum].push_back(timeLeft / 3.);
                        }
                    }
                    else {
                        noLongerFlashedPlayers.push_back(playerId);

                    }
                }

                for (const auto & playerId : noLongerFlashedPlayers) {
                    playerToFlashState.erase(playerId);
                }
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }
        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           tickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                               victimId.push_back(tmpVictimId[minThreadId][tmpRowId]);
                               flashAmount.push_back(tmpFlashAmount[minThreadId][tmpRowId]);
                           });
        vector<const int64_t *> foreignKeyCols{tickId.data()};
        playerFlashedPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}
